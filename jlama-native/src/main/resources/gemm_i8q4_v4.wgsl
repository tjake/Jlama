struct Params {
    m: u32,        // Number of rows in submatrix C (Expected to be 1 mostly)
    n: u32,        // Number of columns in submatrix C
    k: u32,        // Shared dimension
    lda: u32,      // Leading dimension of A (elements)
    ldb: u32,      // Leading dimension of B (elements)
    ldc: u32,      // Leading dimension of C (elements)
};

// Bindings remain the same
@group(0) @binding(0) var<storage, read> A: array<u32>;       // Signed i8 quantized A (1xK)
@group(0) @binding(1) var<storage, read> A2: array<f32>;      // Scales for A (per BLOCK_SIZE)
@group(0) @binding(2) var<storage, read> B: array<u32>;       // Unsigned u4 quantized B (KxN?)
@group(0) @binding(3) var<storage, read> B2: array<f32>;      // Scales for B (per BLOCK_SIZE)
@group(0) @binding(4) var<storage, read_write> C: array<f32>; // Output C (1xN)
@group(0) @binding(5) var<uniform> params: Params;

// Constants
const RN = 8u;          // Columns per workgroup tile (Adjust based on hardware)
const BLOCK_SIZE = 32u; // Inner loop dimension / quantization block size
const TILE_B_HEIGHT_U32 = BLOCK_SIZE / 8u; // 32 nibbles vertically / 8 nibbles per u32 = 4
const TILE_B_TILE = TILE_B_HEIGHT_U32 * RN;

// --- Shared Memory ---
// Tile for B: Stores BLOCK_SIZE rows (k-dim) x RN cols (n-dim) of u4 nibbles
// Each u32 stores 8 nibbles (2 per byte). Size = (32 * RN) / 8 = 4 * RN u32s
var<workgroup> tileB: array<u32, TILE_B_TILE>; // e.g., 4 * 8 = 32 u32s
// Tile for B scales: Stores RN scale values for the columns this workgroup handles
var<workgroup> tileB2: array<f32, RN>; // e.g., 8 f32s

// Helper function to extract an 8-bit value from a u32,
// interpret it as a signed byte (-128 to 127), and return it as f32.
fn i8_to_f32(byte_u: u32) -> f32 {
    let byte_i = i32(byte_u);
    let signed_val_i32 = select(byte_i, byte_i - 256, byte_u > 127u);
    return f32(signed_val_i32);
}

// --- Compute Shader ---
// Workgroup size optimized for M=1: Process RN columns per workgroup.
// Each thread handles one column jj.
@compute @workgroup_size(RN, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Global column index computed by this thread
    let lx = local_id.x; // Local column index (0..RN-1)
    let jj = workgroup_id.x * RN + lx;

    // Bounds check for the output column this thread handles
    let thread_in_bounds = (jj < params.n);

    // Leading dimensions in terms of element counts for indexing
    // ldan is not used when reading A[0, k] directly
    // ldas is used for A2 scale index
    let ldas = params.lda / BLOCK_SIZE;
    let ldbn = params.ldb / 8u; // Stride between columns of B in u32s
    let ldbs = params.ldb / BLOCK_SIZE; // Stride between columns of B2

    var sum: f32 = 0.0; // Accumulator for C[0, jj]

    // Loop over the K dimension in blocks of BLOCK_SIZE
    for (var k = 0u; k < params.k; k = k + BLOCK_SIZE) {

        // --- Load B and B2 tiles into Shared Memory ---
        // Each thread lx loads the data for its column jj

        // Load B tile data (4 u32s per thread/column)
        let load_global_jj_b = workgroup_id.x * RN + lx;
        if (load_global_jj_b < params.n && k < params.k) { // Check if column & k are valid
            let load_global_bIdx_base = (ldbn * load_global_jj_b) + (k / 8u);
            for (var offset = 0u; offset < TILE_B_HEIGHT_U32; offset = offset + 1u) {
                // Store column lx's data contiguously in the tile
                let dest_idx = lx * TILE_B_HEIGHT_U32 + offset;
                // Check if this specific read is within k bounds (might be slightly redundant with outer k check)
                // Assuming reads within the k-block are safe if k itself is < params.k
                tileB[dest_idx] = B[load_global_bIdx_base + offset];
            }
        } else { // Pad column with zeros if out of bounds
            for (var offset = 0u; offset < TILE_B_HEIGHT_U32; offset = offset + 1u) {
                 let dest_idx = lx * TILE_B_HEIGHT_U32 + offset;
                 tileB[dest_idx] = 0u;
            }
        }

        // Load B2 scale data (1 f32 per thread/column)
        let load_global_jj_b2 = workgroup_id.x * RN + lx;
        var loaded_b2_val : f32 = 0.0;
        if (load_global_jj_b2 < params.n && k < params.k) { // Check if column & k are valid
            let load_bIdx2 = (ldbs * load_global_jj_b2) + (k / BLOCK_SIZE);
            loaded_b2_val = B2[load_bIdx2];
        }
        tileB2[lx] = loaded_b2_val;

        // --- Barrier: Ensure loading is complete ---
        workgroupBarrier();

        // --- Computation Phase ---

        // Get A scale (same for all threads in workgroup for this k)
        var a_scale : f32 = 0.0;
        let aIdx2 = k / BLOCK_SIZE; // A is 1xK, so index depends only on k
        if (k < params.k) {
             // Read A scale - Assuming A2 layout matches this indexing
             // If A is truly 1xK, ldas might be irrelevant, A2 might just be k/BLOCK_SIZE
             // Sticking with original indexing for now:
             // let aIdx2_full = (ldas * 0) + (k / BLOCK_SIZE); // If ldas was row based
             a_scale = A2[aIdx2];
        }

        // Get B scale for this thread's column from shared memory
        let b_scale = tileB2[lx];
        let scale = a_scale * b_scale; // Combined scale for this block/column

        // Dequantize B for column lx using tileB
        // Read the 4 u32s for this column (lx) from the correct location
        let tile_col_base_idx = lx * TILE_B_HEIGHT_U32;
        let b_val0 = tileB[tile_col_base_idx + 0u];
        let b_val1 = tileB[tile_col_base_idx + 1u];
        let b_val2 = tileB[tile_col_base_idx + 2u];
        let b_val3 = tileB[tile_col_base_idx + 3u];

        // Calculate the 32 dequantized B values (q0..q31) using the permutation
        let q0  = f32((b_val0 >> 0u) & 0xFu) - 8.0;
        let q16 = f32((b_val0 >> 4u) & 0xFu) - 8.0;
        let q1  = f32((b_val0 >> 8u) & 0xFu) - 8.0;
        let q17 = f32((b_val0 >> 12u) & 0xFu) - 8.0;
        let q2  = f32((b_val0 >> 16u) & 0xFu) - 8.0;
        let q18 = f32((b_val0 >> 20u) & 0xFu) - 8.0;
        let q3  = f32((b_val0 >> 24u) & 0xFu) - 8.0;
        let q19 = f32((b_val0 >> 28u) & 0xFu) - 8.0;

        let q4  = f32((b_val1 >> 0u) & 0xFu) - 8.0;
        let q20 = f32((b_val1 >> 4u) & 0xFu) - 8.0;
        let q5  = f32((b_val1 >> 8u) & 0xFu) - 8.0;
        let q21 = f32((b_val1 >> 12u) & 0xFu) - 8.0;
        let q6  = f32((b_val1 >> 16u) & 0xFu) - 8.0;
        let q22 = f32((b_val1 >> 20u) & 0xFu) - 8.0;
        let q7  = f32((b_val1 >> 24u) & 0xFu) - 8.0;
        let q23 = f32((b_val1 >> 28u) & 0xFu) - 8.0;

        let q8  = f32((b_val2 >> 0u) & 0xFu) - 8.0;
        let q24 = f32((b_val2 >> 4u) & 0xFu) - 8.0;
        let q9  = f32((b_val2 >> 8u) & 0xFu) - 8.0;
        let q25 = f32((b_val2 >> 12u) & 0xFu) - 8.0;
        let q10 = f32((b_val2 >> 16u) & 0xFu) - 8.0;
        let q26 = f32((b_val2 >> 20u) & 0xFu) - 8.0;
        let q11 = f32((b_val2 >> 24u) & 0xFu) - 8.0;
        let q27 = f32((b_val2 >> 28u) & 0xFu) - 8.0;

        let q12 = f32((b_val3 >> 0u) & 0xFu) - 8.0;
        let q28 = f32((b_val3 >> 4u) & 0xFu) - 8.0;
        let q13 = f32((b_val3 >> 8u) & 0xFu) - 8.0;
        let q29 = f32((b_val3 >> 12u) & 0xFu) - 8.0;
        let q14 = f32((b_val3 >> 16u) & 0xFu) - 8.0;
        let q30 = f32((b_val3 >> 20u) & 0xFu) - 8.0;
        let q15 = f32((b_val3 >> 24u) & 0xFu) - 8.0;
        let q31 = f32((b_val3 >> 28u) & 0xFu) - 8.0;


        // Read A data directly from global memory for the current k-block
        let base_a_idx = k / 4u; // Base u32 index into A for this block
        var packed_a : u32;

        // Accumulate the 32 products for this k-block
        // Loop unrolled for clarity, ensure correct A byte matches correct q value
        // Offset 0: Bytes 0-3 -> q0, q1, q2, q3
        packed_a = select(0u, A[base_a_idx + 0u], k < params.k); // Read or pad
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q0 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q1 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q2 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q3 * scale);

        // Offset 1: Bytes 4-7 -> q4, q5, q6, q7
        packed_a = select(0u, A[base_a_idx + 1u], k + 4u < params.k);
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q4 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q5 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q6 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q7 * scale);

        // Offset 2: Bytes 8-11 -> q8, q9, q10, q11
        packed_a = select(0u, A[base_a_idx + 2u], k + 8u < params.k);
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q8 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q9 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q10 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q11 * scale);

        // Offset 3: Bytes 12-15 -> q12, q13, q14, q15
        packed_a = select(0u, A[base_a_idx + 3u], k + 12u < params.k);
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q12 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q13 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q14 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q15 * scale);

        // Offset 4: Bytes 16-19 -> q16, q17, q18, q19
        packed_a = select(0u, A[base_a_idx + 4u], k + 16u < params.k);
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q16 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q17 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q18 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q19 * scale);

        // Offset 5: Bytes 20-23 -> q20, q21, q22, q23
        packed_a = select(0u, A[base_a_idx + 5u], k + 20u < params.k);
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q20 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q21 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q22 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q23 * scale);

        // Offset 6: Bytes 24-27 -> q24, q25, q26, q27
        packed_a = select(0u, A[base_a_idx + 6u], k + 24u < params.k);
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q24 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q25 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q26 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q27 * scale);

        // Offset 7: Bytes 28-31 -> q28, q29, q30, q31
        packed_a = select(0u, A[base_a_idx + 7u], k + 28u < params.k);
        sum = sum + (i8_to_f32((packed_a >> 0u) & 0xFFu) * q28 * scale);
        sum = sum + (i8_to_f32((packed_a >> 8u) & 0xFFu) * q29 * scale);
        sum = sum + (i8_to_f32((packed_a >> 16u) & 0xFFu) * q30 * scale);
        sum = sum + (i8_to_f32((packed_a >> 24u) & 0xFFu) * q31 * scale);

        // --- Barrier: Ensure computation is complete before next load ---
        workgroupBarrier();

    } // End k loop

    // --- Final Write ---
    // Only threads within the logical bounds of C write their result.
    if (thread_in_bounds) {
        // Since M=1, the output row is 0. Index is just the column jj.
        // params.ldc is irrelevant if we only write the first row.
        let cIdx = jj;
        C[cIdx] = sum;
    }
}