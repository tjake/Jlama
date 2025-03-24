struct Params {
    m: u32,        // Number of rows in submatrix C
    n: u32,        // Number of columns in submatrix C
    k: u32,        // Shared dimension
    lda: u32,      // Leading dimension of A
    ldb: u32,      // Leading dimension of B
    ldc: u32,      // Leading dimension of C
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Tile sizes
const RM = 8u;  // Rows per tile
const RN = 8u;  // Columns per tile

// Extracts the bf16 at logical index i and converts it to f32
fn get_bf16_as_f32(bf16_packed: ptr<storage, array<u32>, read>, i: u32) -> f32 {
    // Map logical index i to array index and half
    let array_idx = i / 2u;        // Which u32 element
    let half_idx = i % 2u;         // 0 = lower 16 bits, 1 = upper 16 bits
    let packed = (*bf16_packed)[array_idx];

    // Extract the correct 16-bit bf16 value
    var bf16_bits: u32;
    if (half_idx == 0u) {
        bf16_bits = packed & 0xFFFFu; // Lower 16 bits
    } else {
        bf16_bits = (packed >> 16u) & 0xFFFFu; // Upper 16 bits
    }

    // Convert bf16 to f32 by shifting left 16 bits and reinterpreting
    let f32_bits = bf16_bits << 16u;
    return bitcast<f32>(f32_bits);
}

// Workgroup size: RM x RN threads, each computing one element of C
@compute @workgroup_size(RM, RN)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Calculate global indices for this thread
    let ii = workgroup_id.y * RM + local_id.y;  // Row index in submatrix
    let jj = workgroup_id.x * RN + local_id.x;  // Column index in submatrix

    if (ii < params.m && jj < params.n) {
        var sum: f32 = 0.0;
        for (var k = 0u; k < params.k; k = k + 1u) {
            let aIdx = (params.lda * ii) + k;
            let bIdx = (params.ldb * jj) + k;
            sum = sum + (A[aIdx] * get_bf16_as_f32(&B, bIdx));
        }

        // Store the result in C
        let cIdx = (params.ldc * ii) + jj;
        C[cIdx] = sum;
    }
}