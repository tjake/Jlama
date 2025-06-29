struct Params {
    m: u32,        // Number of rows in submatrix C
    n: u32,        // Number of columns in submatrix C
    k: u32,        // Shared dimension
    lda: u32,      // Leading dimension of A
    ldb: u32,      // Leading dimension of B
    ldc: u32,      // Leading dimension of C
    boffset: u32,  // Offset for B in the global memory (due to memory alignment)
    b2offset: u32, // Offset for B2 in the global memory (due to memory alignment)
};

@group(0) @binding(0) var<storage, read> A: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> A2: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> B2: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// Tile sizes
const RM = 8u;  // Rows per tile
const RN = 8u;  // Columns per tile
const BLOCK_SIZE = 32u;
const HALF_BLOCK = 16u;

// Helper function to extract an 8-bit value from a u32,
// interpret it as a signed byte (-128 to 127), and return it as f32.
fn i8_to_f32(byte_u: u32) -> f32 {
    //  Convert the u32 (0-255) representation to the correct signed f32 value
    //    If the most significant bit (bit 7) is set (value > 127), it's negative.
    //    We use select for potentially branchless execution.
    let byte_i = i32(byte_u); // Cast to i32 (value remains 0-255)
    let signed_val_i32 = select(
        byte_i,          // Value if false (byte_u <= 127u -> positive/zero)
        byte_i - 256,    // Value if true (byte_u > 127u -> negative)
        byte_u > 127u    // Condition: Is the sign bit set?
    );

    // 3. Cast the final signed i32 to f32
    return f32(signed_val_i32);
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
        let ldan = params.lda / 16u;  // 4 values per 32 bits, each in vec4
        let ldas = params.lda / BLOCK_SIZE;

        let ldbn = params.ldb / 32u;  // 8 values per 32 bits, each in vec4
        let ldbs = params.ldb / BLOCK_SIZE;

        var sum: f32 = 0.0;
        var nibble_val: vec4<u32>;
        for (var k = 0u; k < params.k; k = k + BLOCK_SIZE) {

            let aIdx = (ldan * ii) + (k / 16u);
            let aIdx2 = (ldas * ii) + (k / BLOCK_SIZE);
            let bIdx = (ldbn * jj) + (k / 32u) + (params.boffset);
            let bIdx2 = (ldbs * jj) + (k / BLOCK_SIZE) + (params.b2offset / 4u);

            let scale = B2[bIdx2] * A2[aIdx2];

            // The data is permuted
            nibble_val = B[bIdx];
            let q0  = f32((nibble_val.x >> 0u) & 0xFu) - 8.0;
            let q16 = f32((nibble_val.x >> 4u) & 0xFu) - 8.0;
            let q1  = f32((nibble_val.x >> 8u) & 0xFu) - 8.0;
            let q17 = f32((nibble_val.x >> 12u) & 0xFu) - 8.0;
            let q2  = f32((nibble_val.x >> 16u) & 0xFu) - 8.0;
            let q18 = f32((nibble_val.x >> 20u) & 0xFu) - 8.0;
            let q3  = f32((nibble_val.x >> 24u) & 0xFu) - 8.0;
            let q19 = f32((nibble_val.x >> 28u) & 0xFu) - 8.0;

            let q4  = f32((nibble_val.y >> 0u) & 0xFu) - 8.0;
            let q20 = f32((nibble_val.y >> 4u) & 0xFu) - 8.0;
            let q5  = f32((nibble_val.y >> 8u) & 0xFu) - 8.0;
            let q21 = f32((nibble_val.y >> 12u) & 0xFu) - 8.0;
            let q6  = f32((nibble_val.y >> 16u) & 0xFu) - 8.0;
            let q22 = f32((nibble_val.y >> 20u) & 0xFu) - 8.0;
            let q7  = f32((nibble_val.y >> 24u) & 0xFu) - 8.0;
            let q23 = f32((nibble_val.y >> 28u) & 0xFu) - 8.0;

            let q8  = f32((nibble_val.z >> 0u) & 0xFu) - 8.0;
            let q24 = f32((nibble_val.z >> 4u) & 0xFu) - 8.0;
            let q9  = f32((nibble_val.z >> 8u) & 0xFu) - 8.0;
            let q25 = f32((nibble_val.z >> 12u) & 0xFu) - 8.0;
            let q10 = f32((nibble_val.z >> 16u) & 0xFu) - 8.0;
            let q26 = f32((nibble_val.z >> 20u) & 0xFu) - 8.0;
            let q11 = f32((nibble_val.z >> 24u) & 0xFu) - 8.0;
            let q27 = f32((nibble_val.z >> 28u) & 0xFu) - 8.0;

            let q12 = f32((nibble_val.w >> 0u) & 0xFu) - 8.0;
            let q28 = f32((nibble_val.w >> 4u) & 0xFu) - 8.0;
            let q13 = f32((nibble_val.w >> 8u) & 0xFu) - 8.0;
            let q29 = f32((nibble_val.w >> 12u) & 0xFu) - 8.0;
            let q14 = f32((nibble_val.w >> 16u) & 0xFu) - 8.0;
            let q30 = f32((nibble_val.w >> 20u) & 0xFu) - 8.0;
            let q15 = f32((nibble_val.w >> 24u) & 0xFu) - 8.0;
            let q31 = f32((nibble_val.w >> 28u) & 0xFu) - 8.0;

            nibble_val = A[aIdx];
            // Get a single byte
            sum = sum + (i8_to_f32((nibble_val.x >> 0u) & 0xFFu) * q0 * scale);
            sum = sum + (i8_to_f32((nibble_val.x >> 8u) & 0xFFu) * q1 * scale);
            sum = sum + (i8_to_f32((nibble_val.x >> 16u) & 0xFFu) * q2 * scale);
            sum = sum + (i8_to_f32((nibble_val.x >> 24u) & 0xFFu) * q3 * scale);

            sum = sum + (i8_to_f32((nibble_val.y >> 0u) & 0xFFu) * q4 * scale);
            sum = sum + (i8_to_f32((nibble_val.y >> 8u) & 0xFFu) * q5 * scale);
            sum = sum + (i8_to_f32((nibble_val.y >> 16u) & 0xFFu) * q6 * scale);
            sum = sum + (i8_to_f32((nibble_val.y >> 24u) & 0xFFu) * q7 * scale);

            sum = sum + (i8_to_f32((nibble_val.z >> 0u) & 0xFFu) * q8 * scale);
            sum = sum + (i8_to_f32((nibble_val.z >> 8u) & 0xFFu) * q9 * scale);
            sum = sum + (i8_to_f32((nibble_val.z >> 16u) & 0xFFu) * q10 * scale);
            sum = sum + (i8_to_f32((nibble_val.z >> 24u) & 0xFFu) * q11 * scale);

            sum = sum + (i8_to_f32((nibble_val.w >> 0u) & 0xFFu) * q12 * scale);
            sum = sum + (i8_to_f32((nibble_val.w >> 8u) & 0xFFu) * q13 * scale);
            sum = sum + (i8_to_f32((nibble_val.w >> 16u) & 0xFFu) * q14 * scale);
            sum = sum + (i8_to_f32((nibble_val.w >> 24u) & 0xFFu) * q15 * scale);

            nibble_val = A[aIdx + 1u];
            sum = sum + (i8_to_f32((nibble_val.x >> 0u) & 0xFFu) * q16 * scale);
            sum = sum + (i8_to_f32((nibble_val.x >> 8u) & 0xFFu) * q17 * scale);
            sum = sum + (i8_to_f32((nibble_val.x >> 16u) & 0xFFu) * q18 * scale);
            sum = sum + (i8_to_f32((nibble_val.x >> 24u) & 0xFFu) * q19 * scale);

            sum = sum + (i8_to_f32((nibble_val.y >> 0u) & 0xFFu) * q20 * scale);
            sum = sum + (i8_to_f32((nibble_val.y >> 8u) & 0xFFu) * q21 * scale);
            sum = sum + (i8_to_f32((nibble_val.y >> 16u) & 0xFFu) * q22 * scale);
            sum = sum + (i8_to_f32((nibble_val.y >> 24u) & 0xFFu) * q23 * scale);

            sum = sum + (i8_to_f32((nibble_val.z >> 0u) & 0xFFu) * q24 * scale);
            sum = sum + (i8_to_f32((nibble_val.z >> 8u) & 0xFFu) * q25 * scale);
            sum = sum + (i8_to_f32((nibble_val.z >> 16u) & 0xFFu) * q26 * scale);
            sum = sum + (i8_to_f32((nibble_val.z >> 24u) & 0xFFu) * q27 * scale);

            sum = sum + (i8_to_f32((nibble_val.w >> 0u) & 0xFFu) * q28 * scale);
            sum = sum + (i8_to_f32((nibble_val.w >> 8u) & 0xFFu) * q29 * scale);
            sum = sum + (i8_to_f32((nibble_val.w >> 16u) & 0xFFu) * q30 * scale);
            sum = sum + (i8_to_f32((nibble_val.w >> 24u) & 0xFFu) * q31 * scale);
        }

        // Store the result in C
        let cIdx = (params.ldc * ii) + jj;
        C[cIdx] = sum;
    }
}