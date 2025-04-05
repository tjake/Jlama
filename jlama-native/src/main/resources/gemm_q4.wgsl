struct Params {
    m: u32,        // Number of rows in submatrix C
    n: u32,        // Number of columns in submatrix C
    k: u32,        // Shared dimension
    lda: u32,      // Leading dimension of A
    ldb: u32,      // Leading dimension of B
    ldc: u32,      // Leading dimension of C
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> A2: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<u32>;
@group(0) @binding(3) var<storage, read> B2: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// Tile sizes
const RM = 8u;  // Rows per tile
const RN = 8u;  // Columns per tile
const BLOCK_SIZE = 32u;
const HALF_BLOCK = 16u;

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
        let ldbn = params.ldb / 8u;  // 8 values per 32 bits
        let ldbs = params.ldb / BLOCK_SIZE;

        var sum: f32 = 0.0;
        for (var k = 0u; k < params.k; k = k + BLOCK_SIZE) {

            let aIdx = (params.lda * ii) + k;
            let bIdx = (ldbn * jj) + (k / 8u);
            let bIdx2 = (ldbs * jj) + (k / BLOCK_SIZE);

            let scale = B2[bIdx2];

            // The data is permuted
            let nibble_val = B[bIdx];
            let q0  = f32((nibble_val >> 0u) & 0xFu) - 8.0;
            let q16 = f32((nibble_val >> 4u) & 0xFu) - 8.0;
            let q1  = f32((nibble_val >> 8u) & 0xFu) - 8.0;
            let q17 = f32((nibble_val >> 12u) & 0xFu) - 8.0;
            let q2  = f32((nibble_val >> 16u) & 0xFu) - 8.0;
            let q18 = f32((nibble_val >> 20u) & 0xFu) - 8.0;
            let q3  = f32((nibble_val >> 24u) & 0xFu) - 8.0;
            let q19 = f32((nibble_val >> 28u) & 0xFu) - 8.0;

            let nibble_val2 = B[bIdx + 1u];
            let q4  = f32((nibble_val2 >> 0u) & 0xFu) - 8.0;
            let q20 = f32((nibble_val2 >> 4u) & 0xFu) - 8.0;
            let q5  = f32((nibble_val2 >> 8u) & 0xFu) - 8.0;
            let q21 = f32((nibble_val2 >> 12u) & 0xFu) - 8.0;
            let q6  = f32((nibble_val2 >> 16u) & 0xFu) - 8.0;
            let q22 = f32((nibble_val2 >> 20u) & 0xFu) - 8.0;
            let q7  = f32((nibble_val2 >> 24u) & 0xFu) - 8.0;
            let q23 = f32((nibble_val2 >> 28u) & 0xFu) - 8.0;

            let nibble_val3 = B[bIdx + 2u];
            let q8  = f32((nibble_val3 >> 0u) & 0xFu) - 8.0;
            let q24 = f32((nibble_val3 >> 4u) & 0xFu) - 8.0;
            let q9  = f32((nibble_val3 >> 8u) & 0xFu) - 8.0;
            let q25 = f32((nibble_val3 >> 12u) & 0xFu) - 8.0;
            let q10 = f32((nibble_val3 >> 16u) & 0xFu) - 8.0;
            let q26 = f32((nibble_val3 >> 20u) & 0xFu) - 8.0;
            let q11 = f32((nibble_val3 >> 24u) & 0xFu) - 8.0;
            let q27 = f32((nibble_val3 >> 28u) & 0xFu) - 8.0;

            let nibble_val4 = B[bIdx + 3u];
            let q12 = f32((nibble_val4 >> 0u) & 0xFu) - 8.0;
            let q28 = f32((nibble_val4 >> 4u) & 0xFu) - 8.0;
            let q13 = f32((nibble_val4 >> 8u) & 0xFu) - 8.0;
            let q29 = f32((nibble_val4 >> 12u) & 0xFu) - 8.0;
            let q14 = f32((nibble_val4 >> 16u) & 0xFu) - 8.0;
            let q30 = f32((nibble_val4 >> 20u) & 0xFu) - 8.0;
            let q15 = f32((nibble_val4 >> 24u) & 0xFu) - 8.0;
            let q31 = f32((nibble_val4 >> 28u) & 0xFu) - 8.0;

            sum = sum + (A[aIdx + 0u] * (scale * q0));
            sum = sum + (A[aIdx + 1u] * (scale * q1));
            sum = sum + (A[aIdx + 2u] * (scale * q2));
            sum = sum + (A[aIdx + 3u] * (scale * q3));
            sum = sum + (A[aIdx + 4u] * (scale * q4));
            sum = sum + (A[aIdx + 5u] * (scale * q5));
            sum = sum + (A[aIdx + 6u] * (scale * q6));
            sum = sum + (A[aIdx + 7u] * (scale * q7));
            sum = sum + (A[aIdx + 8u] * (scale * q8));
            sum = sum + (A[aIdx + 9u] * (scale * q9));
            sum = sum + (A[aIdx + 10u] * (scale * q10));
            sum = sum + (A[aIdx + 11u] * (scale * q11));
            sum = sum + (A[aIdx + 12u] * (scale * q12));
            sum = sum + (A[aIdx + 13u] * (scale * q13));
            sum = sum + (A[aIdx + 14u] * (scale * q14));
            sum = sum + (A[aIdx + 15u] * (scale * q15));
            sum = sum + (A[aIdx + 16u] * (scale * q16));
            sum = sum + (A[aIdx + 17u] * (scale * q17));
            sum = sum + (A[aIdx + 18u] * (scale * q18));
            sum = sum + (A[aIdx + 19u] * (scale * q19));
            sum = sum + (A[aIdx + 20u] * (scale * q20));
            sum = sum + (A[aIdx + 21u] * (scale * q21));
            sum = sum + (A[aIdx + 22u] * (scale * q22));
            sum = sum + (A[aIdx + 23u] * (scale * q23));
            sum = sum + (A[aIdx + 24u] * (scale * q24));
            sum = sum + (A[aIdx + 25u] * (scale * q25));
            sum = sum + (A[aIdx + 26u] * (scale * q26));
            sum = sum + (A[aIdx + 27u] * (scale * q27));
            sum = sum + (A[aIdx + 28u] * (scale * q28));
            sum = sum + (A[aIdx + 29u] * (scale * q29));
            sum = sum + (A[aIdx + 30u] * (scale * q30));
            sum = sum + (A[aIdx + 31u] * (scale * q31));
        }

        // Store the result in C
        let cIdx = (params.ldc * ii) + jj;
        C[cIdx] = sum;
    }
}