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

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> A2: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> B2: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// Tile sizes
const RM = 8u;  // Rows per tile
const RN = 8u;  // Columns per tile

// Workgroup size: RM x RN threads, each computing one element of C
@compute @workgroup_size(RM, RN)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let boffset = params.boffset/4u; // Convert byte offset to u32 index
    // Calculate global indices for this thread
    let ii = workgroup_id.y * RM + local_id.y;  // Row index in submatrix
    let jj = workgroup_id.x * RN + local_id.x;  // Column index in submatrix

    if (ii < params.m && jj < params.n) {
        var sum: f32 = 0.0;
        for (var k = 0u; k < params.k; k = k + 1u) {
            let aIdx = (params.lda * ii) + k;
            let bIdx = (params.ldb * jj) + k;
            sum = sum + (A[aIdx] * B[boffset + bIdx]);
        }
        // Store the result in C
        let cIdx = (params.ldc * ii) + jj;
        C[cIdx] = sum;
    }
}