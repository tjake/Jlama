struct Params {
    m: u32,        // Number of rows in C (and A)
    n: u32,        // Number of columns in C (and rows in B)
    k: u32,        // Shared dimension (columns of A and B)
    lda: u32,      // Leading dimension of A (k for m x k)
    ldb: u32,      // Leading dimension of B (k for n x k)
    ldc: u32,      // Leading dimension of C (n for m x n)
    m0: u32,
    n0: u32,
    aoffset: u32,
    boffset: u32,
    roffset: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;  // A: m x k row-major
@group(0) @binding(1) var<storage, read> B: array<f32>;  // B: n x k row-major
@group(0) @binding(2) var<storage, read_write> C: array<f32>;  // C: m x n row-major
@group(0) @binding(3) var<uniform> params: Params;

const RM = 16u;
const RN = 16u;
const BK = 16u;

const RMBK = RM * BK;
const RNBK = RN * BK;

var<workgroup> As: array<f32, RMBK>;
var<workgroup> Bs: array<f32, RNBK>;

@compute @workgroup_size(RM, RN)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tile_ii = workgroup_id.y * RM;
    let tile_jj = workgroup_id.x * RN;
    let ii = tile_ii + local_id.y;
    let jj = tile_jj + local_id.x;

    var sum: f32 = 0.0;

    // Loop over tiles of the k-dimension
        for (var kt = 0u; kt < params.k; kt += BK) {
            // Load shared memory As: all threads participate
            if (tile_ii + local_id.y < params.m && kt + local_id.x < params.k) {
                As[local_id.y * BK + local_id.x] = A[(tile_ii + local_id.y) * params.lda + (kt + local_id.x)];
            } else {
                As[local_id.y * BK + local_id.x] = 0.0; // Out-of-bounds threads load 0
            }

            // Load shared memory Bs: all threads participate
            if (tile_jj + local_id.x < params.n && kt + local_id.y < params.k) {
                Bs[local_id.x * BK + local_id.y] = B[(tile_jj + local_id.x) * params.ldb + (kt + local_id.y)];
            } else {
                Bs[local_id.x * BK + local_id.y] = 0.0; // Out-of-bounds threads load 0
            }

            // Synchronize all threads after loading shared memory
            workgroupBarrier();

            // Perform computation only for threads within bounds
            if (ii < params.m && jj < params.n) {
                for (var k = 0u; k < BK; k++) {
                    if (kt + k < params.k) {
                        sum += As[local_id.y * BK + k] * Bs[local_id.x * BK + k];
                    }
                }
            }

            // Synchronize all threads before the next iteration
            workgroupBarrier();
        }

        // Write result only for threads within bounds
        if (ii < params.m && jj < params.n) {
            C[ii * params.ldc + jj] = sum;
        }
}