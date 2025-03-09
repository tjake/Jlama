struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    m0: u32,
    n0: u32,
    aoffset: u32,
    boffset: u32,
    roffset: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RM = 16u;  // Rows per tile
const RN = 16u;  // Columns per tile
const BK = 16u;  // Tile size along k dimension

const RMBK = RM * BK;
const RNBK = RN * BK;

// Shared memory for tiles
var<workgroup> As: array<f32, RMBK>;
var<workgroup> Bs: array<f32, RNBK>;

@compute @workgroup_size(RN, RM)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Workgroup base indices
    let tii = workgroup_id.y * RM;
    let tjj = workgroup_id.x * RN;
    // Thread-specific indices
    let ii = tii + local_id.y;
    let jj = tjj + local_id.x;

    var sum: f32 = 0.0;

    // Tile along k dimension
    for (var kt = 0u; kt < params.k; kt += BK) {
        // Load tile of A
        if (kt + local_id.x < params.k) {
            As[local_id.y * BK + local_id.x] = A[(tii + local_id.y) * params.lda + (kt + local_id.x)];
        } else {
            As[local_id.y * BK + local_id.x] = 0.0;
        }

        // Load tile of B
        if (kt + local_id.x < params.k) {
            Bs[local_id.x * RN + local_id.y] = B[(tjj + local_id.y) * params.ldb + (kt + local_id.x)];
        } else {
            Bs[local_id.x * RN + local_id.y] = 0.0;
        }

        // Wait for all threads to load
        workgroupBarrier();

        // Compute partial sum using shared memory
        for (var kp = 0u; kp < BK; kp++) {
            if (kt + kp < params.k) {
                sum += As[local_id.y * BK + kp] * Bs[kp * RN + local_id.x];
            }
        }

        // Wait before loading next tile
        workgroupBarrier();
    }

    // Store result
    if (ii < params.m && jj < params.n) {
        C[ii * params.ldc + jj] = sum;
    }
}