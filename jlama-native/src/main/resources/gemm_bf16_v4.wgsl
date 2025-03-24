struct Params {
    m: u32,        // Number of rows in C (and A)
    n: u32,        // Number of columns in C (and rows in B)
    k: u32,        // Shared dimension (columns of A and B)
    lda: u32,      // Leading dimension of A (k for m x k)
    ldb: u32,      // Leading dimension of B (k for n x k)
    ldc: u32,      // Leading dimension of C (n for m x n)
};

@group(0) @binding(0) var<storage, read> A: array<f32>;  // A: m x k row-major
@group(0) @binding(1) var<storage, read> B: array<u32>;  // B: n x k row-major
@group(0) @binding(2) var<storage, read_write> C: array<f32>;  // C: m x n row-major
@group(0) @binding(3) var<uniform> params: Params;

const RM = 8u;
const RN = 8u;
const BK = 32u;

const RMBK = RM * BK;
const RNBK = RN * BK;

var<workgroup> As: array<f32, RMBK>;
var<workgroup> Bs: array<f32, RNBK>;

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

@compute @workgroup_size(RM, RN)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tile_ii = workgroup_id.y * RM;
    let tile_jj = workgroup_id.x * RN;
    let ii = tile_ii + local_id.y;
    let jj = tile_jj + local_id.x;

    let aoffset = ii * params.lda;
    let boffset = jj * params.ldb;

    var sum: f32 = 0.0;

    for (var kt = 0u; kt < params.k; kt += BK) {
        // Load As: each thread loads multiple k-elements
        for (var s = 0u; s < BK / RM; s++) {
            let k_idx = local_id.x + s * RM;
            if (ii < params.m && kt + k_idx < params.k) {
                As[local_id.y * BK + k_idx] = A[aoffset + (kt + k_idx)];
            } else {
                As[local_id.y * BK + k_idx] = 0.0;
            }
        }

        // Load Bs: each thread loads multiple k-elements
        for (var s = 0u; s < BK / RN; s++) {
            let k_idx = local_id.y + s * RN;
            if (jj < params.n && kt + k_idx < params.k) {
                Bs[local_id.x * BK + k_idx] = get_bf16_as_f32(&B, boffset + (kt + k_idx));
            } else {
                Bs[local_id.x * BK + k_idx] = 0.0;
            }
        }

        workgroupBarrier();

        // Compute with larger BK
        if (ii < params.m && jj < params.n) {
            for (var k = 0u; k < BK; k++) {
                if (kt + k < params.k) {
                    sum += As[local_id.y * BK + k] * Bs[local_id.x * BK + k];
                }
            }
        }
    }

    // Synchronize all threads before summing
    workgroupBarrier();

    // Write result only for threads within bounds
    if (ii < params.m && jj < params.n) {
        C[ii * params.ldc + jj] = sum;
    }
}