struct Params {
    m: u32,        // Assumed == 1
    n: u32,
    k: u32,
    lda: u32,      // Leading dim of A (u32 elements)
    ldb: u32,
    ldc: u32,
    boffset: u32,  // Offset for B in the global memory (due to memory alignment)
    b2offset: u32, // Offset for B2 in the global memory (due to memory alignment)
};

// --- Inputs from Global Memory ---
@group(0) @binding(0) var<storage, read> A: array<u32>;         // Input A (Q8 bytes packed into u32s)
@group(0) @binding(1) var<storage, read> A2: array<f32>;      // Scales for A
@group(0) @binding(2) var<storage, read> B: array<vec4<u32>>; // Input B (Q4 packed)
@group(0) @binding(3) var<storage, read> B2: array<f32>;      // Scales for B
@group(0) @binding(4) var<storage, read_write> C: array<f32>; // Output C
@group(0) @binding(5) var<uniform> params: Params;

// --- Constants ---
const RN = 64u;
const RM = 1u;
const BLOCK_SIZE = 32u;

// --- Workgroup Shared Memory ---
const MAX_K_FOR_SHARED_MEM = 8192u * 2;
const WG_A_U32_SIZE = (MAX_K_FOR_SHARED_MEM + 3u) / 4u;
const WG_A2_SIZE = (MAX_K_FOR_SHARED_MEM + BLOCK_SIZE - 1u) / BLOCK_SIZE;

var<workgroup> arow: array<u32, WG_A_U32_SIZE>;
var<workgroup> wg_a2: array<f32, WG_A2_SIZE>;

// --- Helper Functions ---

// Dequantize u4 nibble -> f32
fn u4_to_f32(nibble_u: u32) -> f32 {
    // Center the u4 range [0, 15] around zero -> [-8.0, 7.0]
    return f32(nibble_u) - 8.0;
}

// Unpack 4xi8 from u32 -> vec4<f32> using the required signed conversion [-128, 127] -> f32
fn unpack_i8_to_f32(packed_val: u32) -> vec4<f32> {
     let v0 = i32((packed_val >> 0u) & 0xFFu);
     let v1 = i32((packed_val >> 8u) & 0xFFu);
     let v2 = i32((packed_val >> 16u) & 0xFFu);
     let v3 = i32((packed_val >> 24u) & 0xFFu);
     // Use select to handle signed conversion: if > 127, subtract 256
     return vec4<f32>(
         f32(select(v0, v0 - 256, v0 > 127)),
         f32(select(v1, v1 - 256, v1 > 127)),
         f32(select(v2, v2 - 256, v2 > 127)),
         f32(select(v3, v3 - 256, v3 > 127))
     );
}

// --- Main Compute Kernel ---
@compute @workgroup_size(RN, RM)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let lid = local_id.x;

    // --- Step 1: Load Vector A (u32) and Scales A2 (Combined Loop) ---
    let k_actual = params.k;
    let num_a_u32s_to_load = (k_actual + 3u) / 4u;
    let num_a2_floats_to_load = (k_actual + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    // Combine loading into a single loop iterating up to the max needed for wg_a_u32
    for (var i = lid; i < num_a_u32s_to_load; i = i + RN) {
        // Load wg_a_u32 element
        if (i < WG_A_U32_SIZE) {
            arow[i] = A[i];
        }

        // Conditionally load wg_a2 element if within its required range *and* bounds
        if (i < num_a2_floats_to_load && i < WG_A2_SIZE) {
             wg_a2[i] = A2[i];
        }
    }

    // --- Step 2: Synchronize Workgroup ---
    workgroupBarrier();

    // --- Step 3: Compute Output Elements C[jj] ---
    // (Computation logic remains exactly the same as the previous version)
    let jj = workgroup_id.x * RN + local_id.x;
    if (jj >= params.n) { return; }

    let ldbn = params.ldb / 32u;
    let ldbs = params.ldb / BLOCK_SIZE;
    var sum: f32 = 0.0;
    var nib: vec4<u32>;

    for (var k = 0u; k < k_actual; k = k + BLOCK_SIZE) {
        let bIdx = (ldbn * jj) + (k / 32u) + (params.boffset);
        let bIdx2 = (ldbs * jj) + (k / BLOCK_SIZE) + (params.b2offset / 4u);
        let abase = k / 4u;
        if ((abase + 7u) >= WG_A_U32_SIZE) { continue; }

        let wg_a2_idx = k / BLOCK_SIZE;
        if (wg_a2_idx >= WG_A2_SIZE) { continue; }
        let scale_a = wg_a2[wg_a2_idx];
        let scale_b = B2[bIdx2];
        let scale = scale_a * scale_b;

        nib = B[bIdx];

        // Dequantize all 32 B values into 8 vec4 variables matching A's linear order
        // (Compiler might optimize these sequential constructions)
        let b_vec0 = vec4(u4_to_f32((nib.x >> 0u)&15u), u4_to_f32((nib.x >> 8u)&15u), u4_to_f32((nib.x >> 16u)&15u), u4_to_f32((nib.x >> 24u)&15u));
        let b_vec4 = vec4(u4_to_f32((nib.x >> 4u)&15u), u4_to_f32((nib.x >> 12u)&15u), u4_to_f32((nib.x >> 20u)&15u), u4_to_f32((nib.x >> 28u)&15u));

        let b_vec1 = vec4(u4_to_f32((nib.y >> 0u)&15u), u4_to_f32((nib.y >> 8u)&15u), u4_to_f32((nib.y >> 16u)&15u), u4_to_f32((nib.y >> 24u)&15u));
        let b_vec5 = vec4(u4_to_f32((nib.y >> 4u)&15u), u4_to_f32((nib.y >> 12u)&15u), u4_to_f32((nib.y >> 20u)&15u), u4_to_f32((nib.y >> 28u)&15u));

        let b_vec2 = vec4(u4_to_f32((nib.z >> 0u)&15u), u4_to_f32((nib.z >> 8u)&15u), u4_to_f32((nib.z >> 16u)&15u), u4_to_f32((nib.z >> 24u)&15u));
        let b_vec6 = vec4(u4_to_f32((nib.z >> 4u)&15u), u4_to_f32((nib.z >> 12u)&15u), u4_to_f32((nib.z >> 20u)&15u), u4_to_f32((nib.z >> 28u)&15u));

        let b_vec3 = vec4(u4_to_f32((nib.w >> 0u)&15u), u4_to_f32((nib.w >> 8u)&15u), u4_to_f32((nib.w >> 16u)&15u), u4_to_f32((nib.w >> 24u)&15u));
        let b_vec7 = vec4(u4_to_f32((nib.w >> 4u)&15u), u4_to_f32((nib.w >> 12u)&15u), u4_to_f32((nib.w >> 20u)&15u), u4_to_f32((nib.w >> 28u)&15u));


        sum = fma(dot(unpack_i8_to_f32(arow[abase + 0u]), b_vec0), scale, sum);
        sum = fma(dot(unpack_i8_to_f32(arow[abase + 1u]), b_vec1), scale, sum);
        sum = fma(dot(unpack_i8_to_f32(arow[abase + 2u]), b_vec2), scale, sum);
        sum = fma(dot(unpack_i8_to_f32(arow[abase + 3u]), b_vec3), scale, sum);
        sum = fma(dot(unpack_i8_to_f32(arow[abase + 4u]), b_vec4), scale, sum);
        sum = fma(dot(unpack_i8_to_f32(arow[abase + 5u]), b_vec5), scale, sum);
        sum = fma(dot(unpack_i8_to_f32(arow[abase + 6u]), b_vec6), scale, sum);
        sum = fma(dot(unpack_i8_to_f32(arow[abase + 7u]), b_vec7), scale, sum);
    }

    // Store the final result in C
    let cIdx = jj;
    C[cIdx] = sum;
}