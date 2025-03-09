// Define the parameter struct containing matrix dimensions and strides
struct TParams {
    k: u32,      // Number of rows in B
    n: u32,      // Number of columns in B
    sdb: u32,    // Leading dimension of B (stride between rows)
    // Other parameters could be included, but these suffice for transposition
};

// Input: Matrix B in row-major order
@group(0) @binding(0) var<storage, read> B: array<f32>;

// Output: Matrix B_transposed in row-major order
@group(0) @binding(1) var<storage, read_write> B_transposed: array<f32>;

// Uniforms containing matrix parameters
@group(0) @binding(2) var<uniform> params: TParams;

// Workgroup size constants
const TX = 16u;  // Tile width
const TY = 16u;  // Tile height

// Compute shader with a 16x16 workgroup
@compute @workgroup_size(TX, TY)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // Compute global indices for B_transposed (j, i)
    let j = workgroup_id.x * TX + local_id.x;  // Column index of B, row index of B_transposed
    let i = workgroup_id.y * TY + local_id.y;  // Row index of B, column index of B_transposed

    // Check bounds: j < N and i < K
    if (j < params.n && i < params.k) {
        // B_transposed[j][i] = B[i][j]
        // In row-major order:
        // - B[i][j] is at i * sdb + j in the input array
        // - B_transposed[j][i] is at j * K + i in the output array
        B_transposed[j * params.k + i] = B[i * params.sdb + j];
    }
}