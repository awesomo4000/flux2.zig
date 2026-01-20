//! SIMD compute kernels for flux2.zig
//!
//! Pure Zig implementations of neural network primitives:
//! - Matrix multiplication (matmul)
//! - Activation functions (GELU, SiLU, softmax)
//! - Normalization (LayerNorm, RMSNorm)
//! - Positional encoding (RoPE)
//!
//! All functions use portable SIMD via @Vector

const std = @import("std");
const math = std.math;

/// Preferred vector size for this platform
pub const VEC_SIZE = std.simd.suggestVectorLength(f32) orelse 8;
pub const Vec = @Vector(VEC_SIZE, f32);

// ─────────────────────────────────────────────────────────────────────────────
// Matrix Operations
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix multiplication: C = A @ B
/// A: [M, K], B: [K, N], C: [M, N]
/// Row-major layout assumed
pub fn matmul(
    C: []f32,
    A: []const f32,
    B: []const f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    // Tile sizes for cache efficiency
    const TILE_M = 4;
    const TILE_N = VEC_SIZE * 4;
    const TILE_K = 64;

    // Clear output
    @memset(C, 0);

    // Tiled matrix multiplication
    var i: usize = 0;
    while (i < M) : (i += TILE_M) {
        const m_end = @min(i + TILE_M, M);

        var j: usize = 0;
        while (j < N) : (j += TILE_N) {
            const n_end = @min(j + TILE_N, N);

            var k: usize = 0;
            while (k < K) : (k += TILE_K) {
                const k_end = @min(k + TILE_K, K);

                // Inner tile computation
                matmulTile(C, A, B, M, N, K, i, m_end, j, n_end, k, k_end);
            }
        }
    }
}

fn matmulTile(
    C: []f32,
    A: []const f32,
    B: []const f32,
    _: usize, // M
    N: usize,
    K: usize,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
) void {
    var ii = i_start;
    while (ii < i_end) : (ii += 1) {
        var jj = j_start;

        // SIMD inner loop
        while (jj + VEC_SIZE <= j_end) : (jj += VEC_SIZE) {
            var acc: Vec = @splat(0);

            var kk = k_start;
            while (kk < k_end) : (kk += 1) {
                const a_val: Vec = @splat(A[ii * K + kk]);
                const b_vec: Vec = B[kk * N + jj ..][0..VEC_SIZE].*;
                acc += a_val * b_vec;
            }

            // Add to existing C values
            const c_ptr: *[VEC_SIZE]f32 = @ptrCast(C[ii * N + jj ..].ptr);
            const c_vec: Vec = c_ptr.*;
            c_ptr.* = c_vec + acc;
        }

        // Scalar remainder
        while (jj < j_end) : (jj += 1) {
            var sum: f32 = 0;
            var kk = k_start;
            while (kk < k_end) : (kk += 1) {
                sum += A[ii * K + kk] * B[kk * N + jj];
            }
            C[ii * N + jj] += sum;
        }
    }
}

/// Matrix-vector multiplication: y = A @ x
/// A: [M, K], x: [K], y: [M]
pub fn matvec(y: []f32, A: []const f32, x: []const f32, M: usize, K: usize) void {
    for (0..M) |i| {
        var acc: Vec = @splat(0);
        var k: usize = 0;

        // SIMD accumulation
        while (k + VEC_SIZE <= K) : (k += VEC_SIZE) {
            const a_vec: Vec = A[i * K + k ..][0..VEC_SIZE].*;
            const x_vec: Vec = x[k..][0..VEC_SIZE].*;
            acc += a_vec * x_vec;
        }

        // Horizontal sum
        var sum = @reduce(.Add, acc);

        // Scalar remainder
        while (k < K) : (k += 1) {
            sum += A[i * K + k] * x[k];
        }

        y[i] = sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Activation Functions
// ─────────────────────────────────────────────────────────────────────────────

/// GELU activation (Gaussian Error Linear Unit)
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: []f32) void {
    const sqrt_2_pi: f32 = 0.7978845608028654; // sqrt(2/pi)
    const coeff: f32 = 0.044715;

    var i: usize = 0;
    while (i + VEC_SIZE <= x.len) : (i += VEC_SIZE) {
        const vec: Vec = x[i..][0..VEC_SIZE].*;
        const vec3 = vec * vec * vec;
        const inner = @as(Vec, @splat(sqrt_2_pi)) * (vec + @as(Vec, @splat(coeff)) * vec3);
        const tanh_val = tanhVec(inner);
        const result = @as(Vec, @splat(0.5)) * vec * (@as(Vec, @splat(1.0)) + tanh_val);
        @as(*[VEC_SIZE]f32, @ptrCast(x[i..].ptr)).* = result;
    }

    // Scalar remainder
    while (i < x.len) : (i += 1) {
        const val = x[i];
        const inner = sqrt_2_pi * (val + coeff * val * val * val);
        x[i] = 0.5 * val * (1.0 + math.tanh(inner));
    }
}

/// SiLU activation (Sigmoid Linear Unit) aka Swish
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(x: []f32) void {
    var i: usize = 0;
    while (i + VEC_SIZE <= x.len) : (i += VEC_SIZE) {
        const vec: Vec = x[i..][0..VEC_SIZE].*;
        const sig = sigmoidVec(vec);
        const result = vec * sig;
        @as(*[VEC_SIZE]f32, @ptrCast(x[i..].ptr)).* = result;
    }

    while (i < x.len) : (i += 1) {
        x[i] = x[i] * sigmoid(x[i]);
    }
}

/// Softmax along the last dimension
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
pub fn softmax(x: []f32, size: usize) void {
    const n_rows = x.len / size;

    for (0..n_rows) |row| {
        const start = row * size;
        const row_data = x[start..][0..size];

        // Find max for numerical stability
        var max_val: f32 = -math.inf(f32);
        for (row_data) |v| {
            max_val = @max(max_val, v);
        }

        // Compute exp(x - max) and sum
        var sum: f32 = 0;
        for (row_data) |*v| {
            v.* = @exp(v.* - max_val);
            sum += v.*;
        }

        // Normalize
        const inv_sum = 1.0 / sum;
        for (row_data) |*v| {
            v.* *= inv_sum;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalization
// ─────────────────────────────────────────────────────────────────────────────

/// Layer Normalization
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
pub fn layerNorm(
    out: []f32,
    x: []const f32,
    gamma: []const f32,
    beta: []const f32,
    eps: f32,
) void {
    const n = x.len;

    // Compute mean
    var sum: f32 = 0;
    for (x) |v| sum += v;
    const mean = sum / @as(f32, @floatFromInt(n));

    // Compute variance
    var var_sum: f32 = 0;
    for (x) |v| {
        const diff = v - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / @as(f32, @floatFromInt(n));
    const inv_std = 1.0 / @sqrt(variance + eps);

    // Normalize and scale
    for (out, x, gamma, beta) |*o, xi, g, b| {
        o.* = (xi - mean) * inv_std * g + b;
    }
}

/// RMS Normalization (used in LLaMA, Qwen)
/// y = x / sqrt(mean(x^2) + eps) * weight
pub fn rmsNorm(
    out: []f32,
    x: []const f32,
    weight: []const f32,
    eps: f32,
) void {
    const n = x.len;

    // Compute mean of squares
    var sum_sq: f32 = 0;
    var i: usize = 0;

    // SIMD accumulation
    var acc: Vec = @splat(0);
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const vec: Vec = x[i..][0..VEC_SIZE].*;
        acc += vec * vec;
    }
    sum_sq = @reduce(.Add, acc);

    // Scalar remainder
    while (i < n) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
    const inv_rms = 1.0 / rms;

    // Scale
    for (out, x, weight) |*o, xi, w| {
        o.* = xi * inv_rms * w;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Positional Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding (RoPE)
/// Applies rotation to query and key tensors
pub fn rope(
    q: []f32, // [seq_len, num_heads, head_dim]
    k: []f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    start_pos: usize,
    theta: f32,
) void {
    const half_dim = head_dim / 2;

    for (0..seq_len) |pos| {
        const actual_pos = start_pos + pos;

        for (0..num_heads) |h| {
            const q_offset = pos * num_heads * head_dim + h * head_dim;
            const k_offset = pos * num_heads * head_dim + h * head_dim;

            for (0..half_dim) |i| {
                const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(actual_pos)) * freq;
                const cos_val = @cos(angle);
                const sin_val = @sin(angle);

                // Rotate q
                const q0 = q[q_offset + i];
                const q1 = q[q_offset + i + half_dim];
                q[q_offset + i] = q0 * cos_val - q1 * sin_val;
                q[q_offset + i + half_dim] = q0 * sin_val + q1 * cos_val;

                // Rotate k
                const k0 = k[k_offset + i];
                const k1 = k[k_offset + i + half_dim];
                k[k_offset + i] = k0 * cos_val - k1 * sin_val;
                k[k_offset + i + half_dim] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise Operations
// ─────────────────────────────────────────────────────────────────────────────

/// Add two vectors: y = a + b
pub fn add(y: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(y.len == a.len and a.len == b.len);

    var i: usize = 0;
    while (i + VEC_SIZE <= y.len) : (i += VEC_SIZE) {
        const va: Vec = a[i..][0..VEC_SIZE].*;
        const vb: Vec = b[i..][0..VEC_SIZE].*;
        @as(*[VEC_SIZE]f32, @ptrCast(y[i..].ptr)).* = va + vb;
    }
    while (i < y.len) : (i += 1) {
        y[i] = a[i] + b[i];
    }
}

/// Multiply two vectors element-wise: y = a * b
pub fn mul(y: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(y.len == a.len and a.len == b.len);

    var i: usize = 0;
    while (i + VEC_SIZE <= y.len) : (i += VEC_SIZE) {
        const va: Vec = a[i..][0..VEC_SIZE].*;
        const vb: Vec = b[i..][0..VEC_SIZE].*;
        @as(*[VEC_SIZE]f32, @ptrCast(y[i..].ptr)).* = va * vb;
    }
    while (i < y.len) : (i += 1) {
        y[i] = a[i] * b[i];
    }
}

/// Scale vector: y = a * scalar
pub fn scale(y: []f32, a: []const f32, scalar: f32) void {
    std.debug.assert(y.len == a.len);
    const s: Vec = @splat(scalar);

    var i: usize = 0;
    while (i + VEC_SIZE <= y.len) : (i += VEC_SIZE) {
        const va: Vec = a[i..][0..VEC_SIZE].*;
        @as(*[VEC_SIZE]f32, @ptrCast(y[i..].ptr)).* = va * s;
    }
    while (i < y.len) : (i += 1) {
        y[i] = a[i] * scalar;
    }
}

/// Dot product of two vectors
pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var acc: Vec = @splat(0);
    var i: usize = 0;

    while (i + VEC_SIZE <= a.len) : (i += VEC_SIZE) {
        const va: Vec = a[i..][0..VEC_SIZE].*;
        const vb: Vec = b[i..][0..VEC_SIZE].*;
        acc += va * vb;
    }

    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn sigmoidVec(x: Vec) Vec {
    const ones: Vec = @splat(1.0);
    return ones / (ones + @exp(-x));
}

fn tanhVec(x: Vec) Vec {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    const exp2x = @exp(@as(Vec, @splat(2.0)) * x);
    const ones: Vec = @splat(1.0);
    return (exp2x - ones) / (exp2x + ones);
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear (Matrix Multiply) without bias
// ─────────────────────────────────────────────────────────────────────────────

/// Linear layer without bias: out = x @ W^T
/// x: [batch, in_features]
/// weight: [out_features, in_features]
/// out: [batch, out_features]
pub fn linearNoBias(out: []f32, x: []const f32, weight: []const f32, batch: usize, in_features: usize, out_features: usize) void {
    // out = x @ W^T is equivalent to matmul(x, W^T)
    // But W is [out, in], so we do out[b, o] = sum_i x[b, i] * W[o, i]
    for (0..batch) |b| {
        for (0..out_features) |o| {
            var sum: f32 = 0;
            for (0..in_features) |i| {
                sum += x[b * in_features + i] * weight[o * in_features + i];
            }
            out[b * out_features + o] = sum;
        }
    }
}

/// Elementwise multiply-accumulate: out += scale_vec * x (per-row scale)
pub fn addScaled(out: []f32, x: []const f32, scale_vec: []const f32, seq: usize, hidden: usize) void {
    for (0..seq) |s| {
        for (0..hidden) |h| {
            const idx = s * hidden + h;
            out[idx] += scale_vec[h] * x[idx];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolution
// ─────────────────────────────────────────────────────────────────────────────

/// 2D convolution: out = conv2d(input, weight) + bias
/// input: [batch, in_ch, H, W]
/// weight: [out_ch, in_ch, kH, kW]
/// bias: [out_ch] or null
/// out: [batch, out_ch, out_H, out_W]
pub fn conv2d(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    H: usize,
    W: usize,
    kH: usize,
    kW: usize,
    stride: usize,
    padding: usize,
) void {
    const out_H = (H + 2 * padding - kH) / stride + 1;
    const out_W = (W + 2 * padding - kW) / stride + 1;

    for (0..batch) |b| {
        for (0..out_ch) |oc| {
            const bias_val = if (bias) |bb| bb[oc] else 0.0;

            for (0..out_H) |oh| {
                for (0..out_W) |ow| {
                    var sum: f32 = bias_val;

                    for (0..in_ch) |ic| {
                        for (0..kH) |kh| {
                            for (0..kW) |kw| {
                                const ih_raw = oh * stride + kh;
                                const iw_raw = ow * stride + kw;

                                // Handle padding
                                if (ih_raw >= padding and ih_raw < H + padding and
                                    iw_raw >= padding and iw_raw < W + padding)
                                {
                                    const ih = ih_raw - padding;
                                    const iw = iw_raw - padding;

                                    const in_idx = b * in_ch * H * W + ic * H * W + ih * W + iw;
                                    const w_idx = oc * in_ch * kH * kW + ic * kH * kW + kh * kW + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }

                    const out_idx = b * out_ch * out_H * out_W + oc * out_H * out_W + oh * out_W + ow;
                    out[out_idx] = sum;
                }
            }
        }
    }
}

/// 2D convolution (im2col + matmul version for better performance)
/// For larger kernels and channel counts
pub fn conv2dFast(
    allocator: std.mem.Allocator,
    out: []f32,
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    H: usize,
    W: usize,
    kH: usize,
    kW: usize,
    stride: usize,
    padding: usize,
) !void {
    const out_H = (H + 2 * padding - kH) / stride + 1;
    const out_W = (W + 2 * padding - kW) / stride + 1;
    const col_size = in_ch * kH * kW;
    const spatial = out_H * out_W;

    // Allocate im2col buffer
    const col = try allocator.alloc(f32, col_size * spatial);
    defer allocator.free(col);

    for (0..batch) |b| {
        // im2col: transform input patch to column
        for (0..out_H) |oh| {
            for (0..out_W) |ow| {
                const col_idx = oh * out_W + ow;
                for (0..in_ch) |ic| {
                    for (0..kH) |kh| {
                        for (0..kW) |kw| {
                            const ih_raw = oh * stride + kh;
                            const iw_raw = ow * stride + kw;
                            const row = ic * kH * kW + kh * kW + kw;

                            if (ih_raw >= padding and ih_raw < H + padding and
                                iw_raw >= padding and iw_raw < W + padding)
                            {
                                const ih = ih_raw - padding;
                                const iw = iw_raw - padding;
                                const in_idx = b * in_ch * H * W + ic * H * W + ih * W + iw;
                                col[row * spatial + col_idx] = input[in_idx];
                            } else {
                                col[row * spatial + col_idx] = 0;
                            }
                        }
                    }
                }
            }
        }

        // matmul: out = weight @ col
        // weight: [out_ch, col_size], col: [col_size, spatial]
        const out_offset = b * out_ch * spatial;
        matmul(out[out_offset..][0 .. out_ch * spatial], weight, col, out_ch, spatial, col_size);

        // Add bias
        if (bias) |bb| {
            for (0..out_ch) |oc| {
                const b_val = bb[oc];
                for (0..spatial) |s| {
                    out[out_offset + oc * spatial + s] += b_val;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Group Normalization
// ─────────────────────────────────────────────────────────────────────────────

/// Group Normalization
/// input: [batch, channels, H, W]
/// weight, bias: [channels]
/// num_groups: must divide channels evenly
pub fn groupNorm(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    batch: usize,
    channels: usize,
    H: usize,
    W: usize,
    num_groups: usize,
    eps: f32,
) void {
    const channels_per_group = channels / num_groups;
    const spatial = H * W;
    const group_size = channels_per_group * spatial;

    for (0..batch) |b| {
        for (0..num_groups) |g| {
            // Compute mean and variance for this group
            var sum: f32 = 0;
            var sum_sq: f32 = 0;

            const group_start = g * channels_per_group;
            for (0..channels_per_group) |c_off| {
                const c = group_start + c_off;
                for (0..spatial) |s| {
                    const idx = b * channels * spatial + c * spatial + s;
                    const val = input[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }

            const mean = sum / @as(f32, @floatFromInt(group_size));
            const variance = sum_sq / @as(f32, @floatFromInt(group_size)) - mean * mean;
            const inv_std = 1.0 / @sqrt(variance + eps);

            // Normalize and scale
            for (0..channels_per_group) |c_off| {
                const c = group_start + c_off;
                const w = weight[c];
                const bb = bias[c];
                for (0..spatial) |s| {
                    const idx = b * channels * spatial + c * spatial + s;
                    out[idx] = (input[idx] - mean) * inv_std * w + bb;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Upsampling
// ─────────────────────────────────────────────────────────────────────────────

/// Nearest neighbor 2x upsampling
/// input: [batch, channels, H, W]
/// out: [batch, channels, H*2, W*2]
pub fn upsampleNearest2x(
    out: []f32,
    input: []const f32,
    batch: usize,
    channels: usize,
    H: usize,
    W: usize,
) void {
    const out_H = H * 2;
    const out_W = W * 2;

    for (0..batch) |b| {
        for (0..channels) |c| {
            for (0..out_H) |oh| {
                const ih = oh / 2;
                for (0..out_W) |ow| {
                    const iw = ow / 2;
                    const in_idx = b * channels * H * W + c * H * W + ih * W + iw;
                    const out_idx = b * channels * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
                    out[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "matmul basic" {
    // 2x3 @ 3x2 = 2x2
    const A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const B = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var C = [_]f32{ 0, 0, 0, 0 };

    matmul(&C, &A, &B, 2, 2, 3);

    // Expected: [[58, 64], [139, 154]]
    try std.testing.expectApproxEqAbs(@as(f32, 58), C[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64), C[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 139), C[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 154), C[3], 0.001);
}

test "gelu activation" {
    var x = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    gelu(&x);

    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[2], 0.001);
    // GELU(1) ≈ 0.841
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), x[3], 0.01);
}

test "softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    softmax(&x, 4);

    // Sum should be 1
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    // Values should be monotonically increasing
    try std.testing.expect(x[0] < x[1]);
    try std.testing.expect(x[1] < x[2]);
    try std.testing.expect(x[2] < x[3]);
}

test "rms_norm" {
    var out = [_]f32{ 0, 0, 0, 0 };
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    rmsNorm(&out, &x, &weight, 1e-6);

    // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Normalized values should sum squares to ~n
    var sum_sq: f32 = 0;
    for (out) |v| sum_sq += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), sum_sq, 0.1);
}

test "dot product" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };

    const result = dot(&a, &b);
    // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
    try std.testing.expectApproxEqAbs(@as(f32, 120.0), result, 0.001);
}
