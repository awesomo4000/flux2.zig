//! Tensor abstraction for flux2.zig
//!
//! Provides a multi-dimensional array type with:
//! - Arbitrary rank (compile-time or runtime shapes)
//! - Strided views without copying
//! - SIMD-friendly memory layout

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum tensor rank (dimensions)
pub const MAX_RANK: usize = 8;

/// Supported data types
pub const DType = enum {
    f32,
    f16,
    bf16,
    i32,
    u8,

    pub fn size(self: DType) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16 => 2,
            .u8 => 1,
        };
    }
};

/// Multi-dimensional tensor with strided storage
pub const Tensor = struct {
    /// Raw data storage (always f32 for computation)
    data: []f32,
    /// Shape of each dimension
    shape: [MAX_RANK]usize,
    /// Strides for each dimension (in elements, not bytes)
    strides: [MAX_RANK]usize,
    /// Number of dimensions
    ndim: usize,
    /// Allocator used (null for views)
    allocator: ?Allocator,
    /// Whether this tensor owns its data
    owns_data: bool,

    const Self = @This();

    /// Create a new tensor with given shape
    pub fn init(allocator: Allocator, shape: []const usize) !Self {
        if (shape.len > MAX_RANK) return error.TooManyDimensions;
        if (shape.len == 0) return error.EmptyShape;

        const total_elements = blk: {
            var n: usize = 1;
            for (shape) |s| {
                if (s == 0) return error.ZeroDimension;
                n = std.math.mul(usize, n, s) catch return error.TensorTooLarge;
            }
            break :blk n;
        };

        const data = try allocator.alloc(f32, total_elements);
        errdefer allocator.free(data);

        var result = Self{
            .data = data,
            .shape = [_]usize{0} ** MAX_RANK,
            .strides = [_]usize{0} ** MAX_RANK,
            .ndim = shape.len,
            .allocator = allocator,
            .owns_data = true,
        };

        // Copy shape
        @memcpy(result.shape[0..shape.len], shape);

        // Compute row-major strides
        result.computeStrides();

        return result;
    }

    /// Create a tensor initialized to zeros
    pub fn zeros(allocator: Allocator, shape: []const usize) !Self {
        const t = try init(allocator, shape);
        @memset(t.data, 0.0);
        return t;
    }

    /// Create a tensor initialized to ones
    pub fn ones(allocator: Allocator, shape: []const usize) !Self {
        const t = try init(allocator, shape);
        @memset(t.data, 1.0);
        return t;
    }

    /// Create a tensor filled with random values in [0, 1)
    pub fn rand(allocator: Allocator, shape: []const usize, rng: std.Random) !Self {
        const t = try init(allocator, shape);
        for (t.data) |*x| {
            x.* = rng.float(f32);
        }
        return t;
    }

    /// Create a tensor filled with random normal values
    pub fn randn(allocator: Allocator, shape: []const usize, rng: std.Random) !Self {
        const t = try init(allocator, shape);
        var i: usize = 0;
        while (i < t.data.len) : (i += 2) {
            // Box-Muller transform
            const rand_u1 = rng.float(f32);
            const rand_u2 = rng.float(f32);
            const r = @sqrt(-2.0 * @log(rand_u1 + 1e-10));
            const theta = 2.0 * std.math.pi * rand_u2;
            t.data[i] = r * @cos(theta);
            if (i + 1 < t.data.len) {
                t.data[i + 1] = r * @sin(theta);
            }
        }
        return t;
    }

    /// Release tensor memory
    pub fn deinit(self: *Self) void {
        if (self.owns_data) {
            if (self.allocator) |alloc| {
                alloc.free(self.data);
            }
        }
        self.* = undefined;
    }

    /// Total number of elements
    pub fn numel(self: *const Self) usize {
        var n: usize = 1;
        for (self.shape[0..self.ndim]) |s| n *= s;
        return n;
    }

    /// Get shape as a slice
    pub fn getShape(self: *const Self) []const usize {
        return self.shape[0..self.ndim];
    }

    /// Get strides as a slice
    pub fn getStrides(self: *const Self) []const usize {
        return self.strides[0..self.ndim];
    }

    /// Check if tensor is contiguous in memory
    pub fn isContiguous(self: *const Self) bool {
        var expected_stride: usize = 1;
        var i = self.ndim;
        while (i > 0) {
            i -= 1;
            if (self.strides[i] != expected_stride) return false;
            expected_stride *= self.shape[i];
        }
        return true;
    }

    /// Compute linear index from multi-dimensional indices
    pub fn linearIndex(self: *const Self, indices: []const usize) usize {
        std.debug.assert(indices.len == self.ndim);
        var idx: usize = 0;
        for (indices, 0..) |i, d| {
            std.debug.assert(i < self.shape[d]);
            idx += i * self.strides[d];
        }
        return idx;
    }

    /// Get element at indices
    pub fn at(self: *const Self, indices: []const usize) f32 {
        return self.data[self.linearIndex(indices)];
    }

    /// Set element at indices
    pub fn set(self: *Self, indices: []const usize, value: f32) void {
        self.data[self.linearIndex(indices)] = value;
    }

    /// Get element at 1D index (for rank-1 tensors or flat access)
    pub fn at1(self: *const Self, i: usize) f32 {
        return self.data[i * self.strides[0]];
    }

    /// Get element at 2D indices (for matrices)
    pub fn at2(self: *const Self, i: usize, j: usize) f32 {
        return self.data[i * self.strides[0] + j * self.strides[1]];
    }

    /// Get element at 3D indices
    pub fn at3(self: *const Self, i: usize, j: usize, k: usize) f32 {
        return self.data[i * self.strides[0] + j * self.strides[1] + k * self.strides[2]];
    }

    /// Create a view into a slice of the tensor (no copy)
    pub fn slice(self: *const Self, dim: usize, start: usize, end: usize) Self {
        std.debug.assert(dim < self.ndim);
        std.debug.assert(start < end and end <= self.shape[dim]);

        var result = self.*;
        result.shape[dim] = end - start;
        result.data = self.data[start * self.strides[dim] ..];
        result.owns_data = false;
        result.allocator = null;
        return result;
    }

    /// Reshape tensor (must have same total elements)
    pub fn reshape(self: *const Self, new_shape: []const usize) !Self {
        if (!self.isContiguous()) return error.NonContiguous;

        const new_numel = blk: {
            var n: usize = 1;
            for (new_shape) |s| n *= s;
            break :blk n;
        };
        if (new_numel != self.numel()) return error.ShapeMismatch;

        var result = self.*;
        result.ndim = new_shape.len;
        @memcpy(result.shape[0..new_shape.len], new_shape);
        result.computeStrides();
        result.owns_data = false;
        result.allocator = null;
        return result;
    }

    /// Transpose dimensions (returns a view)
    pub fn transpose(self: *const Self, dim0: usize, dim1: usize) Self {
        std.debug.assert(dim0 < self.ndim and dim1 < self.ndim);

        var result = self.*;
        std.mem.swap(usize, &result.shape[dim0], &result.shape[dim1]);
        std.mem.swap(usize, &result.strides[dim0], &result.strides[dim1]);
        result.owns_data = false;
        result.allocator = null;
        return result;
    }

    /// Create a contiguous copy
    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var result = try init(allocator, self.getShape());
        errdefer result.deinit();

        if (self.isContiguous()) {
            @memcpy(result.data, self.data[0..self.numel()]);
        } else {
            // Slow path: copy element by element
            var indices = [_]usize{0} ** MAX_RANK;
            for (0..self.numel()) |flat_idx| {
                result.data[flat_idx] = self.data[self.linearIndex(indices[0..self.ndim])];
                // Increment indices
                var d = self.ndim;
                while (d > 0) {
                    d -= 1;
                    indices[d] += 1;
                    if (indices[d] < self.shape[d]) break;
                    indices[d] = 0;
                }
            }
        }
        return result;
    }

    /// Fill tensor with value
    pub fn fill(self: *Self, value: f32) void {
        if (self.isContiguous()) {
            @memset(self.data[0..self.numel()], value);
        } else {
            var indices = [_]usize{0} ** MAX_RANK;
            for (0..self.numel()) |_| {
                self.data[self.linearIndex(indices[0..self.ndim])] = value;
                var d = self.ndim;
                while (d > 0) {
                    d -= 1;
                    indices[d] += 1;
                    if (indices[d] < self.shape[d]) break;
                    indices[d] = 0;
                }
            }
        }
    }

    /// Compute row-major strides from shape
    fn computeStrides(self: *Self) void {
        if (self.ndim == 0) return;
        var stride: usize = 1;
        var i = self.ndim;
        while (i > 0) {
            i -= 1;
            self.strides[i] = stride;
            stride *= self.shape[i];
        }
    }

    /// Get raw pointer for SIMD operations
    pub fn ptr(self: *Self) [*]f32 {
        return self.data.ptr;
    }

    /// Get const raw pointer
    pub fn constPtr(self: *const Self) [*]const f32 {
        return self.data.ptr;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "tensor creation and basic ops" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &.{ 2, 3, 4 });
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 3), t.ndim);
    try std.testing.expectEqual(@as(usize, 24), t.numel());
    try std.testing.expect(t.isContiguous());

    // Strides should be [12, 4, 1] for shape [2, 3, 4]
    try std.testing.expectEqual(@as(usize, 12), t.strides[0]);
    try std.testing.expectEqual(@as(usize, 4), t.strides[1]);
    try std.testing.expectEqual(@as(usize, 1), t.strides[2]);
}

test "tensor indexing" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, &.{ 2, 3 });
    defer t.deinit();

    t.set(&.{ 0, 1 }, 3.14);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), t.at(&.{ 0, 1 }), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), t.at2(0, 1), 0.001);
}

test "tensor reshape" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &.{ 2, 6 });
    defer t.deinit();

    // Fill with sequential values
    for (t.data, 0..) |*x, i| x.* = @floatFromInt(i);

    const reshaped = try t.reshape(&.{ 3, 4 });

    try std.testing.expectEqual(@as(usize, 2), reshaped.ndim);
    try std.testing.expectEqual(@as(usize, 3), reshaped.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), reshaped.shape[1]);
    try std.testing.expectEqual(@as(usize, 12), reshaped.numel());
}

test "tensor transpose" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &.{ 2, 3 });
    defer t.deinit();

    for (t.data, 0..) |*x, i| x.* = @floatFromInt(i);

    const transposed = t.transpose(0, 1);

    try std.testing.expectEqual(@as(usize, 3), transposed.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), transposed.shape[1]);
    try std.testing.expect(!transposed.isContiguous());

    // Check values: t[i,j] == transposed[j,i]
    try std.testing.expectEqual(t.at2(0, 2), transposed.at2(2, 0));
    try std.testing.expectEqual(t.at2(1, 1), transposed.at2(1, 1));
}

test "tensor clone" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &.{ 2, 3 });
    defer t.deinit();

    for (t.data, 0..) |*x, i| x.* = @floatFromInt(i);

    var cloned = try t.clone(allocator);
    defer cloned.deinit();

    // Modify original
    t.data[0] = 999.0;

    // Clone should be independent
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cloned.data[0], 0.001);
}
