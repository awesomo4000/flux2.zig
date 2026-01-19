//! Safetensors file parser for flux2.zig
//!
//! Loads model weights from HuggingFace safetensors format.
//! Supports memory-mapped I/O for efficient large model loading.
//!
//! Format specification:
//! - 8 bytes: header size (little-endian u64)
//! - N bytes: JSON header
//! - remaining: raw tensor data

const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const DType = @import("tensor.zig").DType;

/// Supported tensor data types in safetensors
pub const SafeTensorsDType = enum {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    BOOL,

    pub fn fromString(s: []const u8) ?SafeTensorsDType {
        const map = std.StaticStringMap(SafeTensorsDType).initComptime(.{
            .{ "F32", .F32 },
            .{ "F16", .F16 },
            .{ "BF16", .BF16 },
            .{ "I32", .I32 },
            .{ "I64", .I64 },
            .{ "U8", .U8 },
            .{ "BOOL", .BOOL },
        });
        return map.get(s);
    }

    pub fn size(self: SafeTensorsDType) usize {
        return switch (self) {
            .F32, .I32 => 4,
            .F16, .BF16 => 2,
            .I64 => 8,
            .U8, .BOOL => 1,
        };
    }
};

/// Information about a tensor in the file
pub const TensorInfo = struct {
    dtype: SafeTensorsDType,
    shape: []const usize,
    data_offsets: [2]usize, // [start, end]
};

/// Safetensors file handle
pub const SafeTensors = struct {
    allocator: Allocator,
    file: std.fs.File,
    mmap: ?[]align(std.heap.page_size_min) const u8,
    file_data: ?[]const u8, // Fallback if mmap fails
    header_size: usize,
    tensors: std.StringHashMap(TensorInfo),
    shapes_arena: std.heap.ArenaAllocator,

    const Self = @This();

    /// Open a safetensors file
    pub fn open(allocator: Allocator, path: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .file = undefined,
            .mmap = null,
            .file_data = null,
            .header_size = 0,
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .shapes_arena = std.heap.ArenaAllocator.init(allocator),
        };
        errdefer self.close();

        // Open file
        self.file = try std.fs.cwd().openFile(path, .{});
        errdefer self.file.close();

        // Read header size (8 bytes, little-endian)
        var header_size_buf: [8]u8 = undefined;
        const bytes_read = try self.file.readAll(&header_size_buf);
        if (bytes_read != 8) return error.InvalidFile;

        self.header_size = std.mem.readInt(u64, &header_size_buf, .little);
        if (self.header_size > 100 * 1024 * 1024) return error.HeaderTooLarge; // 100MB sanity check

        // Memory map the file
        const file_size = try self.file.getEndPos();
        self.mmap = std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            self.file.handle,
            0,
        ) catch null;

        // Fallback to reading into memory if mmap fails
        if (self.mmap == null) {
            const data = try allocator.alloc(u8, file_size);
            try self.file.seekTo(0);
            const read = try self.file.readAll(data);
            if (read != file_size) {
                allocator.free(data);
                return error.ReadFailed;
            }
            self.file_data = data;
        }

        // Parse header
        try self.parseHeader();

        return self;
    }

    /// Close the file and free resources
    pub fn close(self: *Self) void {
        if (self.mmap) |m| {
            std.posix.munmap(m);
        }
        if (self.file_data) |d| {
            self.allocator.free(d);
        }
        self.tensors.deinit();
        self.shapes_arena.deinit();
        self.file.close();
    }

    /// Get raw bytes for the file
    fn getFileBytes(self: *const Self) []const u8 {
        if (self.mmap) |m| return m;
        if (self.file_data) |d| return d;
        unreachable;
    }

    /// Parse the JSON header
    fn parseHeader(self: *Self) !void {
        const file_bytes = self.getFileBytes();
        const header_start = 8;
        const header_end = header_start + self.header_size;

        if (header_end > file_bytes.len) return error.InvalidFile;

        const header_json = file_bytes[header_start..header_end];

        // Parse JSON
        var parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            header_json,
            .{},
        );
        defer parsed.deinit();

        const root = parsed.value.object;

        // Extract tensor info
        var it = root.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;

            // Skip metadata
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            const tensor_obj = entry.value_ptr.object;

            // Parse dtype
            const dtype_str = tensor_obj.get("dtype").?.string;
            const dtype = SafeTensorsDType.fromString(dtype_str) orelse return error.UnsupportedDtype;

            // Parse shape
            const shape_arr = tensor_obj.get("shape").?.array;
            const shape = try self.shapes_arena.allocator().alloc(usize, shape_arr.items.len);
            for (shape_arr.items, 0..) |dim, i| {
                shape[i] = @intCast(dim.integer);
            }

            // Parse data offsets
            const offsets_arr = tensor_obj.get("data_offsets").?.array;
            const start: usize = @intCast(offsets_arr.items[0].integer);
            const end: usize = @intCast(offsets_arr.items[1].integer);

            try self.tensors.put(
                try self.shapes_arena.allocator().dupe(u8, name),
                TensorInfo{
                    .dtype = dtype,
                    .shape = shape,
                    .data_offsets = .{ start, end },
                },
            );
        }
    }

    /// Get information about a tensor
    pub fn getTensorInfo(self: *const Self, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    /// Load a tensor by name, converting to f32
    pub fn getTensor(self: *const Self, allocator: Allocator, name: []const u8) !Tensor {
        const info = self.tensors.get(name) orelse return error.TensorNotFound;
        return self.loadTensor(allocator, info);
    }

    /// Load tensor data, converting to f32 if necessary
    fn loadTensor(self: *const Self, allocator: Allocator, info: TensorInfo) !Tensor {
        const file_bytes = self.getFileBytes();
        const data_start = 8 + self.header_size + info.data_offsets[0];
        const data_end = 8 + self.header_size + info.data_offsets[1];

        if (data_end > file_bytes.len) return error.InvalidDataOffset;

        const raw_data = file_bytes[data_start..data_end];

        // Calculate number of elements
        var numel: usize = 1;
        for (info.shape) |s| numel *= s;

        // Create output tensor
        var tensor = try Tensor.init(allocator, info.shape);
        errdefer tensor.deinit();

        // Convert data to f32
        switch (info.dtype) {
            .F32 => {
                const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw_data));
                @memcpy(tensor.data, src);
            },
            .F16 => {
                const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, raw_data));
                for (tensor.data, src) |*dst, s| {
                    dst.* = @floatCast(s);
                }
            },
            .BF16 => {
                // BF16 is stored as u16, needs manual conversion
                const src = std.mem.bytesAsSlice(u16, raw_data);
                for (tensor.data, src) |*dst, s| {
                    dst.* = bf16ToF32(s);
                }
            },
            .I32 => {
                const src: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, raw_data));
                for (tensor.data, src) |*dst, s| {
                    dst.* = @floatFromInt(s);
                }
            },
            else => return error.UnsupportedDtype,
        }

        return tensor;
    }

    /// Get raw bytes for a tensor (no conversion)
    pub fn getTensorBytes(self: *const Self, name: []const u8) ![]const u8 {
        const info = self.tensors.get(name) orelse return error.TensorNotFound;
        const file_bytes = self.getFileBytes();
        const data_start = 8 + self.header_size + info.data_offsets[0];
        const data_end = 8 + self.header_size + info.data_offsets[1];

        if (data_end > file_bytes.len) return error.InvalidDataOffset;
        return file_bytes[data_start..data_end];
    }

    /// List all tensor names
    pub fn listTensors(self: *const Self, allocator: Allocator) ![][]const u8 {
        var names = std.ArrayList([]const u8).init(allocator);
        errdefer names.deinit();

        var it = self.tensors.keyIterator();
        while (it.next()) |key| {
            try names.append(key.*);
        }

        return names.toOwnedSlice();
    }

    /// Check if a tensor exists
    pub fn hasTensor(self: *const Self, name: []const u8) bool {
        return self.tensors.contains(name);
    }
};

/// Convert BF16 (stored as u16) to f32
fn bf16ToF32(bf16_bits: u16) f32 {
    // BF16 is just the top 16 bits of f32
    const f32_bits: u32 = @as(u32, bf16_bits) << 16;
    return @bitCast(f32_bits);
}

/// Convert f32 to BF16 (stored as u16)
fn f32ToBf16(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    // Simple truncation (can add rounding for better accuracy)
    return @truncate(bits >> 16);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "bf16 conversion" {
    // Test some known values
    const test_vals = [_]f32{ 0.0, 1.0, -1.0, 3.14159, 1000.0, 0.001 };

    for (test_vals) |f| {
        const bf16 = f32ToBf16(f);
        const back = bf16ToF32(bf16);
        // BF16 has ~3 significant digits
        const rel_err = if (f != 0) @abs((back - f) / f) else @abs(back);
        try std.testing.expect(rel_err < 0.01);
    }
}

test "safetensors dtype parsing" {
    try std.testing.expectEqual(SafeTensorsDType.F32, SafeTensorsDType.fromString("F32").?);
    try std.testing.expectEqual(SafeTensorsDType.BF16, SafeTensorsDType.fromString("BF16").?);
    try std.testing.expect(SafeTensorsDType.fromString("INVALID") == null);
}
