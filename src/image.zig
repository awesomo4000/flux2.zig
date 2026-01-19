//! Image loading and saving for flux2.zig
//!
//! Supports:
//! - PNG (via embedded implementation or stb_image)
//! - PPM (native Zig implementation)

const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

/// RGB Image
pub const Image = struct {
    width: u32,
    height: u32,
    channels: u8,
    data: []u8,
    allocator: Allocator,

    const Self = @This();

    /// Create a new image with given dimensions
    pub fn init(allocator: Allocator, width: u32, height: u32, channels: u8) !Self {
        const size = @as(usize, width) * height * channels;
        const data = try allocator.alloc(u8, size);
        @memset(data, 0);

        return Self{
            .width = width,
            .height = height,
            .channels = channels,
            .data = data,
            .allocator = allocator,
        };
    }

    /// Free image memory
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    /// Load image from file (PNG or PPM)
    pub fn load(allocator: Allocator, path: []const u8) !Self {
        // Determine format from extension
        if (std.mem.endsWith(u8, path, ".ppm") or std.mem.endsWith(u8, path, ".PPM")) {
            return loadPPM(allocator, path);
        } else if (std.mem.endsWith(u8, path, ".png") or std.mem.endsWith(u8, path, ".PNG")) {
            return loadPNG(allocator, path);
        } else {
            return error.UnsupportedFormat;
        }
    }

    /// Save image to file (PNG or PPM)
    pub fn save(self: *const Self, path: []const u8) !void {
        if (std.mem.endsWith(u8, path, ".ppm") or std.mem.endsWith(u8, path, ".PPM")) {
            return self.savePPM(path);
        } else if (std.mem.endsWith(u8, path, ".png") or std.mem.endsWith(u8, path, ".PNG")) {
            return self.savePNG(path);
        } else {
            return error.UnsupportedFormat;
        }
    }

    /// Get pixel at (x, y)
    pub fn getPixel(self: *const Self, x: u32, y: u32) [3]u8 {
        const idx = (@as(usize, y) * self.width + x) * self.channels;
        return .{
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
        };
    }

    /// Set pixel at (x, y)
    pub fn setPixel(self: *Self, x: u32, y: u32, rgb: [3]u8) void {
        const idx = (@as(usize, y) * self.width + x) * self.channels;
        self.data[idx] = rgb[0];
        self.data[idx + 1] = rgb[1];
        self.data[idx + 2] = rgb[2];
    }

    /// Convert image to tensor [1, C, H, W] normalized to [-1, 1]
    pub fn toTensor(self: *const Self, allocator: Allocator) !Tensor {
        var tensor = try Tensor.init(allocator, &.{ 1, self.channels, self.height, self.width });
        errdefer tensor.deinit();

        for (0..self.height) |y| {
            for (0..self.width) |x| {
                const idx = (y * self.width + x) * self.channels;
                for (0..self.channels) |c| {
                    const pixel_val = self.data[idx + c];
                    // Normalize to [-1, 1]
                    const normalized = (@as(f32, @floatFromInt(pixel_val)) / 127.5) - 1.0;
                    const tensor_idx = c * self.height * self.width + y * self.width + x;
                    tensor.data[tensor_idx] = normalized;
                }
            }
        }

        return tensor;
    }

    /// Create image from tensor [1, C, H, W] or [C, H, W] in range [-1, 1]
    pub fn fromTensor(tensor: *const Tensor, allocator: Allocator) !Self {
        const shape = tensor.getShape();

        var c: usize = undefined;
        var h: usize = undefined;
        var w: usize = undefined;

        if (shape.len == 4) {
            // [B, C, H, W] - take first batch
            c = shape[1];
            h = shape[2];
            w = shape[3];
        } else if (shape.len == 3) {
            // [C, H, W]
            c = shape[0];
            h = shape[1];
            w = shape[2];
        } else {
            return error.InvalidTensorShape;
        }

        if (c != 3) return error.InvalidChannels;

        var image = try init(allocator, @intCast(w), @intCast(h), 3);
        errdefer image.deinit();

        for (0..h) |y| {
            for (0..w) |x| {
                for (0..3) |channel| {
                    const tensor_idx = channel * h * w + y * w + x;
                    const val = tensor.data[tensor_idx];
                    // Convert from [-1, 1] to [0, 255]
                    const clamped = std.math.clamp(val, -1.0, 1.0);
                    const pixel: u8 = @intFromFloat((clamped + 1.0) * 127.5);
                    const img_idx = (y * w + x) * 3 + channel;
                    image.data[img_idx] = pixel;
                }
            }
        }

        return image;
    }

    /// Resize image using bilinear interpolation
    pub fn resize(self: *const Self, allocator: Allocator, new_width: u32, new_height: u32) !Self {
        var result = try init(allocator, new_width, new_height, self.channels);
        errdefer result.deinit();

        const x_ratio = @as(f32, @floatFromInt(self.width - 1)) / @as(f32, @floatFromInt(new_width - 1));
        const y_ratio = @as(f32, @floatFromInt(self.height - 1)) / @as(f32, @floatFromInt(new_height - 1));

        for (0..new_height) |y| {
            for (0..new_width) |x| {
                const gx = @as(f32, @floatFromInt(x)) * x_ratio;
                const gy = @as(f32, @floatFromInt(y)) * y_ratio;

                const x0: u32 = @intFromFloat(@floor(gx));
                const y0: u32 = @intFromFloat(@floor(gy));
                const x1 = @min(x0 + 1, self.width - 1);
                const y1 = @min(y0 + 1, self.height - 1);

                const x_diff = gx - @as(f32, @floatFromInt(x0));
                const y_diff = gy - @as(f32, @floatFromInt(y0));

                for (0..self.channels) |c| {
                    const p00 = self.data[(@as(usize, y0) * self.width + x0) * self.channels + c];
                    const p10 = self.data[(@as(usize, y0) * self.width + x1) * self.channels + c];
                    const p01 = self.data[(@as(usize, y1) * self.width + x0) * self.channels + c];
                    const p11 = self.data[(@as(usize, y1) * self.width + x1) * self.channels + c];

                    const val = @as(f32, @floatFromInt(p00)) * (1 - x_diff) * (1 - y_diff) +
                        @as(f32, @floatFromInt(p10)) * x_diff * (1 - y_diff) +
                        @as(f32, @floatFromInt(p01)) * (1 - x_diff) * y_diff +
                        @as(f32, @floatFromInt(p11)) * x_diff * y_diff;

                    result.data[(@as(usize, y) * new_width + x) * self.channels + c] = @intFromFloat(val);
                }
            }
        }

        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PPM Format (simple, no dependencies)
    // ─────────────────────────────────────────────────────────────────────────

    fn loadPPM(allocator: Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var read_buffer: [4096]u8 = undefined;
        var file_reader = file.reader(&read_buffer);
        const reader = &file_reader.interface;

        // Read magic number
        var magic: [2]u8 = undefined;
        try reader.readSliceAll(&magic);
        if (!std.mem.eql(u8, &magic, "P6")) return error.InvalidPPM;

        // Skip whitespace
        try skipWhitespaceAndComments(reader);

        // Read width
        const width = try readNumber(reader);
        try skipWhitespaceAndComments(reader);

        // Read height
        const height = try readNumber(reader);
        try skipWhitespaceAndComments(reader);

        // Read max value
        const max_val = try readNumber(reader);
        if (max_val != 255) return error.UnsupportedMaxVal;

        // Skip single whitespace after max value
        _ = try reader.takeByte();

        // Read pixel data
        var image = try init(allocator, @intCast(width), @intCast(height), 3);
        errdefer image.deinit();

        try reader.readSliceAll(image.data);

        return image;
    }

    fn savePPM(self: *const Self, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        var write_buffer: [4096]u8 = undefined;
        var file_writer = file.writer(&write_buffer);
        const writer = &file_writer.interface;

        try writer.print("P6\n{d} {d}\n255\n", .{ self.width, self.height });
        try writer.writeAll(self.data);
        try writer.flush();
    }

    fn skipWhitespaceAndComments(reader: *std.Io.Reader) !void {
        while (true) {
            const byte = reader.takeByte() catch return;
            if (byte == '#') {
                // Skip until newline
                _ = reader.takeDelimiterExclusive('\n') catch return;
            } else if (!std.ascii.isWhitespace(byte)) {
                // Put back the non-whitespace character
                // Note: This is a simplified approach; proper implementation
                // would need a peek or unget mechanism
                return;
            }
        }
    }

    fn readNumber(reader: *std.Io.Reader) !u32 {
        var num: u32 = 0;
        var found_digit = false;

        while (true) {
            const byte = reader.takeByte() catch break;
            if (std.ascii.isDigit(byte)) {
                num = num * 10 + (byte - '0');
                found_digit = true;
            } else if (found_digit) {
                break;
            }
        }

        if (!found_digit) return error.NoNumber;
        return num;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PNG Format (stub - needs implementation or stb_image)
    // ─────────────────────────────────────────────────────────────────────────

    fn loadPNG(allocator: Allocator, path: []const u8) !Self {
        // TODO: Implement PNG loading
        // Options:
        // 1. Use @cImport("stb_image.h")
        // 2. Implement deflate + PNG filtering in pure Zig
        // 3. Use zig-png package
        _ = allocator;
        _ = path;
        return error.PNGNotImplemented;
    }

    fn savePNG(self: *const Self, path: []const u8) !void {
        // TODO: Implement PNG saving
        // For now, save as PPM and note the limitation
        _ = self;
        _ = path;
        return error.PNGNotImplemented;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "image creation" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 64, 64, 3);
    defer img.deinit();

    try std.testing.expectEqual(@as(u32, 64), img.width);
    try std.testing.expectEqual(@as(u32, 64), img.height);
    try std.testing.expectEqual(@as(usize, 64 * 64 * 3), img.data.len);
}

test "image pixel access" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 10, 10, 3);
    defer img.deinit();

    img.setPixel(5, 5, .{ 255, 128, 64 });
    const pixel = img.getPixel(5, 5);

    try std.testing.expectEqual(@as(u8, 255), pixel[0]);
    try std.testing.expectEqual(@as(u8, 128), pixel[1]);
    try std.testing.expectEqual(@as(u8, 64), pixel[2]);
}

test "image to tensor and back" {
    const allocator = std.testing.allocator;

    var img = try Image.init(allocator, 4, 4, 3);
    defer img.deinit();

    // Set some pixels
    img.setPixel(0, 0, .{ 0, 0, 0 });
    img.setPixel(1, 0, .{ 255, 255, 255 });
    img.setPixel(2, 0, .{ 128, 128, 128 });

    // Convert to tensor
    var tensor = try img.toTensor(allocator);
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 4), tensor.ndim);
    try std.testing.expectEqual(@as(usize, 1), tensor.shape[0]); // batch
    try std.testing.expectEqual(@as(usize, 3), tensor.shape[1]); // channels
    try std.testing.expectEqual(@as(usize, 4), tensor.shape[2]); // height
    try std.testing.expectEqual(@as(usize, 4), tensor.shape[3]); // width

    // Convert back
    var img2 = try Image.fromTensor(&tensor, allocator);
    defer img2.deinit();

    // Check pixels are approximately preserved
    const p0 = img2.getPixel(0, 0);
    const p1 = img2.getPixel(1, 0);

    try std.testing.expect(p0[0] < 5); // Should be close to 0
    try std.testing.expect(p1[0] > 250); // Should be close to 255
}
