//! flux.zig - Main API for FLUX.2 image generation
//!
//! Pure Zig implementation of FLUX.2-klein-4B inference.
//! This is the public API for the library.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Tensor = @import("tensor.zig").Tensor;
pub const Image = @import("image.zig").Image;
pub const kernels = @import("kernels.zig");
pub const SafeTensors = @import("safetensors.zig").SafeTensors;

/// Generation parameters
pub const Params = struct {
    /// Output width in pixels (must be multiple of 16)
    width: u32 = 256,
    /// Output height in pixels (must be multiple of 16)
    height: u32 = 256,
    /// Number of denoising steps (4 is optimal for klein model)
    num_steps: usize = 4,
    /// Classifier-free guidance scale (1.0 for klein)
    guidance_scale: f32 = 1.0,
    /// Random seed (null for random)
    seed: ?u64 = null,
    /// Image-to-image strength (0.0 = no change, 1.0 = full regeneration)
    strength: f32 = 0.75,
};

/// Default parameters optimized for FLUX.2-klein
pub const PARAMS_DEFAULT = Params{};

/// Flux model context
pub const FluxContext = struct {
    allocator: Allocator,

    // Model components (loaded on demand)
    vae: ?VAE = null,
    transformer: ?Transformer = null,
    text_encoder: ?TextEncoder = null,
    tokenizer: ?Tokenizer = null,

    // Model directory
    model_dir: []const u8,

    // Random number generator
    rng: std.Random.DefaultPrng,

    const Self = @This();

    /// Load model from directory
    pub fn loadDir(allocator: Allocator, model_dir: []const u8) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = Self{
            .allocator = allocator,
            .model_dir = try allocator.dupe(u8, model_dir),
            .rng = std.Random.DefaultPrng.init(@bitCast(std.time.timestamp())),
        };

        // Load components lazily or eagerly based on preference
        // For now, verify model directory exists
        var dir = std.fs.cwd().openDir(model_dir, .{}) catch |err| {
            std.log.err("Failed to open model directory '{s}': {}", .{ model_dir, err });
            return error.ModelDirNotFound;
        };
        dir.close();

        return self;
    }

    /// Free all resources
    pub fn deinit(self: *Self) void {
        if (self.vae) |*v| v.deinit();
        if (self.transformer) |*t| t.deinit();
        if (self.text_encoder) |*e| e.deinit();
        if (self.tokenizer) |*t| t.deinit();
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }

    /// Generate an image from a text prompt
    pub fn generate(self: *Self, prompt: []const u8, params: Params) !Image {
        const actual_seed = params.seed orelse @as(u64, @bitCast(std.time.timestamp()));
        self.rng = std.Random.DefaultPrng.init(actual_seed);

        std.log.info("Generating image with seed: {d}", .{actual_seed});
        std.log.info("Prompt: \"{s}\"", .{prompt});
        std.log.info("Size: {d}x{d}, steps: {d}", .{ params.width, params.height, params.num_steps });

        // Validate dimensions
        if (params.width % 16 != 0 or params.height % 16 != 0) {
            return error.InvalidDimensions;
        }
        if (params.width > 1024 or params.height > 1024) {
            return error.DimensionsTooLarge;
        }

        // TODO: Implement actual inference pipeline:
        // 1. Tokenize prompt
        // 2. Encode text with Qwen3
        // 3. Initialize random latents
        // 4. Run diffusion sampling
        // 5. Decode latents with VAE
        // 6. Convert to image

        // For now, return a placeholder
        std.log.warn("Inference not yet implemented - returning placeholder", .{});
        return self.createPlaceholder(params.width, params.height);
    }

    /// Transform an existing image based on a prompt
    pub fn img2img(
        self: *Self,
        prompt: []const u8,
        input: *const Image,
        params: Params,
    ) !Image {
        const actual_seed = params.seed orelse @as(u64, @bitCast(std.time.timestamp()));
        self.rng = std.Random.DefaultPrng.init(actual_seed);

        std.log.info("Image-to-image with seed: {d}, strength: {d:.2}", .{ actual_seed, params.strength });
        std.log.info("Prompt: \"{s}\"", .{prompt});
        std.log.info("Input size: {d}x{d}", .{ input.width, input.height });

        // TODO: Implement img2img pipeline:
        // 1. Encode input image to latents
        // 2. Add noise based on strength
        // 3. Run diffusion from noisy latents
        // 4. Decode and return

        // For now, return a copy of input
        std.log.warn("img2img not yet implemented - returning input copy", .{});
        return input.resize(self.allocator, params.width, params.height);
    }

    /// Manually release text encoder to free ~8GB memory
    pub fn releaseTextEncoder(self: *Self) void {
        if (self.text_encoder) |*e| {
            e.deinit();
            self.text_encoder = null;
            std.log.info("Text encoder released", .{});
        }
    }

    /// Set random seed for reproducibility
    pub fn setSeed(self: *Self, seed: u64) void {
        self.rng = std.Random.DefaultPrng.init(seed);
    }

    // Create a placeholder image (gradient for testing)
    fn createPlaceholder(self: *Self, width: u32, height: u32) !Image {
        var img = try Image.init(self.allocator, width, height, 3);

        for (0..height) |y| {
            for (0..width) |x| {
                const r: u8 = @intCast((x * 255) / width);
                const g: u8 = @intCast((y * 255) / height);
                const b: u8 = @intCast(((x + y) * 128) / (width + height));
                img.setPixel(@intCast(x), @intCast(y), .{ r, g, b });
            }
        }

        return img;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Component Stubs (to be implemented)
// ─────────────────────────────────────────────────────────────────────────────

/// VAE encoder/decoder
pub const VAE = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator, model_path: []const u8) !VAE {
        _ = model_path;
        return VAE{ .allocator = allocator };
    }

    pub fn deinit(self: *VAE) void {
        _ = self;
    }

    pub fn encode(self: *VAE, image: *const Image) !Tensor {
        _ = self;
        _ = image;
        return error.NotImplemented;
    }

    pub fn decode(self: *VAE, latents: *const Tensor) !Tensor {
        _ = self;
        _ = latents;
        return error.NotImplemented;
    }
};

/// Flux transformer
pub const Transformer = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator, model_path: []const u8) !Transformer {
        _ = model_path;
        return Transformer{ .allocator = allocator };
    }

    pub fn deinit(self: *Transformer) void {
        _ = self;
    }

    pub fn forward(
        self: *Transformer,
        img_latents: *Tensor,
        txt_embeddings: *const Tensor,
        timesteps: *const Tensor,
    ) !Tensor {
        _ = self;
        _ = img_latents;
        _ = txt_embeddings;
        _ = timesteps;
        return error.NotImplemented;
    }
};

/// Qwen3 text encoder
pub const TextEncoder = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator, model_path: []const u8) !TextEncoder {
        _ = model_path;
        return TextEncoder{ .allocator = allocator };
    }

    pub fn deinit(self: *TextEncoder) void {
        _ = self;
    }

    pub fn encode(self: *TextEncoder, token_ids: []const u32) !Tensor {
        _ = self;
        _ = token_ids;
        return error.NotImplemented;
    }
};

/// BPE Tokenizer
pub const Tokenizer = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator, vocab_path: []const u8) !Tokenizer {
        _ = vocab_path;
        return Tokenizer{ .allocator = allocator };
    }

    pub fn deinit(self: *Tokenizer) void {
        _ = self;
    }

    pub fn encode(self: *Tokenizer, text: []const u8) ![]u32 {
        _ = self;
        _ = text;
        return error.NotImplemented;
    }

    pub fn decode(self: *Tokenizer, ids: []const u32) ![]u8 {
        _ = self;
        _ = ids;
        return error.NotImplemented;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Error handling
// ─────────────────────────────────────────────────────────────────────────────

pub const FluxError = error{
    ModelDirNotFound,
    ModelLoadFailed,
    InvalidDimensions,
    DimensionsTooLarge,
    NotImplemented,
    OutOfMemory,
    EncodingFailed,
    DecodingFailed,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "params default" {
    const params = PARAMS_DEFAULT;
    try std.testing.expectEqual(@as(u32, 256), params.width);
    try std.testing.expectEqual(@as(usize, 4), params.num_steps);
}

test "import submodules" {
    // Ensure all submodules compile
    _ = Tensor;
    _ = Image;
    _ = kernels;
    _ = SafeTensors;
}
