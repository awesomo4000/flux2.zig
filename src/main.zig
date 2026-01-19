//! flux CLI - Command-line interface for FLUX.2 image generation
//!
//! Usage:
//!   flux -d <model_dir> -p "prompt" -o output.png
//!
//! Options:
//!   -d, --dir PATH        Path to model directory (required)
//!   -p, --prompt TEXT     Text prompt for generation (required)
//!   -o, --output PATH     Output image path (required)
//!   -W, --width N         Output width in pixels (default: 256)
//!   -H, --height N        Output height in pixels (default: 256)
//!   -s, --steps N         Sampling steps (default: 4)
//!   -S, --seed N          Random seed for reproducibility
//!   -i, --input PATH      Input image for img2img
//!   -t, --strength N      Img2img strength, 0.0-1.0 (default: 0.75)
//!   -v, --verbose         Show detailed info
//!   -q, --quiet           Silent mode
//!   -h, --help            Show help

const std = @import("std");
const flux = @import("flux");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse arguments
    var args = try Args.parse(allocator);
    defer args.deinit();

    if (args.help) {
        printUsage();
        return;
    }

    // Validate required arguments
    const model_dir = args.model_dir orelse {
        std.debug.print("Error: --dir is required\n", .{});
        printUsage();
        std.process.exit(1);
    };

    const prompt = args.prompt orelse {
        std.debug.print("Error: --prompt is required\n", .{});
        printUsage();
        std.process.exit(1);
    };

    const output = args.output orelse {
        std.debug.print("Error: --output is required\n", .{});
        printUsage();
        std.process.exit(1);
    };

    // Create parameters
    const params = flux.Params{
        .width = args.width,
        .height = args.height,
        .num_steps = args.steps,
        .seed = args.seed,
        .strength = args.strength,
    };

    // Load model
    if (!args.quiet) {
        std.debug.print("Loading model from: {s}\n", .{model_dir});
    }

    var ctx = flux.FluxContext.loadDir(allocator, model_dir) catch |err| {
        std.debug.print("Error loading model: {}\n", .{err});
        std.process.exit(1);
    };
    defer ctx.deinit();

    // Generate or transform image
    var image: flux.Image = undefined;

    if (args.input) |input_path| {
        // Image-to-image
        if (!args.quiet) {
            std.debug.print("Loading input image: {s}\n", .{input_path});
        }

        var input_image = flux.Image.load(allocator, input_path) catch |err| {
            std.debug.print("Error loading input image: {}\n", .{err});
            std.process.exit(1);
        };
        defer input_image.deinit();

        image = ctx.img2img(prompt, &input_image, params) catch |err| {
            std.debug.print("Error during img2img: {}\n", .{err});
            std.process.exit(1);
        };
    } else {
        // Text-to-image
        image = ctx.generate(prompt, params) catch |err| {
            std.debug.print("Error during generation: {}\n", .{err});
            std.process.exit(1);
        };
    }
    defer image.deinit();

    // Print seed for reproducibility
    const actual_seed = params.seed orelse @as(u64, @bitCast(std.time.timestamp()));
    std.debug.print("Seed: {d}\n", .{actual_seed});

    // Save output
    image.save(output) catch |err| {
        std.debug.print("Error saving image: {}\n", .{err});
        std.process.exit(1);
    };

    std.debug.print("{s}\n", .{output});

    if (args.verbose) {
        std.debug.print("\nGeneration complete:\n", .{});
        std.debug.print("  Size: {d}x{d}\n", .{ image.width, image.height });
        std.debug.print("  Steps: {d}\n", .{params.num_steps});
        std.debug.print("  Seed: {d}\n", .{actual_seed});
    }
}

/// Command-line arguments
const Args = struct {
    allocator: std.mem.Allocator,

    model_dir: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    output: ?[]const u8 = null,
    input: ?[]const u8 = null,
    width: u32 = 256,
    height: u32 = 256,
    steps: usize = 4,
    seed: ?u64 = null,
    strength: f32 = 0.75,
    verbose: bool = false,
    quiet: bool = false,
    help: bool = false,

    // Owned strings
    owned_strings: std.ArrayList([]const u8),

    fn parse(allocator: std.mem.Allocator) !Args {
        var self = Args{
            .allocator = allocator,
            .owned_strings = .{},
        };
        errdefer self.deinit();

        var args_iter = try std.process.argsWithAllocator(allocator);
        defer args_iter.deinit();

        // Skip program name
        _ = args_iter.skip();

        while (args_iter.next()) |arg| {
            if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
                self.help = true;
            } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
                self.verbose = true;
            } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
                self.quiet = true;
            } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--dir")) {
                self.model_dir = args_iter.next();
            } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
                self.prompt = args_iter.next();
            } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
                self.output = args_iter.next();
            } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--input")) {
                self.input = args_iter.next();
            } else if (std.mem.eql(u8, arg, "-W") or std.mem.eql(u8, arg, "--width")) {
                if (args_iter.next()) |val| {
                    self.width = std.fmt.parseInt(u32, val, 10) catch 256;
                }
            } else if (std.mem.eql(u8, arg, "-H") or std.mem.eql(u8, arg, "--height")) {
                if (args_iter.next()) |val| {
                    self.height = std.fmt.parseInt(u32, val, 10) catch 256;
                }
            } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--steps")) {
                if (args_iter.next()) |val| {
                    self.steps = std.fmt.parseInt(usize, val, 10) catch 4;
                }
            } else if (std.mem.eql(u8, arg, "-S") or std.mem.eql(u8, arg, "--seed")) {
                if (args_iter.next()) |val| {
                    self.seed = std.fmt.parseInt(u64, val, 10) catch null;
                }
            } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--strength")) {
                if (args_iter.next()) |val| {
                    self.strength = std.fmt.parseFloat(f32, val) catch 0.75;
                }
            }
        }

        return self;
    }

    fn deinit(self: *Args) void {
        for (self.owned_strings.items) |s| {
            self.allocator.free(s);
        }
        self.owned_strings.deinit(self.allocator);
    }
};

fn printUsage() void {
    const usage =
        \\FLUX.2 Image Generation - Zig Implementation
        \\
        \\Usage:
        \\  flux -d <model_dir> -p "prompt" -o output.png [options]
        \\
        \\Required:
        \\  -d, --dir PATH        Path to model directory
        \\  -p, --prompt TEXT     Text prompt for generation
        \\  -o, --output PATH     Output image path (.png or .ppm)
        \\
        \\Generation options:
        \\  -W, --width N         Output width in pixels (default: 256)
        \\  -H, --height N        Output height in pixels (default: 256)
        \\  -s, --steps N         Sampling steps (default: 4)
        \\  -S, --seed N          Random seed for reproducibility
        \\
        \\Image-to-image:
        \\  -i, --input PATH      Input image for img2img
        \\  -t, --strength N      How much to change, 0.0-1.0 (default: 0.75)
        \\
        \\Output:
        \\  -v, --verbose         Show detailed config and timing
        \\  -q, --quiet           Silent mode
        \\  -h, --help            Show this help
        \\
        \\Examples:
        \\  flux -d flux-klein-model -p "A fluffy cat" -o cat.png
        \\  flux -d flux-klein-model -p "oil painting" -i photo.png -o art.png -t 0.7
        \\
    ;
    std.debug.print("{s}", .{usage});
}
