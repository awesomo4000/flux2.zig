//! flux server - Keep model loaded for multiple generations
//!
//! Modes:
//! - REPL: Interactive stdin/stdout
//! - HTTP: REST API server
//! - Socket: Unix domain socket
//!
//! The model stays in memory between requests, avoiding the ~30s load time.

const std = @import("std");
const Allocator = std.mem.Allocator;
const flux = @import("flux.zig");

pub const Server = struct {
    allocator: Allocator,
    ctx: *flux.FluxContext,
    config: Config,

    // Prompt cache for faster repeated generations
    last_prompt_hash: u64 = 0,

    pub const Config = struct {
        mode: Mode = .repl,
        port: u16 = 8080,
        socket_path: ?[]const u8 = null,
        keep_encoder: bool = false,
    };

    pub const Mode = enum {
        repl,
        http,
        socket,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, model_dir: []const u8, config: Config) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = Self{
            .allocator = allocator,
            .ctx = try flux.FluxContext.loadDir(allocator, model_dir),
            .config = config,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
        self.allocator.destroy(self);
    }

    pub fn run(self: *Self) !void {
        switch (self.config.mode) {
            .repl => try self.runRepl(),
            .http => try self.runHttp(),
            .socket => try self.runSocket(),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // REPL Mode
    // ─────────────────────────────────────────────────────────────────────────

    fn runRepl(self: *Self) !void {
        var stdin_buf: [4096]u8 = undefined;
        var stdin_reader = std.fs.File.stdin().reader(&stdin_buf);
        const stdin = &stdin_reader.interface;

        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
        const stdout = &stdout_writer.interface;

        try stdout.print("flux server (REPL mode)\n", .{});
        try stdout.print("Model loaded. Enter JSON requests or 'quit' to exit.\n\n", .{});
        try stdout.print("Example: {{\"prompt\": \"A cat\", \"output\": \"cat.ppm\"}}\n\n", .{});
        try stdout.flush();

        while (true) {
            try stdout.print("> ", .{});
            try stdout.flush();

            const line = stdin.takeDelimiterExclusive('\n') catch |err| {
                if (err == error.ReadFailed) break;
                return err;
            };

            const trimmed = std.mem.trim(u8, line, " \t\r\n");
            if (trimmed.len == 0) continue;

            if (std.mem.eql(u8, trimmed, "quit") or std.mem.eql(u8, trimmed, "exit")) {
                try stdout.print("Goodbye!\n", .{});
                try stdout.flush();
                break;
            }

            if (std.mem.eql(u8, trimmed, "help")) {
                try self.printReplHelp(stdout);
                try stdout.flush();
                continue;
            }

            if (std.mem.eql(u8, trimmed, "status")) {
                try self.printStatus(stdout);
                try stdout.flush();
                continue;
            }

            // Parse JSON request
            self.handleJsonRequest(trimmed, stdout) catch |err| {
                try stdout.print("Error: {}\n", .{err});
            };
            try stdout.flush();
        }
    }

    fn handleJsonRequest(self: *Self, json: []const u8, writer: anytype) !void {
        var parsed = std.json.parseFromSlice(
            Request,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        ) catch {
            try writer.print("Invalid JSON. Use: {{\"prompt\": \"...\", \"output\": \"file.ppm\"}}\n", .{});
            return;
        };
        defer parsed.deinit();

        const req = parsed.value;

        const params = flux.Params{
            .width = req.width,
            .height = req.height,
            .num_steps = req.steps,
            .seed = req.seed,
            .strength = req.strength,
        };

        const start = std.time.milliTimestamp();

        var image = try self.ctx.generate(req.prompt, params);
        defer image.deinit();

        const elapsed = std.time.milliTimestamp() - start;

        try image.save(req.output);

        try writer.print("Generated: {s} ({d}x{d}) in {d}ms\n", .{
            req.output,
            image.width,
            image.height,
            elapsed,
        });
    }

    fn printReplHelp(self: *Self, writer: anytype) !void {
        _ = self;
        try writer.print(
            \\Commands:
            \\  help     Show this help
            \\  status   Show model status
            \\  quit     Exit server
            \\
            \\Request format (JSON):
            \\  {{
            \\    "prompt": "A fluffy cat",     // Required
            \\    "output": "cat.ppm",          // Required
            \\    "width": 256,                 // Optional (default: 256)
            \\    "height": 256,                // Optional (default: 256)
            \\    "steps": 4,                   // Optional (default: 4)
            \\    "seed": 42                    // Optional (random if omitted)
            \\  }}
            \\
        , .{});
    }

    fn printStatus(self: *Self, writer: anytype) !void {
        _ = self;
        try writer.print(
            \\Model Status:
            \\  VAE:          loaded
            \\  Transformer:  loaded
            \\  Text Encoder: loaded (auto-releases after first gen)
            \\
        , .{});
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HTTP Mode (stub)
    // ─────────────────────────────────────────────────────────────────────────

    fn runHttp(self: *Self) !void {
        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
        const stdout = &stdout_writer.interface;

        try stdout.print("Starting HTTP server on port {d}...\n", .{self.config.port});

        // TODO: Implement HTTP server using std.http.Server
        // For now, just print instructions
        try stdout.print(
            \\
            \\HTTP server not yet implemented.
            \\
            \\Planned API:
            \\  POST /generate
            \\    Request:  {{"prompt": "...", "width": 256, "height": 256}}
            \\    Response: PNG image bytes
            \\
            \\  GET /health
            \\    Response: {{"status": "ok", "model_loaded": true}}
            \\
        , .{});
        try stdout.flush();

        return error.NotImplemented;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Unix Socket Mode (stub)
    // ─────────────────────────────────────────────────────────────────────────

    fn runSocket(self: *Self) !void {
        const path = self.config.socket_path orelse "/tmp/flux.sock";

        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
        const stdout = &stdout_writer.interface;

        try stdout.print("Unix socket mode not yet implemented.\n", .{});
        try stdout.print("Would listen on: {s}\n", .{path});
        try stdout.flush();
        return error.NotImplemented;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Request/Response Types
// ─────────────────────────────────────────────────────────────────────────────

pub const Request = struct {
    prompt: []const u8,
    output: []const u8 = "output.ppm",
    width: u32 = 256,
    height: u32 = 256,
    steps: usize = 4,
    seed: ?u64 = null,
    strength: f32 = 0.75,
    input: ?[]const u8 = null, // For img2img
};

pub const Response = struct {
    success: bool,
    output: ?[]const u8 = null,
    seed: u64,
    elapsed_ms: u64,
    @"error": ?[]const u8 = null,
};

// ─────────────────────────────────────────────────────────────────────────────
// Server CLI Entry Point
// ─────────────────────────────────────────────────────────────────────────────

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args_iter = try std.process.argsWithAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.skip(); // program name

    var model_dir: ?[]const u8 = null;
    var config = Server.Config{};

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--dir")) {
            model_dir = args_iter.next();
        } else if (std.mem.eql(u8, arg, "--port")) {
            if (args_iter.next()) |port_str| {
                config.port = std.fmt.parseInt(u16, port_str, 10) catch 8080;
            }
            config.mode = .http;
        } else if (std.mem.eql(u8, arg, "--socket")) {
            config.socket_path = args_iter.next();
            config.mode = .socket;
        } else if (std.mem.eql(u8, arg, "--repl")) {
            config.mode = .repl;
        } else if (std.mem.eql(u8, arg, "--keep-encoder")) {
            config.keep_encoder = true;
        }
    }

    const dir = model_dir orelse {
        std.debug.print("Error: --dir is required\n\n", .{});
        printUsage();
        std.process.exit(1);
    };

    var server = try Server.init(allocator, dir, config);
    defer server.deinit();

    try server.run();
}

fn printUsage() void {
    const usage =
        \\flux-server - Keep model in memory for fast repeated generations
        \\
        \\USAGE:
        \\    flux-server -d <model_dir> [options]
        \\
        \\REQUIRED:
        \\    -d, --dir PATH        Path to model directory
        \\
        \\MODE (pick one):
        \\    --repl                Interactive mode (default)
        \\    --port PORT           HTTP server on port
        \\    --socket PATH         Unix domain socket
        \\
        \\OPTIONS:
        \\    --keep-encoder        Don't auto-release text encoder
        \\    -h, --help            Show this help
        \\
        \\EXAMPLES:
        \\    flux-server -d ./flux-klein-model --repl
        \\    flux-server -d ./flux-klein-model --port 8080
        \\
        \\REPL COMMANDS:
        \\    Enter JSON: {"prompt": "A cat", "output": "cat.ppm"}
        \\    help        Show commands
        \\    status      Show model status  
        \\    quit        Exit
        \\
    ;
    std.debug.print("{s}", .{usage});
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "request parsing" {
    const json =
        \\{"prompt": "test", "output": "out.ppm", "width": 512}
    ;

    var parsed = try std.json.parseFromSlice(
        Request,
        std.testing.allocator,
        json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    try std.testing.expectEqualStrings("test", parsed.value.prompt);
    try std.testing.expectEqual(@as(u32, 512), parsed.value.width);
    try std.testing.expectEqual(@as(u32, 256), parsed.value.height); // default
}
