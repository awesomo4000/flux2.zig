//! Benchmarks for flux2.zig kernels
//!
//! Run with: zig build bench

const std = @import("std");
const flux = @import("flux");
const kernels = flux.kernels;
const Tensor = flux.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("flux2.zig Kernel Benchmarks\n", .{});
    std.debug.print("SIMD vector size: {d}\n\n", .{kernels.VEC_SIZE});

    try benchMatmul(allocator);
    try benchGelu(allocator);
    try benchSoftmax(allocator);
    try benchRmsNorm(allocator);
}

fn benchMatmul(allocator: std.mem.Allocator) !void {
    const sizes = [_][3]usize{
        .{ 64, 64, 64 },
        .{ 256, 256, 256 },
        .{ 512, 512, 512 },
        .{ 1024, 1024, 1024 },
    };

    std.debug.print("Matrix Multiplication (C = A @ B):\n", .{});

    for (sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        // Allocate matrices
        const A = try allocator.alloc(f32, M * K);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, K * N);
        defer allocator.free(B);
        const C = try allocator.alloc(f32, M * N);
        defer allocator.free(C);

        // Initialize with random values
        var rng = std.Random.DefaultPrng.init(42);
        for (A) |*x| x.* = rng.random().float(f32);
        for (B) |*x| x.* = rng.random().float(f32);

        // Warmup
        kernels.matmul(C, A, B, M, N, K);

        // Benchmark
        const iterations = 10;
        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            kernels.matmul(C, A, B, M, N, K);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(iterations));

        // Calculate GFLOPS (2*M*N*K operations for matmul)
        const flops = 2 * M * N * K;
        const gflops = @as(f64, @floatFromInt(flops)) / (elapsed_ms * 1_000_000.0);

        std.debug.print("  {d}x{d}x{d}: {d:.2} ms, {d:.2} GFLOPS\n", .{ M, N, K, elapsed_ms, gflops });
    }
    std.debug.print("\n", .{});
}

fn benchGelu(allocator: std.mem.Allocator) !void {
    const sizes = [_]usize{ 1024, 4096, 16384, 65536 };

    std.debug.print("GELU Activation:\n", .{});

    for (sizes) |size| {
        const x = try allocator.alloc(f32, size);
        defer allocator.free(x);

        var rng = std.Random.DefaultPrng.init(42);
        for (x) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

        // Warmup
        kernels.gelu(x);

        // Re-initialize for benchmark
        for (x) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

        const iterations = 1000;
        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            kernels.gelu(x);
        }

        const elapsed_ns = timer.read();
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(iterations));
        const throughput = @as(f64, @floatFromInt(size)) / elapsed_us; // elements per microsecond

        std.debug.print("  n={d}: {d:.2} µs, {d:.1} M elem/s\n", .{ size, elapsed_us, throughput });
    }
    std.debug.print("\n", .{});
}

fn benchSoftmax(allocator: std.mem.Allocator) !void {
    const sizes = [_]usize{ 128, 512, 2048, 8192 };

    std.debug.print("Softmax:\n", .{});

    for (sizes) |size| {
        const x = try allocator.alloc(f32, size);
        defer allocator.free(x);

        var rng = std.Random.DefaultPrng.init(42);

        const iterations = 1000;
        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            for (x) |*v| v.* = rng.random().float(f32) * 10.0 - 5.0;
            kernels.softmax(x, size);
        }

        const elapsed_ns = timer.read();
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(iterations));

        std.debug.print("  n={d}: {d:.2} µs\n", .{ size, elapsed_us });
    }
    std.debug.print("\n", .{});
}

fn benchRmsNorm(allocator: std.mem.Allocator) !void {
    const sizes = [_]usize{ 256, 1024, 2560, 4096 }; // Common hidden sizes

    std.debug.print("RMS Normalization:\n", .{});

    for (sizes) |size| {
        const x = try allocator.alloc(f32, size);
        defer allocator.free(x);
        const out = try allocator.alloc(f32, size);
        defer allocator.free(out);
        const weight = try allocator.alloc(f32, size);
        defer allocator.free(weight);

        var rng = std.Random.DefaultPrng.init(42);
        for (x) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
        for (weight) |*v| v.* = 1.0;

        // Warmup
        kernels.rmsNorm(out, x, weight, 1e-6);

        const iterations = 10000;
        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            kernels.rmsNorm(out, x, weight, 1e-6);
        }

        const elapsed_ns = timer.read();
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0 / @as(f64, @floatFromInt(iterations));

        std.debug.print("  hidden_size={d}: {d:.2} µs\n", .{ size, elapsed_us });
    }
    std.debug.print("\n", .{});
}
