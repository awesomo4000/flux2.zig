const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ─────────────────────────────────────────────────────────────────────────
    // Library: libflux
    // ─────────────────────────────────────────────────────────────────────────
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/flux.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "flux",
        .root_module = lib_mod,
        .linkage = .static,
    });

    // Optional: vendor stb_image for PNG support
    // lib.addIncludePath(b.path("vendor"));
    // lib.addCSourceFile(.{ .file = b.path("vendor/stb_image_impl.c") });
    // lib.linkLibC();

    b.installArtifact(lib);

    // ─────────────────────────────────────────────────────────────────────────
    // Executable: flux CLI
    // ─────────────────────────────────────────────────────────────────────────
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("flux", lib_mod);

    const exe = b.addExecutable(.{
        .name = "flux",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the flux CLI");
    run_step.dependOn(&run_cmd.step);

    // ─────────────────────────────────────────────────────────────────────────
    // Tests
    // ─────────────────────────────────────────────────────────────────────────
    const lib_test_mod = b.createModule(.{
        .root_source_file = b.path("src/flux.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib_tests = b.addTest(.{
        .root_module = lib_test_mod,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const kernel_test_mod = b.createModule(.{
        .root_source_file = b.path("src/kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    const kernel_tests = b.addTest(.{
        .root_module = kernel_test_mod,
    });
    const run_kernel_tests = b.addRunArtifact(kernel_tests);

    const safetensors_test_mod = b.createModule(.{
        .root_source_file = b.path("src/safetensors.zig"),
        .target = target,
        .optimize = optimize,
    });
    const safetensors_tests = b.addTest(.{
        .root_module = safetensors_test_mod,
    });
    const run_safetensors_tests = b.addRunArtifact(safetensors_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_kernel_tests.step);
    test_step.dependOn(&run_safetensors_tests.step);

    // ─────────────────────────────────────────────────────────────────────────
    // Server: flux-server for memory-resident operation
    // ─────────────────────────────────────────────────────────────────────────
    const server_mod = b.createModule(.{
        .root_source_file = b.path("src/server.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_mod.addImport("flux", lib_mod);

    const server = b.addExecutable(.{
        .name = "flux-server",
        .root_module = server_mod,
    });
    b.installArtifact(server);

    const server_cmd = b.addRunArtifact(server);
    server_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| server_cmd.addArgs(args);

    const server_step = b.step("server", "Run flux-server");
    server_step.dependOn(&server_cmd.step);

    // ─────────────────────────────────────────────────────────────────────────
    // Benchmarks
    // ─────────────────────────────────────────────────────────────────────────
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("flux", lib_mod);

    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = bench_mod,
    });

    const bench_cmd = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench_cmd.step);

    // ─────────────────────────────────────────────────────────────────────────
    // HuggingFace Downloader: hfd
    // ─────────────────────────────────────────────────────────────────────────
    const hfd_mod = b.createModule(.{
        .root_source_file = b.path("src/hfd.zig"),
        .target = target,
        .optimize = optimize,
    });

    const hfd = b.addExecutable(.{
        .name = "hfd",
        .root_module = hfd_mod,
    });
    b.installArtifact(hfd);

    const hfd_cmd = b.addRunArtifact(hfd);
    hfd_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| hfd_cmd.addArgs(args);

    const hfd_step = b.step("hfd", "Run HuggingFace downloader");
    hfd_step.dependOn(&hfd_cmd.step);

    // hfd tests
    const hfd_test_mod = b.createModule(.{
        .root_source_file = b.path("src/hfd.zig"),
        .target = target,
        .optimize = optimize,
    });
    const hfd_tests = b.addTest(.{
        .root_module = hfd_test_mod,
    });
    test_step.dependOn(&b.addRunArtifact(hfd_tests).step);
}
