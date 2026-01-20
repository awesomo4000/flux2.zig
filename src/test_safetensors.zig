//! Quick test to load and inspect the FLUX.2-klein-4B transformer safetensors

const std = @import("std");
const SafeTensors = @import("safetensors.zig").SafeTensors;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Stdout writer
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_writer.interface;

    try stdout.print("Opening safetensors file...\n", .{});
    try stdout.flush();

    var st = try SafeTensors.open(allocator, "FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors");
    defer st.close();

    try stdout.print("Header size: {d} bytes\n", .{st.header_size});
    try stdout.print("Number of tensors: {d}\n", .{st.tensors.count()});
    try stdout.flush();

    // List ALL tensors
    const names = try st.listTensors(allocator);
    defer allocator.free(names);

    // Sort for consistent output
    std.mem.sort([]const u8, names, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    try stdout.print("\n=== ALL TENSORS ===\n", .{});
    try stdout.flush();

    for (names) |name| {
        const info = st.getTensorInfo(name).?;
        try stdout.print("{s}: {s} [", .{ name, @tagName(info.dtype) });
        for (info.shape, 0..) |dim, i| {
            if (i > 0) try stdout.print(", ", .{});
            try stdout.print("{d}", .{dim});
        }
        try stdout.print("]\n", .{});
        try stdout.flush();
    }

    // Check for expected tensor names (what C code expects)
    try stdout.print("\n=== CHECKING EXPECTED NAMES ===\n", .{});
    try stdout.flush();

    const expected_names = [_][]const u8{
        // Input projections
        "x_embedder.weight",
        "context_embedder.weight",
        // Time embedding
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        // Modulation
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.linear.weight",
        // Final layer
        "norm_out.linear.weight",
        "proj_out.weight",
        // Double block 0
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.ff.linear_in.weight",
        // Single block 0
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
        "single_transformer_blocks.0.attn.norm_q.weight",
    };

    for (expected_names) |name| {
        const found = st.hasTensor(name);
        try stdout.print("  {s}: {s}\n", .{ name, if (found) "FOUND" else "MISSING" });
    }
    try stdout.flush();

    // Try loading a small tensor to verify data conversion works
    try stdout.print("\n=== LOADING SAMPLE TENSOR ===\n", .{});
    try stdout.flush();

    for (names) |name| {
        const info = st.getTensorInfo(name).?;
        var numel: usize = 1;
        for (info.shape) |s| numel *= s;
        const size_bytes = numel * info.dtype.size();

        if (size_bytes < 1024 * 1024 and numel > 1) {
            try stdout.print("Loading '{s}' ({d} elements, {d} bytes)...\n", .{ name, numel, size_bytes });
            try stdout.flush();

            var tensor = try st.getTensor(allocator, name);
            defer tensor.deinit();

            try stdout.print("  First 10 values: [", .{});
            const to_show = @min(10, tensor.data.len);
            for (0..to_show) |i| {
                if (i > 0) try stdout.print(", ", .{});
                try stdout.print("{d:.4}", .{tensor.data[i]});
            }
            try stdout.print("]\n", .{});
            try stdout.flush();
            break;
        }
    }

    try stdout.print("\nDone!\n", .{});
    try stdout.flush();
}
