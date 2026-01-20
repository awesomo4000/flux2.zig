//! Test loading the FLUX.2 transformer from safetensors

const std = @import("std");
const Transformer = @import("transformer.zig").Transformer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_writer.interface;

    try stdout.print("Loading FLUX.2-klein-4B transformer...\n", .{});
    try stdout.flush();

    const tf = Transformer.load(allocator, "FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors") catch |err| {
        try stdout.print("Failed to load: {s}\n", .{@errorName(err)});
        try stdout.flush();
        return err;
    };
    defer tf.deinit();

    try stdout.print("Transformer loaded successfully!\n", .{});
    try stdout.print("  hidden_size: {d}\n", .{tf.config.hidden_size});
    try stdout.print("  num_heads: {d}\n", .{tf.config.num_heads});
    try stdout.print("  num_double_layers: {d}\n", .{tf.config.num_double_layers});
    try stdout.print("  num_single_layers: {d}\n", .{tf.config.num_single_layers});
    try stdout.flush();

    // Check weight shapes
    try stdout.print("\nWeight shapes:\n", .{});
    try stdout.print("  img_in_weight: {d} elements\n", .{tf.img_in_weight.len});
    try stdout.print("  txt_in_weight: {d} elements\n", .{tf.txt_in_weight.len});
    try stdout.print("  time_embed.fc1_weight: {d} elements\n", .{tf.time_embed.fc1_weight.len});
    try stdout.print("  adaln_double_img_weight: {d} elements\n", .{tf.adaln_double_img_weight.len});
    try stdout.flush();

    try stdout.print("\nDone!\n", .{});
    try stdout.flush();
}
