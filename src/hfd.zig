//! hfd - HuggingFace Downloader 🤗
//!
//! Download models from HuggingFace Hub.
//! Watch out for facehuggers. 👽
//!
//! Usage:
//!   hfd <repo_id> [-o output_dir] [--include pattern] [--exclude pattern]
//!
//! Examples:
//!   hfd black-forest-labs/FLUX.2-klein
//!   hfd black-forest-labs/FLUX.2-klein -o ./my-model
//!   hfd black-forest-labs/FLUX.2-klein --include "*.safetensors"

const std = @import("std");
const Allocator = std.mem.Allocator;
const progress = @import("progress.zig");

/// Format bytes into human-readable size using caller-provided buffer
fn formatSizeLocal(bytes: usize, buf: []u8) []const u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
    var size: f64 = @floatFromInt(bytes);
    var unit_idx: usize = 0;

    while (size >= 1024 and unit_idx < units.len - 1) {
        size /= 1024;
        unit_idx += 1;
    }

    if (unit_idx == 0) {
        return std.fmt.bufPrint(buf, "{d} {s}", .{ bytes, units[0] }) catch "?";
    } else {
        return std.fmt.bufPrint(buf, "{d:.1} {s}", .{ size, units[unit_idx] }) catch "?";
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try Args.parse(allocator);
    defer args.deinit();

    if (args.help) {
        printUsage();
        return;
    }

    const repo_id = args.repo_id orelse {
        std.debug.print("Error: repository ID required\n\n", .{});
        printUsage();
        std.process.exit(1);
    };

    var downloader = try HfDownloader.init(allocator, .{
        .repo_id = repo_id,
        .revision = args.revision,
        .output_dir = args.output_dir orelse blk: {
            // Default: repo name (after /)
            if (std.mem.lastIndexOfScalar(u8, repo_id, '/')) |idx| {
                break :blk repo_id[idx + 1 ..];
            }
            break :blk repo_id;
        },
        .token = args.token orelse std.posix.getenv("HF_TOKEN"),
        .include_patterns = args.include_patterns.items,
        .exclude_patterns = args.exclude_patterns.items,
        .resume_downloads = args.resume_downloads,
        .dry_run = args.dry_run,
        .quiet = args.quiet,
        .verbose = args.verbose,
    });
    defer downloader.deinit();

    downloader.run() catch |err| {
        std.debug.print("Error: {}\n", .{err});
        std.process.exit(1);
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// HuggingFace Downloader
// ─────────────────────────────────────────────────────────────────────────────

pub const HfDownloader = struct {
    allocator: Allocator,
    config: Config,
    http_client: std.http.Client,

    pub const Config = struct {
        repo_id: []const u8,
        revision: []const u8,
        output_dir: []const u8,
        token: ?[]const u8,
        include_patterns: []const []const u8,
        exclude_patterns: []const []const u8,
        resume_downloads: bool,
        dry_run: bool,
        quiet: bool,
        verbose: bool,
        // Parallel download settings
        parallel_chunks: usize = 4, // Number of parallel chunks per file
        parallel_threshold: usize = 5 * 1024 * 1024, // Min file size for parallel (5MB)
    };

    const Self = @This();

    pub fn init(allocator: Allocator, config: Config) !Self {
        return Self{
            .allocator = allocator,
            .config = config,
            // Use larger buffers for TLS compatibility (TLS records are up to 16k)
            .http_client = std.http.Client{
                .allocator = allocator,
                .tls_buffer_size = 32 * 1024, // 32k for TLS
                .read_buffer_size = 32 * 1024, // 32k for reading
                .write_buffer_size = 8 * 1024, // 8k for writing
            },
        };
    }

    pub fn deinit(self: *Self) void {
        self.http_client.deinit();
    }

    pub fn run(self: *Self) !void {
        if (!self.config.quiet) {
            std.debug.print("Fetching file list for {s}...\n", .{self.config.repo_id});
        }

        // List files in repository
        const files = try self.listFiles();
        defer {
            for (files) |f| f.deinit(self.allocator);
            self.allocator.free(files);
        }

        // Filter files
        var to_download: std.ArrayList(FileInfo) = .{};
        defer to_download.deinit(self.allocator);

        var total_size: usize = 0;
        for (files) |file| {
            if (self.shouldDownload(file)) {
                try to_download.append(self.allocator, file);
                total_size += file.size;
            }
        }

        if (!self.config.quiet) {
            std.debug.print("Found {d} files ({s} total)\n\n", .{
                to_download.items.len,
                formatSize(total_size),
            });
        }

        if (self.config.dry_run) {
            std.debug.print("Files to download:\n", .{});
            for (to_download.items) |file| {
                std.debug.print("  {s} ({s})\n", .{ file.path, formatSize(file.size) });
            }
            return;
        }

        // Create output directory
        std.fs.cwd().makePath(self.config.output_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        // Download files
        var downloaded: usize = 0;
        var failed: usize = 0;

        for (to_download.items) |file| {
            self.downloadFile(file) catch |err| {
                std.debug.print("\nError downloading {s}: {}\n", .{ file.path, err });
                failed += 1;
                continue;
            };
            downloaded += 1;
        }

        if (!self.config.quiet) {
            std.debug.print("\n\nDownload complete: {s}/ ({d} files", .{
                self.config.output_dir,
                downloaded,
            });
            if (failed > 0) {
                std.debug.print(", {d} failed", .{failed});
            }
            std.debug.print(")\n", .{});
        }
    }

    fn listFiles(self: *Self) ![]FileInfo {
        var result: std.ArrayList(FileInfo) = .{};
        errdefer {
            for (result.items) |f| f.deinit(self.allocator);
            result.deinit(self.allocator);
        }

        try self.listFilesRecursive(&result, "");

        return result.toOwnedSlice(self.allocator);
    }

    fn listFilesRecursive(self: *Self, result: *std.ArrayList(FileInfo), prefix: []const u8) !void {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "https://huggingface.co/api/models/{s}/tree/{s}{s}{s}",
            .{
                self.config.repo_id,
                self.config.revision,
                if (prefix.len > 0) "/" else "",
                prefix,
            },
        );
        defer self.allocator.free(url);

        const body = try self.httpGet(url);
        defer self.allocator.free(body);

        var parsed = std.json.parseFromSlice(
            []const JsonFileEntry,
            self.allocator,
            body,
            .{ .ignore_unknown_fields = true },
        ) catch |err| {
            if (self.config.verbose) {
                std.debug.print("JSON parse error: {}\nBody: {s}\n", .{ err, body });
            }
            return err;
        };
        defer parsed.deinit();

        for (parsed.value) |entry| {
            if (std.mem.eql(u8, entry.type, "directory")) {
                // Recurse into subdirectory
                try self.listFilesRecursive(result, entry.path);
            } else if (std.mem.eql(u8, entry.type, "file")) {
                const sha256: ?[]const u8 = if (entry.lfs) |lfs|
                    if (lfs.oid) |oid| try self.allocator.dupe(u8, oid) else null
                else
                    null;
                try result.append(self.allocator, FileInfo{
                    .path = try self.allocator.dupe(u8, entry.path),
                    .size = entry.size orelse 0,
                    .is_lfs = entry.lfs != null,
                    .sha256 = sha256,
                });
            }
        }
    }

    fn downloadFile(self: *Self, file: FileInfo) !void {
        const out_path = try std.fs.path.join(self.allocator, &.{
            self.config.output_dir,
            file.path,
        });
        defer self.allocator.free(out_path);

        // Create parent directories
        if (std.fs.path.dirname(out_path)) |dir| {
            std.fs.cwd().makePath(dir) catch |err| {
                if (err != error.PathAlreadyExists) return err;
            };
        }

        // Check if already downloaded
        if (std.fs.cwd().statFile(out_path)) |stat| {
            if (stat.size == file.size) {
                if (!self.config.quiet) {
                    const skip_name = truncatePath(std.fs.path.basename(file.path), NAME_WIDTH);
                    std.debug.print("{s}", .{skip_name});
                    for (0..NAME_WIDTH - @min(skip_name.len, NAME_WIDTH)) |_| std.debug.print(" ", .{});
                    std.debug.print(" \x1b[90m- skip\x1b[0m\n", .{});
                }
                return;
            }
        } else |_| {}

        // Check for partial download
        var start_offset: usize = 0;
        const partial_path = try std.fmt.allocPrint(self.allocator, "{s}.partial", .{out_path});
        defer self.allocator.free(partial_path);

        if (self.config.resume_downloads) {
            if (std.fs.cwd().statFile(partial_path)) |stat| {
                start_offset = stat.size;
                if (!self.config.quiet) {
                    std.debug.print("{s}: resuming from {s}\n", .{ file.path, formatSize(start_offset) });
                }
            } else |_| {}
        }

        // Build download URL
        const url = try std.fmt.allocPrint(
            self.allocator,
            "https://huggingface.co/{s}/resolve/{s}/{s}",
            .{ self.config.repo_id, self.config.revision, file.path },
        );
        defer self.allocator.free(url);

        // Use parallel download for large files (no resume support for parallel yet)
        const use_parallel = file.size >= self.config.parallel_threshold and
            start_offset == 0 and
            self.config.parallel_chunks > 1;

        if (use_parallel) {
            if (self.config.verbose) {
                std.debug.print("{s}: using {d} parallel chunks\n", .{ file.path, self.config.parallel_chunks });
            }
            try self.downloadParallel(url, out_path, file.size, self.config.parallel_chunks);
        } else {
            try self.downloadFromUrl(url, out_path, partial_path, start_offset, file.size);
        }

        // Verify SHA256 hash if available
        if (file.sha256) |expected_hash| {
            if (self.config.verbose) {
                std.debug.print("Verifying SHA256...\n", .{});
            }
            const actual_hash = try self.computeFileSha256(out_path);
            if (!std.mem.eql(u8, &actual_hash, expected_hash)) {
                std.debug.print("WARNING: SHA256 mismatch for {s}\n  expected: {s}\n  actual:   {s}\n", .{
                    file.path,
                    expected_hash,
                    &actual_hash,
                });
            } else if (self.config.verbose) {
                std.debug.print("SHA256 verified: {s}\n", .{&actual_hash});
            }
        }
    }

    fn computeFileSha256(self: *Self, path: []const u8) ![64]u8 {
        _ = self;
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        var buf: [64 * 1024]u8 = undefined;
        var read_buf: [4096]u8 = undefined;
        var file_reader = file.reader(&read_buf);
        const reader = &file_reader.interface;

        while (true) {
            const n = try reader.readSliceShort(&buf);
            if (n == 0) break;
            hasher.update(buf[0..n]);
        }

        const digest = hasher.finalResult();
        return std.fmt.bytesToHex(&digest, .lower);
    }

    fn downloadFromUrl(
        self: *Self,
        url: []const u8,
        final_path: []const u8,
        partial_path: []const u8,
        start_offset: usize,
        total_size: usize,
    ) !void {
        // Build extra headers
        var headers_buf: [512]u8 = undefined;
        var headers_list: [2]std.http.Header = undefined;
        var header_count: usize = 0;

        // Auth header
        if (self.config.token) |token| {
            const auth_value = std.fmt.bufPrint(headers_buf[0..256], "Bearer {s}", .{token}) catch return error.TokenTooLong;
            headers_list[header_count] = .{ .name = "Authorization", .value = auth_value };
            header_count += 1;
        }

        // Range header for resume
        if (start_offset > 0) {
            const range_value = std.fmt.bufPrint(headers_buf[256..], "bytes={d}-", .{start_offset}) catch return error.RangeHeaderTooLong;
            headers_list[header_count] = .{ .name = "Range", .value = range_value };
            header_count += 1;
        }

        // Use request API for streaming downloads
        const uri = try std.Uri.parse(url);
        var req = try self.http_client.request(.GET, uri, .{
            .extra_headers = headers_list[0..header_count],
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buf: [4096]u8 = undefined;
        var response = req.receiveHead(&redirect_buf) catch |err| {
            if (self.config.verbose) {
                std.debug.print("Receive error for {s}: {}\n", .{ url, err });
            }
            return error.HttpError;
        };

        if (response.head.status != .ok and response.head.status != .partial_content) {
            if (self.config.verbose) {
                std.debug.print("HTTP {d} for {s}\n", .{ @intFromEnum(response.head.status), url });
            }
            return error.HttpError;
        }

        // Get expected body size from response headers (cast u64 to usize for this platform)
        const expected_size: usize = if (response.head.content_length) |cl|
            @intCast(cl)
        else
            total_size - start_offset;

        // Open output file
        const use_partial = start_offset > 0 or total_size > 1024 * 1024; // Use .partial for files > 1MB
        const write_path = if (use_partial) partial_path else final_path;

        const file = try std.fs.cwd().createFile(write_path, .{
            .truncate = start_offset == 0,
        });
        defer file.close();

        if (start_offset > 0) {
            try file.seekTo(start_offset);
        }

        // Download with progress - use larger buffer for TLS
        var buf: [64 * 1024]u8 = undefined;
        var downloaded = start_offset;
        var body_read: usize = 0;
        var last_print = std.time.milliTimestamp();

        var transfer_buf: [32 * 1024]u8 = undefined;
        var reader = response.reader(&transfer_buf);

        // Read until we've got all expected bytes (avoids reader state bug after EOF)
        while (body_read < expected_size) {
            const remaining = expected_size - body_read;
            const to_read = @min(buf.len, remaining);
            const n = reader.readSliceShort(buf[0..to_read]) catch |err| {
                if (self.config.verbose) {
                    std.debug.print("Read error: {}\n", .{err});
                }
                return error.HttpError;
            };
            if (n == 0) break;

            try file.writeAll(buf[0..n]);
            body_read += n;
            downloaded = start_offset + body_read;

            // Update progress every 100ms
            const now = std.time.milliTimestamp();
            if (!self.config.quiet and (now - last_print > 100 or downloaded == total_size)) {
                self.printProgress(final_path, downloaded, total_size);
                last_print = now;
            }
        }

        // Rename partial to final
        if (use_partial) {
            try std.fs.cwd().rename(partial_path, final_path);
        }

        // Final progress - show completion
        if (!self.config.quiet) {
            var size_buf: [32]u8 = undefined;
            const name = truncatePath(std.fs.path.basename(final_path), NAME_WIDTH);
            std.debug.print("\r\x1b[K{s}", .{name});
            for (0..NAME_WIDTH - @min(name.len, NAME_WIDTH)) |_| std.debug.print(" ", .{});
            std.debug.print(" \x1b[32m✓\x1b[0m {s}\n", .{formatSizeLocal(total_size, &size_buf)});
        }
    }

    const NAME_WIDTH = 25; // Fixed width for filename column

    fn printProgress(self: *Self, path: []const u8, downloaded: usize, total: usize) void {
        _ = self;
        const basename = std.fs.path.basename(path);
        const pct: f64 = if (total > 0)
            @as(f64, @floatFromInt(downloaded)) / @as(f64, @floatFromInt(total)) * 100
        else
            0;

        // Format sizes into separate buffers
        var tot_buf: [32]u8 = undefined;
        const tot_str = formatSizeLocal(total, &tot_buf);

        // Format current with same width as total for stable display
        var dl_buf: [32]u8 = undefined;
        const dl_raw = formatSizeLocal(downloaded, &dl_buf);
        var dl_padded: [32]u8 = undefined;
        const pad_len = if (tot_str.len > dl_raw.len) tot_str.len - dl_raw.len else 0;
        for (0..pad_len) |i| dl_padded[i] = ' ';
        @memcpy(dl_padded[pad_len..][0..dl_raw.len], dl_raw);
        const dl_str = dl_padded[0 .. pad_len + dl_raw.len];

        // Progress bar with fancy chars (yellow filled, gray empty)
        const bar_width = 24;
        const filled = @as(usize, @intFromFloat(pct / 100.0 * @as(f64, @floatFromInt(bar_width))));

        // Pad filename to fixed width
        const name = truncatePath(basename, NAME_WIDTH);
        std.debug.print("\r\x1b[K{s}", .{name});
        // Pad with spaces if needed
        for (0..NAME_WIDTH - @min(name.len, NAME_WIDTH)) |_| std.debug.print(" ", .{});
        std.debug.print(" \x1b[33m", .{});

        for (0..filled) |_| std.debug.print("█", .{});
        std.debug.print("\x1b[90m", .{});
        for (0..bar_width - filled) |_| std.debug.print("░", .{});

        std.debug.print("\x1b[0m [ {s} / {s} ]", .{ dl_str, tot_str });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Parallel Download Support
    // ─────────────────────────────────────────────────────────────────────────

    const ChunkResult = struct {
        success: bool,
        bytes_written: usize,
        err_msg: ?[]const u8,
    };

    const ChunkContext = struct {
        allocator: Allocator,
        url: []const u8,
        file_path: []const u8,
        start_byte: usize,
        end_byte: usize,
        chunk_idx: usize,
        chunk_size: usize,
        token: ?[]const u8,
        result: ChunkResult,
        progress_bytes: *std.atomic.Value(usize), // Per-chunk progress
    };

    fn downloadParallel(
        self: *Self,
        url: []const u8,
        final_path: []const u8,
        total_size: usize,
        num_chunks: usize,
    ) !void {
        const chunk_size = total_size / num_chunks;
        var threads: [16]?std.Thread = .{null} ** 16;
        var contexts: [16]ChunkContext = undefined;
        // Per-thread progress counters
        var chunk_progress: [16]std.atomic.Value(usize) = undefined;
        for (0..16) |i| chunk_progress[i] = std.atomic.Value(usize).init(0);

        // Create file and pre-allocate
        const file = try std.fs.cwd().createFile(final_path, .{ .truncate = true });
        file.close();

        // Spawn download threads
        const actual_chunks = @min(num_chunks, 16);
        for (0..actual_chunks) |i| {
            const start = i * chunk_size;
            const end = if (i == actual_chunks - 1) total_size - 1 else (i + 1) * chunk_size - 1;
            const this_chunk_size = end - start + 1;

            contexts[i] = .{
                .allocator = self.allocator,
                .url = url,
                .file_path = final_path,
                .start_byte = start,
                .end_byte = end,
                .chunk_idx = i,
                .chunk_size = this_chunk_size,
                .token = self.config.token,
                .result = .{ .success = false, .bytes_written = 0, .err_msg = null },
                .progress_bytes = &chunk_progress[i],
            };

            threads[i] = try std.Thread.spawn(.{}, downloadChunkThread, .{&contexts[i]});
        }

        // Progress display loop
        const start_time = std.time.milliTimestamp();
        const bar_width: usize = 24;
        const segment_width = bar_width / actual_chunks;

        while (true) {
            // Calculate total and per-thread progress
            var total_downloaded: usize = 0;
            for (0..actual_chunks) |i| {
                total_downloaded += chunk_progress[i].load(.monotonic);
            }

            const pct = if (total_size > 0) (total_downloaded * 100) / total_size else 0;
            const elapsed_ms = std.time.milliTimestamp() - start_time;
            const speed = if (elapsed_ms > 0) @as(f64, @floatFromInt(total_downloaded)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0) else 0;

            // Format sizes - compact: "24.0 / 577.2 MB"
            const total_mb = @as(f64, @floatFromInt(total_size)) / (1024.0 * 1024.0);
            const current_mb = @as(f64, @floatFromInt(total_downloaded)) / (1024.0 * 1024.0);
            const speed_mbs = speed / (1024.0 * 1024.0);

            // Build half-height progress bar: top=threads (fg), bottom=total (bg)
            // Using ▀ with foreground=top color, background=bottom color
            const name = truncatePath(std.fs.path.basename(final_path), NAME_WIDTH);
            std.debug.print("\r\x1b[K{s}", .{name});
            for (0..NAME_WIDTH - @min(name.len, NAME_WIDTH)) |_| std.debug.print(" ", .{});
            std.debug.print(" ", .{});

            const total_filled = (pct * bar_width) / 100;

            // Foreground colors for threads (bright)
            const fg_colors = [_][]const u8{ "\x1b[93m", "\x1b[96m", "\x1b[95m", "\x1b[92m" }; // bright yellow, cyan, magenta, green
            const fg_empty = "\x1b[90m"; // gray
            // Background colors
            const bg_white = "\x1b[47m"; // white background for total progress
            const bg_dark = "\x1b[100m"; // dark gray background for empty

            for (0..actual_chunks) |chunk_i| {
                const chunk_bytes = chunk_progress[chunk_i].load(.monotonic);
                const chunk_total = contexts[chunk_i].chunk_size;
                const chunk_pct = if (chunk_total > 0) (chunk_bytes * 100) / chunk_total else 0;
                const thread_filled = (chunk_pct * segment_width) / 100;
                const segment_start = chunk_i * segment_width;

                for (0..segment_width) |pos| {
                    const global_pos = segment_start + pos;
                    const has_thread = pos < thread_filled;
                    const has_total = global_pos < total_filled;

                    // Set foreground (top half) based on thread progress
                    const fg = if (has_thread) fg_colors[chunk_i % 4] else fg_empty;
                    // Set background (bottom half) based on total progress
                    const bg = if (has_total) bg_white else bg_dark;

                    std.debug.print("{s}{s}▀", .{ fg, bg });
                }
            }
            std.debug.print("\x1b[0m", .{}); // reset colors

            std.debug.print("\x1b[0m {d:>5.1}/{d:.1} MB {d:>2}% {d:.1} MB/s", .{ current_mb, total_mb, pct, speed_mbs });

            if (total_downloaded >= total_size) break;

            // Check if all threads done
            var all_done = true;
            for (0..actual_chunks) |i| {
                if (threads[i] != null) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;

            std.Thread.sleep(100 * std.time.ns_per_ms);
        }

        // Wait for all threads
        var total_written: usize = 0;
        var had_error = false;
        for (0..actual_chunks) |i| {
            if (threads[i]) |t| {
                t.join();
                threads[i] = null;
                total_written += contexts[i].result.bytes_written;
                if (!contexts[i].result.success) had_error = true;
            }
        }

        var size_buf: [32]u8 = undefined;
        const final_name = truncatePath(std.fs.path.basename(final_path), NAME_WIDTH);
        std.debug.print("\r\x1b[K{s}", .{final_name});
        for (0..NAME_WIDTH - @min(final_name.len, NAME_WIDTH)) |_| std.debug.print(" ", .{});
        std.debug.print(" \x1b[32m✓\x1b[0m {s}\n", .{formatSizeLocal(total_written, &size_buf)});

        if (had_error) {
            return error.ChunkDownloadFailed;
        }
    }

    fn downloadChunkThread(ctx: *ChunkContext) void {
        downloadChunk(ctx) catch |err| {
            ctx.result.success = false;
            ctx.result.err_msg = @errorName(err);
        };
    }

    fn downloadChunk(ctx: *ChunkContext) !void {
        // Each thread needs its own HTTP client
        var client = std.http.Client{
            .allocator = ctx.allocator,
            .tls_buffer_size = 32 * 1024,
            .read_buffer_size = 32 * 1024,
            .write_buffer_size = 8 * 1024,
        };
        defer client.deinit();

        // Build headers with Range
        var headers_buf: [512]u8 = undefined;
        var headers_list: [2]std.http.Header = undefined;
        var header_count: usize = 0;

        // Auth header
        if (ctx.token) |token| {
            const auth_value = std.fmt.bufPrint(headers_buf[0..256], "Bearer {s}", .{token}) catch return error.TokenTooLong;
            headers_list[header_count] = .{ .name = "Authorization", .value = auth_value };
            header_count += 1;
        }

        // Range header
        const range_value = std.fmt.bufPrint(headers_buf[256..], "bytes={d}-{d}", .{ ctx.start_byte, ctx.end_byte }) catch return error.RangeHeaderTooLong;
        headers_list[header_count] = .{ .name = "Range", .value = range_value };
        header_count += 1;

        // Make request
        const uri = try std.Uri.parse(ctx.url);
        var req = try client.request(.GET, uri, .{
            .extra_headers = headers_list[0..header_count],
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buf: [4096]u8 = undefined;
        var response = try req.receiveHead(&redirect_buf);

        if (response.head.status != .ok and response.head.status != .partial_content) {
            return error.HttpError;
        }

        // Open file and seek to our chunk position
        const file = try std.fs.cwd().openFile(ctx.file_path, .{ .mode = .write_only });
        defer file.close();
        try file.seekTo(ctx.start_byte);

        // Download our chunk
        var buf: [64 * 1024]u8 = undefined;
        var transfer_buf: [32 * 1024]u8 = undefined;
        var reader = response.reader(&transfer_buf);

        const expected_size = ctx.end_byte - ctx.start_byte + 1;
        var bytes_read: usize = 0;

        while (bytes_read < expected_size) {
            const remaining = expected_size - bytes_read;
            const to_read = @min(buf.len, remaining);
            const n = reader.readSliceShort(buf[0..to_read]) catch |err| {
                ctx.result.err_msg = @errorName(err);
                return err;
            };
            if (n == 0) break;

            try file.writeAll(buf[0..n]);
            bytes_read += n;
            ctx.result.bytes_written = bytes_read;

            // Update shared progress counter
            _ = ctx.progress_bytes.fetchAdd(n, .monotonic);
        }

        ctx.result.success = true;
    }

    fn httpGet(self: *Self, url: []const u8) ![]u8 {
        // Build extra headers for auth
        var auth_header_buf: [256]u8 = undefined;
        const extra_headers: []const std.http.Header = if (self.config.token) |token| blk: {
            const auth_value = std.fmt.bufPrint(&auth_header_buf, "Bearer {s}", .{token}) catch return error.TokenTooLong;
            break :blk &.{.{ .name = "Authorization", .value = auth_value }};
        } else &.{};

        // Use request API for proper body reading
        const uri = try std.Uri.parse(url);
        var req = try self.http_client.request(.GET, uri, .{
            .extra_headers = extra_headers,
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buf: [4096]u8 = undefined;
        var response = req.receiveHead(&redirect_buf) catch |err| {
            if (self.config.verbose) {
                std.debug.print("Receive error for {s}: {}\n", .{ url, err });
            }
            return error.HttpError;
        };

        if (response.head.status != .ok) {
            if (self.config.verbose) {
                std.debug.print("HTTP {d} for {s}\n", .{ @intFromEnum(response.head.status), url });
            }
            return error.HttpError;
        }

        // Read body using allocRemaining with larger buffer for TLS
        var transfer_buf: [32 * 1024]u8 = undefined;
        var reader = response.reader(&transfer_buf);
        const body = reader.allocRemaining(self.allocator, std.Io.Limit.limited(10 * 1024 * 1024)) catch |err| {
            if (self.config.verbose) {
                std.debug.print("Read error: {}\n", .{err});
            }
            return error.HttpError;
        };

        return body;
    }

    fn shouldDownload(self: *Self, file: FileInfo) bool {
        // Check excludes first
        for (self.config.exclude_patterns) |pattern| {
            if (matchGlob(file.path, pattern)) return false;
        }

        // If no includes specified, download all
        if (self.config.include_patterns.len == 0) return true;

        // Check includes
        for (self.config.include_patterns) |pattern| {
            if (matchGlob(file.path, pattern)) return true;
        }

        return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// File Info
// ─────────────────────────────────────────────────────────────────────────────

const FileInfo = struct {
    path: []const u8,
    size: usize,
    is_lfs: bool,
    sha256: ?[]const u8, // SHA256 hash for LFS files (hex string)

    fn deinit(self: FileInfo, allocator: Allocator) void {
        allocator.free(self.path);
        if (self.sha256) |h| allocator.free(h);
    }
};

const JsonFileEntry = struct {
    type: []const u8,
    path: []const u8,
    size: ?usize = null,
    lfs: ?JsonLfsInfo = null,
};

const JsonLfsInfo = struct {
    oid: ?[]const u8 = null, // SHA256 hash (called 'oid' in HF API)
    size: ?usize = null,
};

// ─────────────────────────────────────────────────────────────────────────────
// Argument Parsing
// ─────────────────────────────────────────────────────────────────────────────

const Args = struct {
    allocator: Allocator,
    repo_id: ?[]const u8 = null,
    output_dir: ?[]const u8 = null,
    revision: []const u8 = "main",
    token: ?[]const u8 = null,
    include_patterns: std.ArrayList([]const u8),
    exclude_patterns: std.ArrayList([]const u8),
    resume_downloads: bool = false,
    dry_run: bool = false,
    quiet: bool = false,
    verbose: bool = false,
    help: bool = false,

    fn parse(allocator: Allocator) !Args {
        var self = Args{
            .allocator = allocator,
            .include_patterns = .{},
            .exclude_patterns = .{},
        };
        errdefer self.deinit();

        var args_iter = try std.process.argsWithAllocator(allocator);
        defer args_iter.deinit();

        _ = args_iter.skip(); // program name

        while (args_iter.next()) |arg| {
            if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
                self.help = true;
            } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
                self.output_dir = args_iter.next();
            } else if (std.mem.eql(u8, arg, "-r") or std.mem.eql(u8, arg, "--revision")) {
                self.revision = args_iter.next() orelse "main";
            } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--token")) {
                self.token = args_iter.next();
            } else if (std.mem.eql(u8, arg, "--include")) {
                if (args_iter.next()) |pattern| {
                    try self.include_patterns.append(allocator, pattern);
                }
            } else if (std.mem.eql(u8, arg, "--exclude")) {
                if (args_iter.next()) |pattern| {
                    try self.exclude_patterns.append(allocator, pattern);
                }
            } else if (std.mem.eql(u8, arg, "--resume")) {
                self.resume_downloads = true;
            } else if (std.mem.eql(u8, arg, "--dry-run")) {
                self.dry_run = true;
            } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
                self.quiet = true;
            } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
                self.verbose = true;
            } else if (!std.mem.startsWith(u8, arg, "-")) {
                self.repo_id = arg;
            }
        }

        return self;
    }

    fn deinit(self: *Args) void {
        self.include_patterns.deinit(self.allocator);
        self.exclude_patterns.deinit(self.allocator);
    }
};

fn printUsage() void {
    const usage =
        \\hfd - HuggingFace Downloader
        \\
        \\Download models from HuggingFace Hub.
        \\Watch out for facehuggers.
        \\
        \\USAGE:
        \\    hfd <repo_id> [options]
        \\
        \\ARGUMENTS:
        \\    <repo_id>             Repository ID (e.g., black-forest-labs/FLUX.2-klein)
        \\
        \\OPTIONS:
        \\    -o, --output DIR      Output directory (default: repo name)
        \\    -r, --revision REV    Git revision/branch (default: main)
        \\    -t, --token TOKEN     HuggingFace token (or set HF_TOKEN env var)
        \\    --include PATTERN     Only download files matching glob pattern
        \\    --exclude PATTERN     Skip files matching glob pattern
        \\    --resume              Resume interrupted downloads
        \\    --dry-run             Show files without downloading
        \\    -q, --quiet           Suppress progress output
        \\    -v, --verbose         Show detailed output
        \\    -h, --help            Show this help
        \\
        \\EXAMPLES:
        \\    hfd black-forest-labs/FLUX.2-klein
        \\    hfd black-forest-labs/FLUX.2-klein -o ./my-model
        \\    hfd black-forest-labs/FLUX.2-klein --include "*.safetensors"
        \\    hfd black-forest-labs/FLUX.2-klein --exclude "*.bin" --exclude "*.onnx"
        \\
        \\ENVIRONMENT:
        \\    HF_TOKEN              HuggingFace API token for gated models
        \\
    ;
    std.debug.print("{s}", .{usage});
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Simple glob matching (* and ?)
fn matchGlob(text: []const u8, pattern: []const u8) bool {
    var ti: usize = 0;
    var pi: usize = 0;
    var star_ti: ?usize = null;
    var star_pi: ?usize = null;

    while (ti < text.len) {
        if (pi < pattern.len and (pattern[pi] == '?' or pattern[pi] == text[ti])) {
            ti += 1;
            pi += 1;
        } else if (pi < pattern.len and pattern[pi] == '*') {
            star_ti = ti;
            star_pi = pi;
            pi += 1;
        } else if (star_pi) |sp| {
            pi = sp + 1;
            star_ti.? += 1;
            ti = star_ti.?;
        } else {
            return false;
        }
    }

    while (pi < pattern.len and pattern[pi] == '*') {
        pi += 1;
    }

    return pi == pattern.len;
}

fn formatSize(bytes: usize) []const u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
    var size: f64 = @floatFromInt(bytes);
    var unit_idx: usize = 0;

    while (size >= 1024 and unit_idx < units.len - 1) {
        size /= 1024;
        unit_idx += 1;
    }

    // Return static buffer (not ideal but works for display)
    const Static = struct {
        var buf: [32]u8 = undefined;
    };

    return std.fmt.bufPrint(&Static.buf, "{d:.1} {s}", .{ size, units[unit_idx] }) catch return "???";
}

fn truncatePath(path: []const u8, max_len: usize) []const u8 {
    if (path.len <= max_len) return path;
    return path[0..max_len];
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "glob matching" {
    try std.testing.expect(matchGlob("model.safetensors", "*.safetensors"));
    try std.testing.expect(matchGlob("vae/model.safetensors", "*.safetensors"));
    try std.testing.expect(!matchGlob("model.bin", "*.safetensors"));
    try std.testing.expect(matchGlob("config.json", "config.*"));
    try std.testing.expect(matchGlob("anything", "*"));
    try std.testing.expect(matchGlob("test", "t?st"));
    try std.testing.expect(!matchGlob("test", "t?t"));
}

test "format size" {
    try std.testing.expectEqualStrings("0.0 B", formatSize(0));
    try std.testing.expectEqualStrings("1.0 KB", formatSize(1024));
    try std.testing.expectEqualStrings("1.0 MB", formatSize(1024 * 1024));
}
