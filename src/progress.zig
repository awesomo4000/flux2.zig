//! progress.zig - Beautiful terminal progress bars 📊
//!
//! Zig-compiler-style progress with bars, ETA, and transfer speeds.
//!
//! ```
//! [2/4] transformer/model.safetensors (4 threads)
//!       ████████████░░░░░░░░░░░░░░░░░░░░ 2.1/7.2 GB @ 52.1 MB/s eta 1:42
//! ```

const std = @import("std");

/// Progress bar configuration
pub const Config = struct {
    /// Total width of the progress bar (excluding decorations)
    bar_width: usize = 32,
    /// Filled character
    filled: u8 = 0xE2, // UTF-8 block: █ (we'll use ASCII # for safety)
    /// Empty character
    empty: u8 = 0xE2,
    /// Show speed
    show_speed: bool = true,
    /// Show ETA
    show_eta: bool = true,
    /// Use colors (ANSI)
    use_colors: bool = true,
    /// Refresh rate in ms
    refresh_ms: u64 = 100,
};

/// Represents a single download/task being tracked
pub const Task = struct {
    name: []const u8,
    current: usize,
    total: usize,
    started_at: i64,
    last_update: i64,
    bytes_at_last_update: usize,
    speed_samples: [8]f64, // Rolling average
    sample_idx: usize,
    thread_count: usize,
    status: Status,

    pub const Status = enum {
        pending,
        active,
        complete,
        failed,
    };

    pub fn init(name: []const u8, total: usize) Task {
        const now = std.time.milliTimestamp();
        return .{
            .name = name,
            .current = 0,
            .total = total,
            .started_at = now,
            .last_update = now,
            .bytes_at_last_update = 0,
            .speed_samples = .{0} ** 8,
            .sample_idx = 0,
            .thread_count = 1,
            .status = .pending,
        };
    }

    pub fn update(self: *Task, current: usize) void {
        const now = std.time.milliTimestamp();
        const elapsed = now - self.last_update;

        if (elapsed >= 100) { // Update speed every 100ms
            const bytes_delta = current - self.bytes_at_last_update;
            const speed = @as(f64, @floatFromInt(bytes_delta)) / (@as(f64, @floatFromInt(elapsed)) / 1000.0);

            self.speed_samples[self.sample_idx % 8] = speed;
            self.sample_idx += 1;

            self.bytes_at_last_update = current;
            self.last_update = now;
        }

        self.current = current;
        if (self.status == .pending) self.status = .active;
        if (current >= self.total) self.status = .complete;
    }

    pub fn getSpeed(self: *const Task) f64 {
        var sum: f64 = 0;
        var count: usize = 0;
        for (self.speed_samples) |s| {
            if (s > 0) {
                sum += s;
                count += 1;
            }
        }
        return if (count > 0) sum / @as(f64, @floatFromInt(count)) else 0;
    }

    pub fn getEtaSeconds(self: *const Task) ?u64 {
        const speed = self.getSpeed();
        if (speed <= 0) return null;
        const remaining = self.total - self.current;
        return @intFromFloat(@as(f64, @floatFromInt(remaining)) / speed);
    }

    pub fn getProgress(self: *const Task) f64 {
        if (self.total == 0) return 0;
        return @as(f64, @floatFromInt(self.current)) / @as(f64, @floatFromInt(self.total));
    }
};

/// Multi-task progress display
pub const Display = struct {
    allocator: std.mem.Allocator,
    tasks: std.ArrayList(Task),
    config: Config,
    current_idx: usize,
    total_count: usize,
    mutex: std.Thread.Mutex,
    last_render: i64,
    lines_rendered: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, total_count: usize, config: Config) Self {
        return .{
            .allocator = allocator,
            .tasks = std.ArrayList(Task).init(allocator),
            .config = config,
            .current_idx = 0,
            .total_count = total_count,
            .mutex = .{},
            .last_render = 0,
            .lines_rendered = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.tasks.deinit();
    }

    pub fn addTask(self: *Self, name: []const u8, total: usize) !*Task {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.tasks.append(Task.init(name, total));
        self.current_idx += 1;
        return &self.tasks.items[self.tasks.items.len - 1];
    }

    pub fn render(self: *Self, writer: anytype) !void {
        const now = std.time.milliTimestamp();
        if (now - self.last_render < self.config.refresh_ms) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        // Move cursor up to overwrite previous render
        if (self.lines_rendered > 0) {
            for (0..self.lines_rendered) |_| {
                try writer.print("\x1b[A\x1b[2K", .{}); // Up + clear line
            }
        }

        self.lines_rendered = 0;

        for (self.tasks.items, 0..) |*task, idx| {
            try self.renderTask(writer, task, idx + 1);
            self.lines_rendered += 2; // Task name + progress bar
        }

        self.last_render = now;
    }

    fn renderTask(self: *Self, writer: anytype, task: *const Task, idx: usize) !void {
        const c = self.config;

        // Line 1: [n/total] filename (threads)
        const status_icon: []const u8 = switch (task.status) {
            .pending => "○",
            .active => "◐",
            .complete => "●",
            .failed => "✗",
        };

        const thread_info = if (task.thread_count > 1)
            try std.fmt.allocPrint(self.allocator, " ({d} threads)", .{task.thread_count})
        else
            "";
        defer if (thread_info.len > 0) self.allocator.free(thread_info);

        // Truncate filename if needed
        const max_name_len = 40;
        const display_name = if (task.name.len > max_name_len)
            task.name[task.name.len - max_name_len ..]
        else
            task.name;

        if (c.use_colors) {
            const color: []const u8 = switch (task.status) {
                .pending => "\x1b[90m", // Gray
                .active => "\x1b[36m", // Cyan
                .complete => "\x1b[32m", // Green
                .failed => "\x1b[31m", // Red
            };
            try writer.print("{s}[{d}/{d}]\x1b[0m {s} {s}{s}\n", .{
                color,
                idx,
                self.total_count,
                status_icon,
                display_name,
                thread_info,
            });
        } else {
            try writer.print("[{d}/{d}] {s} {s}{s}\n", .{
                idx,
                self.total_count,
                status_icon,
                display_name,
                thread_info,
            });
        }

        // Line 2: Progress bar with stats
        try writer.print("      ", .{}); // Indent

        // Draw the bar
        const progress = task.getProgress();
        const filled_count = @as(usize, @intFromFloat(progress * @as(f64, @floatFromInt(c.bar_width))));

        if (c.use_colors) {
            try writer.print("\x1b[32m", .{}); // Green for filled
        }
        for (0..filled_count) |_| {
            try writer.print("█", .{});
        }
        if (c.use_colors) {
            try writer.print("\x1b[90m", .{}); // Gray for empty
        }
        for (0..c.bar_width - filled_count) |_| {
            try writer.print("░", .{});
        }
        if (c.use_colors) {
            try writer.print("\x1b[0m", .{}); // Reset
        }

        // Stats: size
        try writer.print(" {s}/{s}", .{
            formatSize(task.current),
            formatSize(task.total),
        });

        // Speed
        if (c.show_speed and task.status == .active) {
            const speed = task.getSpeed();
            if (speed > 0) {
                try writer.print(" @ {s}/s", .{formatSize(@intFromFloat(speed))});
            }
        }

        // ETA
        if (c.show_eta and task.status == .active) {
            if (task.getEtaSeconds()) |eta| {
                try writer.print(" eta {s}", .{formatDuration(eta)});
            }
        }

        // Done marker
        if (task.status == .complete) {
            if (c.use_colors) {
                try writer.print(" \x1b[32m✓\x1b[0m", .{});
            } else {
                try writer.print(" done", .{});
            }
        }

        try writer.print("\n", .{});
    }

    /// Render final summary
    pub fn finish(self: *Self, writer: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var total_bytes: usize = 0;
        var completed: usize = 0;
        var failed: usize = 0;
        var min_start: i64 = std.math.maxInt(i64);

        for (self.tasks.items) |task| {
            total_bytes += task.current;
            if (task.status == .complete) completed += 1;
            if (task.status == .failed) failed += 1;
            if (task.started_at < min_start) min_start = task.started_at;
        }

        const elapsed_ms = std.time.milliTimestamp() - min_start;
        const elapsed_s: f64 = @as(f64, @floatFromInt(elapsed_ms)) / 1000.0;
        const avg_speed = if (elapsed_s > 0)
            @as(f64, @floatFromInt(total_bytes)) / elapsed_s
        else
            0;

        try writer.print("\n", .{});
        if (self.config.use_colors) {
            try writer.print("\x1b[32m✓\x1b[0m ", .{});
        }
        try writer.print("Downloaded {s} in {s}", .{
            formatSize(total_bytes),
            formatDuration(@intFromFloat(elapsed_s)),
        });
        if (avg_speed > 0) {
            try writer.print(" (avg {s}/s)", .{formatSize(@intFromFloat(avg_speed))});
        }
        try writer.print("\n", .{});

        if (failed > 0) {
            if (self.config.use_colors) {
                try writer.print("\x1b[31m✗\x1b[0m {d} failed\n", .{failed});
            } else {
                try writer.print("{d} failed\n", .{failed});
            }
        }
    }
};

/// Simple single-line progress bar (for individual files)
pub const Bar = struct {
    current: usize,
    total: usize,
    started_at: i64,
    last_bytes: usize,
    last_time: i64,
    speed: f64,
    width: usize,
    name: []const u8,

    const Self = @This();

    pub fn init(name: []const u8, total: usize) Self {
        const now = std.time.milliTimestamp();
        return .{
            .current = 0,
            .total = total,
            .started_at = now,
            .last_bytes = 0,
            .last_time = now,
            .speed = 0,
            .width = 30,
            .name = name,
        };
    }

    pub fn update(self: *Self, current: usize) void {
        const now = std.time.milliTimestamp();
        const dt = now - self.last_time;

        if (dt >= 100) {
            const db = current - self.last_bytes;
            const new_speed = @as(f64, @floatFromInt(db)) / (@as(f64, @floatFromInt(dt)) / 1000.0);
            // Exponential moving average
            self.speed = self.speed * 0.7 + new_speed * 0.3;
            self.last_bytes = current;
            self.last_time = now;
        }

        self.current = current;
    }

    pub fn render(self: *const Self, writer: anytype) !void {
        const progress = if (self.total > 0)
            @as(f64, @floatFromInt(self.current)) / @as(f64, @floatFromInt(self.total))
        else
            0;

        const filled = @as(usize, @intFromFloat(progress * @as(f64, @floatFromInt(self.width))));

        // Truncate name
        const max_name = 25;
        const display_name = if (self.name.len > max_name)
            self.name[self.name.len - max_name ..]
        else
            self.name;

        try writer.print("\r\x1b[K{s}: ", .{display_name}); // Clear line + name

        // Bar
        try writer.print("\x1b[32m", .{});
        for (0..filled) |_| try writer.print("█", .{});
        try writer.print("\x1b[90m", .{});
        for (0..self.width - filled) |_| try writer.print("░", .{});
        try writer.print("\x1b[0m", .{});

        // Stats
        try writer.print(" {s}/{s}", .{ formatSize(self.current), formatSize(self.total) });

        if (self.speed > 0) {
            try writer.print(" @ {s}/s", .{formatSize(@intFromFloat(self.speed))});

            // ETA
            if (self.current < self.total) {
                const remaining = self.total - self.current;
                const eta_s = @as(f64, @floatFromInt(remaining)) / self.speed;
                if (eta_s < 86400) { // Less than a day
                    try writer.print(" eta {s}", .{formatDuration(@intFromFloat(eta_s))});
                }
            }
        }

        if (self.current >= self.total) {
            try writer.print(" \x1b[32m✓\x1b[0m", .{});
        }
    }

    pub fn finish(self: *const Self, writer: anytype) !void {
        try self.render(writer);
        try writer.print("\n", .{});
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

var size_buf: [32]u8 = undefined;

pub fn formatSize(bytes: usize) []const u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
    var size: f64 = @floatFromInt(bytes);
    var unit_idx: usize = 0;

    while (size >= 1024 and unit_idx < units.len - 1) {
        size /= 1024;
        unit_idx += 1;
    }

    if (unit_idx == 0) {
        return std.fmt.bufPrint(&size_buf, "{d} {s}", .{ bytes, units[0] }) catch "?";
    } else {
        return std.fmt.bufPrint(&size_buf, "{d:.1} {s}", .{ size, units[unit_idx] }) catch "?";
    }
}

var duration_buf: [32]u8 = undefined;

pub fn formatDuration(seconds: u64) []const u8 {
    if (seconds < 60) {
        return std.fmt.bufPrint(&duration_buf, "{d}s", .{seconds}) catch "?";
    } else if (seconds < 3600) {
        const m = seconds / 60;
        const s = seconds % 60;
        return std.fmt.bufPrint(&duration_buf, "{d}:{d:0>2}", .{ m, s }) catch "?";
    } else {
        const h = seconds / 3600;
        const m = (seconds % 3600) / 60;
        const s = seconds % 60;
        return std.fmt.bufPrint(&duration_buf, "{d}:{d:0>2}:{d:0>2}", .{ h, m, s }) catch "?";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "formatSize" {
    try std.testing.expectEqualStrings("0 B", formatSize(0));
    try std.testing.expectEqualStrings("512 B", formatSize(512));
    try std.testing.expectEqualStrings("1.0 KB", formatSize(1024));
    try std.testing.expectEqualStrings("1.5 MB", formatSize(1024 * 1024 + 512 * 1024));
    try std.testing.expectEqualStrings("2.0 GB", formatSize(2 * 1024 * 1024 * 1024));
}

test "formatDuration" {
    try std.testing.expectEqualStrings("0s", formatDuration(0));
    try std.testing.expectEqualStrings("45s", formatDuration(45));
    try std.testing.expectEqualStrings("1:30", formatDuration(90));
    try std.testing.expectEqualStrings("1:00:00", formatDuration(3600));
    try std.testing.expectEqualStrings("2:30:45", formatDuration(2 * 3600 + 30 * 60 + 45));
}
