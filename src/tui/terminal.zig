//! Low-level terminal control using ANSI escape sequences.
//!
//! Provides raw mode, cursor control, colors, and input handling
//! without external dependencies.

const std = @import("std");
const posix = std.posix;

/// Terminal instance for TUI rendering
pub const Terminal = struct {
    tty: std.fs.File,
    original_termios: posix.termios,
    width: u16,
    height: u16,

    pub fn init(allocator: std.mem.Allocator) !Terminal {
        _ = allocator; // Reserved for future use
        const tty = std.fs.cwd().openFile("/dev/tty", .{ .mode = .read_write }) catch |err| {
            // Fall back to stdout if /dev/tty not available
            if (err == error.FileNotFound) {
                return error.NoTTY;
            }
            return err;
        };
        errdefer tty.close();

        const original = try posix.tcgetattr(tty.handle);

        // Get terminal size
        const size = getTerminalSize(tty.handle);

        return .{
            .tty = tty,
            .original_termios = original,
            .width = size.cols,
            .height = size.rows,
        };
    }

    pub fn deinit(self: *Terminal) void {
        self.leaveRawMode() catch {};
        self.tty.close();
    }

    /// Enter raw mode for character-by-character input
    pub fn enterRawMode(self: *Terminal) !void {
        var raw = self.original_termios;

        // Input flags: disable break, CR to NL, parity check, strip, XON/XOFF
        raw.iflag.BRKINT = false;
        raw.iflag.ICRNL = false;
        raw.iflag.INPCK = false;
        raw.iflag.ISTRIP = false;
        raw.iflag.IXON = false;

        // Output flags: disable post-processing
        raw.oflag.OPOST = false;

        // Control flags: set 8-bit chars
        raw.cflag.CSIZE = .CS8;

        // Local flags: disable echo, canonical mode, signals, extended input
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.ISIG = false;
        raw.lflag.IEXTEN = false;

        // Read returns immediately with available chars (non-blocking)
        raw.cc[@intFromEnum(posix.V.MIN)] = 0;
        raw.cc[@intFromEnum(posix.V.TIME)] = 0;

        try posix.tcsetattr(self.tty.handle, .FLUSH, raw);

        // Enter alternate screen buffer and hide cursor
        try self.writeAll(CSI ++ "?1049h" ++ CSI ++ "?25l");
    }

    /// Restore original terminal settings
    pub fn leaveRawMode(self: *Terminal) !void {
        // Show cursor and leave alternate screen
        try self.writeAll(CSI ++ "?25h" ++ CSI ++ "?1049l");
        try posix.tcsetattr(self.tty.handle, .FLUSH, self.original_termios);
    }

    /// Clear screen
    pub fn clear(self: *Terminal) !void {
        try self.writeAll(CSI ++ "2J" ++ CSI ++ "H");
    }

    /// Move cursor to position (1-indexed)
    pub fn moveTo(self: *Terminal, row: u16, col: u16) !void {
        var buf: [32]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, CSI ++ "{d};{d}H", .{ row, col }) catch return;
        try self.writeAll(seq);
    }

    /// Write text at current position
    pub fn write(self: *Terminal, text: []const u8) !void {
        try self.writeAll(text);
    }

    /// Write formatted text
    pub fn print(self: *Terminal, comptime fmt: []const u8, args: anytype) !void {
        var buf: [4096]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
        try self.writeAll(text);
    }

    /// Set foreground color
    pub fn setFg(self: *Terminal, color: Color) !void {
        var buf: [16]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, CSI ++ "{d}m", .{color.toFgCode()}) catch return;
        try self.writeAll(seq);
    }

    /// Set background color
    pub fn setBg(self: *Terminal, color: Color) !void {
        var buf: [16]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, CSI ++ "{d}m", .{color.toBgCode()}) catch return;
        try self.writeAll(seq);
    }

    /// Reset colors and attributes
    pub fn reset(self: *Terminal) !void {
        try self.writeAll(CSI ++ "0m");
    }

    /// Set bold attribute
    pub fn setBold(self: *Terminal) !void {
        try self.writeAll(CSI ++ "1m");
    }

    /// Set dim attribute
    pub fn setDim(self: *Terminal) !void {
        try self.writeAll(CSI ++ "2m");
    }

    /// Poll for keyboard input (non-blocking)
    pub fn pollInput(self: *Terminal) ?Key {
        var buf: [8]u8 = undefined;
        const n = self.tty.read(&buf) catch return null;
        if (n == 0) return null;

        return parseKey(buf[0..n]);
    }

    /// Refresh terminal size
    pub fn refreshSize(self: *Terminal) void {
        const size = getTerminalSize(self.tty.handle);
        self.width = size.cols;
        self.height = size.rows;
    }

    fn writeAll(self: *Terminal, data: []const u8) !void {
        _ = try self.tty.write(data);
    }
};

/// ANSI Control Sequence Introducer
const CSI = "\x1b[";

/// Terminal colors
pub const Color = enum(u8) {
    black = 0,
    red = 1,
    green = 2,
    yellow = 3,
    blue = 4,
    magenta = 5,
    cyan = 6,
    white = 7,
    default = 9,
    // Bright variants
    bright_black = 60,
    bright_red = 61,
    bright_green = 62,
    bright_yellow = 63,
    bright_blue = 64,
    bright_magenta = 65,
    bright_cyan = 66,
    bright_white = 67,

    pub fn toFgCode(self: Color) u8 {
        const base: u8 = @intFromEnum(self);
        if (base >= 60) return base + 30; // 90-97 for bright
        return base + 30; // 30-39 for normal
    }

    pub fn toBgCode(self: Color) u8 {
        const base: u8 = @intFromEnum(self);
        if (base >= 60) return base + 40; // 100-107 for bright
        return base + 40; // 40-49 for normal
    }
};

/// Keyboard keys
pub const Key = union(enum) {
    char: u8,
    enter,
    escape,
    backspace,
    tab,
    up,
    down,
    left,
    right,
    home,
    end,
    page_up,
    page_down,
    delete,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    ctrl_c,
    ctrl_d,
    ctrl_q,
    unknown,
};

fn parseKey(buf: []const u8) Key {
    if (buf.len == 0) return .unknown;

    // Single character
    if (buf.len == 1) {
        return switch (buf[0]) {
            0x03 => .ctrl_c,
            0x04 => .ctrl_d,
            0x11 => .ctrl_q,
            0x09 => .tab,
            0x0D => .enter,
            0x1B => .escape,
            0x7F => .backspace,
            else => if (buf[0] >= 0x20 and buf[0] < 0x7F) .{ .char = buf[0] } else .unknown,
        };
    }

    // Escape sequences
    if (buf[0] == 0x1B) {
        if (buf.len >= 3 and buf[1] == '[') {
            return switch (buf[2]) {
                'A' => .up,
                'B' => .down,
                'C' => .right,
                'D' => .left,
                'H' => .home,
                'F' => .end,
                '1' => if (buf.len >= 4) switch (buf[3]) {
                    '~' => .home,
                    '5' => .f5,
                    '7' => .f6,
                    '8' => .f7,
                    '9' => .f8,
                    else => .unknown,
                } else .unknown,
                '2' => if (buf.len >= 4 and buf[3] == '~') .unknown else .unknown, // Insert
                '3' => if (buf.len >= 4 and buf[3] == '~') .delete else .unknown,
                '4' => if (buf.len >= 4 and buf[3] == '~') .end else .unknown,
                '5' => if (buf.len >= 4 and buf[3] == '~') .page_up else .unknown,
                '6' => if (buf.len >= 4 and buf[3] == '~') .page_down else .unknown,
                else => .unknown,
            };
        }
        // Alt + key
        if (buf.len == 2) return .unknown;
    }

    return .unknown;
}

const TermSize = struct {
    rows: u16,
    cols: u16,
};

fn getTerminalSize(fd: posix.fd_t) TermSize {
    var ws: posix.winsize = undefined;
    const result = posix.system.ioctl(fd, posix.T.IOCGWINSZ, @intFromPtr(&ws));
    if (result == 0) {
        return .{
            .rows = ws.row,
            .cols = ws.col,
        };
    }
    // Default fallback
    return .{ .rows = 24, .cols = 80 };
}

// ============================================================================
// Tests
// ============================================================================

test "color codes" {
    try std.testing.expectEqual(@as(u8, 31), Color.red.toFgCode());
    try std.testing.expectEqual(@as(u8, 42), Color.green.toBgCode());
    try std.testing.expectEqual(@as(u8, 91), Color.bright_red.toFgCode());
}

test "parse key - single chars" {
    try std.testing.expectEqual(Key.ctrl_c, parseKey(&[_]u8{0x03}));
    try std.testing.expectEqual(Key.enter, parseKey(&[_]u8{0x0D}));
    try std.testing.expectEqual(Key{ .char = 'a' }, parseKey(&[_]u8{'a'}));
}

test "parse key - escape sequences" {
    try std.testing.expectEqual(Key.up, parseKey(&[_]u8{ 0x1B, '[', 'A' }));
    try std.testing.expectEqual(Key.down, parseKey(&[_]u8{ 0x1B, '[', 'B' }));
    try std.testing.expectEqual(Key.right, parseKey(&[_]u8{ 0x1B, '[', 'C' }));
    try std.testing.expectEqual(Key.left, parseKey(&[_]u8{ 0x1B, '[', 'D' }));
}
