//! HuggingFace Hub integration.
//!
//! Download models from HuggingFace Hub and convert to .tenzor format.
//!
//! Example:
//!   var hf = HuggingFace.init(allocator, null);
//!   defer hf.deinit();
//!
//!   const model = try hf.downloadModel("Snowflake/snowflake-arctic-embed-xs", null);
//!   std.debug.print("Model downloaded to: {s}\n", .{model.path});

const std = @import("std");
const tenzor_format = @import("tenzor_format.zig");

const HF_BASE_URL = "https://huggingface.co";
const DEFAULT_CACHE_DIR = ".cache/tenzor/models";

pub const HuggingFace = struct {
    allocator: std.mem.Allocator,
    cache_dir: []const u8,
    cache_dir_owned: bool,
    threaded: std.Io.Threaded,
    client: std.http.Client,

    pub fn init(allocator: std.mem.Allocator, cache_dir: ?[]const u8) HuggingFace {
        var cache_dir_owned = false;
        const dir = cache_dir orelse blk: {
            // Use ~/.cache/tenzor/models as default
            if (std.posix.getenv("HOME")) |home| {
                var buf: [512]u8 = undefined;
                const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ home, DEFAULT_CACHE_DIR }) catch DEFAULT_CACHE_DIR;
                const duped = allocator.dupe(u8, path) catch DEFAULT_CACHE_DIR;
                if (duped.ptr != DEFAULT_CACHE_DIR.ptr) cache_dir_owned = true;
                break :blk duped;
            }
            break :blk DEFAULT_CACHE_DIR;
        };

        return .{
            .allocator = allocator,
            .cache_dir = dir,
            .cache_dir_owned = cache_dir_owned,
            .threaded = std.Io.Threaded.init(allocator),
            // client.io will be set in ensureClient()
            .client = .{
                .allocator = allocator,
                .io = undefined, // Will be initialized on first use
            },
        };
    }

    /// Ensure the client's io pointer is valid (must be called after struct is stable)
    fn ensureClient(self: *HuggingFace) void {
        self.client.io = self.threaded.io();
    }

    pub fn deinit(self: *HuggingFace) void {
        self.client.deinit();
        self.threaded.deinit();
        if (self.cache_dir_owned) {
            self.allocator.free(self.cache_dir);
        }
    }

    pub const ModelInfo = struct {
        model_path: []const u8,
        vocab_path: ?[]const u8,
        config_json: ?[]const u8,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *ModelInfo) void {
            self.allocator.free(self.model_path);
            if (self.vocab_path) |p| self.allocator.free(p);
            if (self.config_json) |c| self.allocator.free(c);
        }
    };

    pub const DownloadError = error{
        HttpError,
        InvalidModelId,
        ModelNotFound,
        ParseError,
        IoError,
        OutOfMemory,
    };

    /// Download a model from HuggingFace Hub.
    ///
    /// model_id: e.g., "Snowflake/snowflake-arctic-embed-xs" or "bert-base-uncased"
    pub fn downloadModel(
        self: *HuggingFace,
        model_id: []const u8,
    ) DownloadError!ModelInfo {
        // Validate model_id format
        if (model_id.len == 0) return error.InvalidModelId;

        // Create cache directory structure
        const model_dir = self.getModelCacheDir(model_id) catch return error.IoError;
        defer self.allocator.free(model_dir);

        std.fs.cwd().makePath(model_dir) catch return error.IoError;

        // Files to download
        const files_to_try = [_][]const u8{
            "model.safetensors",
            "vocab.txt",
            "config.json",
        };

        var model_path: ?[]const u8 = null;
        var vocab_path: ?[]const u8 = null;
        var config_json: ?[]const u8 = null;

        for (files_to_try) |filename| {
            const local_path = std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ model_dir, filename }) catch return error.OutOfMemory;

            // Check if file already exists
            if (std.fs.cwd().access(local_path, .{})) |_| {
                // File exists, use cached version
                std.debug.print("  Using cached: {s}\n", .{filename});
                if (std.mem.eql(u8, filename, "model.safetensors")) {
                    model_path = local_path;
                } else if (std.mem.eql(u8, filename, "vocab.txt")) {
                    vocab_path = local_path;
                } else if (std.mem.eql(u8, filename, "config.json")) {
                    config_json = self.readFile(local_path) catch null;
                    self.allocator.free(local_path);
                } else {
                    self.allocator.free(local_path);
                }
                continue;
            } else |_| {}

            // Download file
            const url = std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}/resolve/main/{s}",
                .{ HF_BASE_URL, model_id, filename },
            ) catch return error.OutOfMemory;
            defer self.allocator.free(url);

            std.debug.print("  Downloading: {s}...\n", .{filename});

            if (self.downloadFile(url, local_path)) |_| {
                if (std.mem.eql(u8, filename, "model.safetensors")) {
                    model_path = local_path;
                } else if (std.mem.eql(u8, filename, "vocab.txt")) {
                    vocab_path = local_path;
                } else if (std.mem.eql(u8, filename, "config.json")) {
                    config_json = self.readFile(local_path) catch null;
                    self.allocator.free(local_path);
                } else {
                    self.allocator.free(local_path);
                }
            } else |err| {
                self.allocator.free(local_path);
                // Only fail if model.safetensors is missing
                if (std.mem.eql(u8, filename, "model.safetensors")) {
                    std.debug.print("  Failed to download {s}: {}\n", .{ filename, err });
                    return error.ModelNotFound;
                }
            }
        }

        if (model_path == null) {
            return error.ModelNotFound;
        }

        return ModelInfo{
            .model_path = model_path.?,
            .vocab_path = vocab_path,
            .config_json = config_json,
            .allocator = self.allocator,
        };
    }

    /// Get the cache directory path for a model.
    fn getModelCacheDir(self: *HuggingFace, model_id: []const u8) ![]const u8 {
        // model_id format: "org/model" or just "model"
        // Replace "/" with "--" for flat directory structure
        var buf: [512]u8 = undefined;
        var i: usize = 0;

        for (model_id) |c| {
            if (i >= buf.len - 1) break;
            buf[i] = if (c == '/') '-' else c;
            i += 1;
        }

        return std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.cache_dir, buf[0..i] });
    }

    /// Read a file into memory.
    fn readFile(self: *HuggingFace, path: []const u8) ![]const u8 {
        const file = std.fs.cwd().openFile(path, .{}) catch return error.IoError;
        defer file.close();

        const stat = file.stat() catch return error.IoError;
        const data = self.allocator.alloc(u8, stat.size) catch return error.OutOfMemory;
        errdefer self.allocator.free(data);

        var total_read: usize = 0;
        while (total_read < stat.size) {
            const n = file.read(data[total_read..]) catch return error.IoError;
            if (n == 0) break;
            total_read += n;
        }

        return data[0..total_read];
    }

    /// Download a file from URL to local path.
    fn downloadFile(self: *HuggingFace, url: []const u8, local_path: []const u8) !void {
        // Ensure client.io is properly initialized
        self.ensureClient();

        // Create an allocating writer to accumulate response
        var alloc_writer = std.Io.Writer.Allocating.init(self.allocator);
        defer alloc_writer.deinit();

        // Use fetch API
        const result = self.client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &alloc_writer.writer,
        }) catch return error.HttpError;

        if (result.status != .ok) {
            if (result.status == .not_found) return error.ModelNotFound;
            return error.HttpError;
        }

        // Write accumulated data to file
        const data = alloc_writer.toOwnedSlice() catch return error.OutOfMemory;
        defer self.allocator.free(data);

        const file = std.fs.cwd().createFile(local_path, .{}) catch return error.IoError;
        defer file.close();

        file.writeAll(data) catch return error.IoError;
    }

    /// Download model and convert to .tenzor format.
    pub fn downloadAndConvert(
        self: *HuggingFace,
        model_id: []const u8,
        output_path: ?[]const u8,
    ) ![]const u8 {
        var info = try self.downloadModel(model_id);
        defer info.deinit();

        // Determine output path
        const tenzor_path = if (output_path) |p|
            try self.allocator.dupe(u8, p)
        else blk: {
            // Replace .safetensors with .tenzor
            if (std.mem.endsWith(u8, info.model_path, ".safetensors")) {
                const base_len = info.model_path.len - ".safetensors".len;
                const path = try self.allocator.alloc(u8, base_len + 7);
                @memcpy(path[0..base_len], info.model_path[0..base_len]);
                @memcpy(path[base_len..], ".tenzor");
                break :blk path;
            } else {
                break :blk try std.fmt.allocPrint(self.allocator, "{s}.tenzor", .{info.model_path});
            }
        };

        // Check if .tenzor already exists
        if (std.fs.cwd().access(tenzor_path, .{})) |_| {
            std.debug.print("  Using cached: {s}\n", .{std.fs.path.basename(tenzor_path)});
            return tenzor_path;
        } else |_| {}

        // Convert to .tenzor format
        std.debug.print("  Converting to .tenzor format...\n", .{});
        tenzor_format.convertFromSafetensors(self.allocator, info.model_path, tenzor_path) catch |err| {
            self.allocator.free(tenzor_path);
            return err;
        };

        return tenzor_path;
    }

    pub const OnnxModelInfo = struct {
        /// Path to the .onnx file
        model_path: []const u8,
        /// Directory containing model and external data
        model_dir: []const u8,
        /// List of downloaded external data files
        external_files: []const []const u8,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *OnnxModelInfo) void {
            self.allocator.free(self.model_path);
            self.allocator.free(self.model_dir);
            for (self.external_files) |f| self.allocator.free(f);
            self.allocator.free(self.external_files);
        }
    };

    /// Download an ONNX model and its external data files from HuggingFace Hub.
    ///
    /// model_id: e.g., "ResembleAI/chatterbox-turbo-ONNX"
    /// onnx_file: e.g., "embed_tokens.onnx" or "onnx/model.onnx"
    /// external_data_files: list of external data files to download, e.g., &.{"embed_tokens.onnx_data"}
    pub fn downloadOnnxModel(
        self: *HuggingFace,
        model_id: []const u8,
        onnx_file: []const u8,
        external_data_files: []const []const u8,
    ) DownloadError!OnnxModelInfo {
        if (model_id.len == 0) return error.InvalidModelId;

        // Create cache directory
        const model_dir = self.getModelCacheDir(model_id) catch return error.IoError;
        errdefer self.allocator.free(model_dir);

        std.fs.cwd().makePath(model_dir) catch return error.IoError;

        // Download the .onnx file
        const onnx_local = std.fmt.allocPrint(self.allocator, "{s}/{s}", .{
            model_dir,
            std.fs.path.basename(onnx_file),
        }) catch return error.OutOfMemory;
        errdefer self.allocator.free(onnx_local);

        // Check cache first
        if (std.fs.cwd().access(onnx_local, .{})) |_| {
            std.debug.print("  Using cached: {s}\n", .{std.fs.path.basename(onnx_file)});
        } else |_| {
            const url = std.fmt.allocPrint(
                self.allocator,
                "{s}/{s}/resolve/main/{s}",
                .{ HF_BASE_URL, model_id, onnx_file },
            ) catch return error.OutOfMemory;
            defer self.allocator.free(url);

            std.debug.print("  Downloading: {s}...\n", .{onnx_file});
            self.downloadFile(url, onnx_local) catch return error.ModelNotFound;
        }

        // Download external data files
        var ext_paths: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (ext_paths.items) |p| self.allocator.free(p);
            ext_paths.deinit(self.allocator);
        }

        for (external_data_files) |ext_file| {
            const ext_local = std.fmt.allocPrint(self.allocator, "{s}/{s}", .{
                model_dir,
                std.fs.path.basename(ext_file),
            }) catch return error.OutOfMemory;
            errdefer self.allocator.free(ext_local);

            if (std.fs.cwd().access(ext_local, .{})) |_| {
                std.debug.print("  Using cached: {s}\n", .{std.fs.path.basename(ext_file)});
            } else |_| {
                const url = std.fmt.allocPrint(
                    self.allocator,
                    "{s}/{s}/resolve/main/{s}",
                    .{ HF_BASE_URL, model_id, ext_file },
                ) catch return error.OutOfMemory;
                defer self.allocator.free(url);

                std.debug.print("  Downloading: {s}...\n", .{ext_file});
                self.downloadFile(url, ext_local) catch {
                    self.allocator.free(ext_local);
                    continue; // Skip optional files
                };
            }

            ext_paths.append(self.allocator, ext_local) catch return error.OutOfMemory;
        }

        return OnnxModelInfo{
            .model_path = onnx_local,
            .model_dir = model_dir,
            .external_files = ext_paths.toOwnedSlice(self.allocator) catch return error.OutOfMemory,
            .allocator = self.allocator,
        };
    }

    /// Download an ONNX model, automatically detecting external data files from the model.
    /// Parses the .onnx file to find external data references.
    pub fn downloadOnnxModelAuto(
        self: *HuggingFace,
        model_id: []const u8,
        onnx_file: []const u8,
    ) DownloadError!OnnxModelInfo {
        // First download just the .onnx file
        var info = try self.downloadOnnxModel(model_id, onnx_file, &.{});
        errdefer info.deinit();

        // Parse it to find external data references
        const onnx = @import("../onnx/root.zig");
        const data = self.readFile(info.model_path) catch return error.IoError;
        defer self.allocator.free(data);

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        const model = onnx.parser.parseModel(arena.allocator(), data) catch return error.ParseError;

        // Collect unique external data locations
        var ext_files = std.StringHashMap(void).init(self.allocator);
        defer ext_files.deinit();

        if (model.graph) |graph| {
            for (graph.initializer) |tensor| {
                if (tensor.getExternalLocation()) |loc| {
                    ext_files.put(loc, {}) catch return error.OutOfMemory;
                }
            }
        }

        // Download external files
        var ext_list: std.ArrayListUnmanaged([]const u8) = .empty;
        defer ext_list.deinit(self.allocator);

        var it = ext_files.keyIterator();
        while (it.next()) |key| {
            // Build path relative to onnx file
            const dir = std.fs.path.dirname(onnx_file) orelse "";
            const ext_path = if (dir.len > 0)
                std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir, key.* }) catch return error.OutOfMemory
            else
                self.allocator.dupe(u8, key.*) catch return error.OutOfMemory;
            defer self.allocator.free(ext_path);

            const ext_local = std.fmt.allocPrint(self.allocator, "{s}/{s}", .{
                info.model_dir,
                std.fs.path.basename(ext_path),
            }) catch return error.OutOfMemory;

            if (std.fs.cwd().access(ext_local, .{})) |_| {
                std.debug.print("  Using cached: {s}\n", .{key.*});
                ext_list.append(self.allocator, ext_local) catch return error.OutOfMemory;
            } else |_| {
                const url = std.fmt.allocPrint(
                    self.allocator,
                    "{s}/{s}/resolve/main/{s}",
                    .{ HF_BASE_URL, model_id, ext_path },
                ) catch return error.OutOfMemory;
                defer self.allocator.free(url);

                std.debug.print("  Downloading: {s}...\n", .{key.*});
                if (self.downloadFile(url, ext_local)) |_| {
                    ext_list.append(self.allocator, ext_local) catch return error.OutOfMemory;
                } else |_| {
                    self.allocator.free(ext_local);
                }
            }
        }

        // Update info with external files
        self.allocator.free(info.external_files);
        info.external_files = ext_list.toOwnedSlice(self.allocator) catch return error.OutOfMemory;

        return info;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "huggingface init" {
    const allocator = std.testing.allocator;
    var hf = HuggingFace.init(allocator, "/tmp/test_cache");
    defer hf.deinit();

    try std.testing.expectEqualStrings("/tmp/test_cache", hf.cache_dir);
}

test "model cache dir" {
    const allocator = std.testing.allocator;
    var hf = HuggingFace.init(allocator, "/tmp/cache");
    defer hf.deinit();

    const dir = try hf.getModelCacheDir("Snowflake/snowflake-arctic-embed-xs");
    defer allocator.free(dir);

    try std.testing.expectEqualStrings("/tmp/cache/Snowflake-snowflake-arctic-embed-xs", dir);
}
