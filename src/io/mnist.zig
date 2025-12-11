//! MNIST IDX file format loader.
//!
//! IDX format specification:
//! - 4 bytes: magic number (2051 for images, 2049 for labels)
//! - 4 bytes: number of items (big-endian u32)
//! - For images: 4 bytes rows, 4 bytes cols, then pixel data
//! - For labels: label bytes directly
//!
//! Reference: http://yann.lecun.com/exdb/mnist/

const std = @import("std");

pub const MNISTDataset = struct {
    allocator: std.mem.Allocator,
    images: []f32, // [N, 28, 28, 1] flattened, normalized to [0, 1]
    labels: []u32, // [N]
    num_samples: usize,

    /// Load MNIST dataset from IDX files.
    pub fn load(
        allocator: std.mem.Allocator,
        images_path: []const u8,
        labels_path: []const u8,
    ) !MNISTDataset {
        // Load images
        const images_file = try std.fs.cwd().openFile(images_path, .{});
        defer images_file.close();

        var images_header: [16]u8 = undefined;
        _ = try images_file.readAll(&images_header);

        const images_magic = readU32BE(images_header[0..4]);
        if (images_magic != 2051) return error.InvalidMagicNumber;

        const num_images = readU32BE(images_header[4..8]);
        const rows = readU32BE(images_header[8..12]);
        const cols = readU32BE(images_header[12..16]);

        if (rows != 28 or cols != 28) return error.UnexpectedImageSize;

        // Allocate and read images
        const image_size = rows * cols;
        const raw_images = try allocator.alloc(u8, num_images * image_size);
        defer allocator.free(raw_images);

        var total_read: usize = 0;
        while (total_read < raw_images.len) {
            const n = try images_file.read(raw_images[total_read..]);
            if (n == 0) return error.UnexpectedEOF;
            total_read += n;
        }

        // Normalize to f32 [0, 1] in NHWC format
        const images = try allocator.alloc(f32, num_images * image_size);
        errdefer allocator.free(images);

        for (raw_images, images) |pixel, *f| {
            f.* = @as(f32, @floatFromInt(pixel)) / 255.0;
        }

        // Load labels
        const labels_file = try std.fs.cwd().openFile(labels_path, .{});
        defer labels_file.close();

        var labels_header: [8]u8 = undefined;
        _ = try labels_file.readAll(&labels_header);

        const labels_magic = readU32BE(labels_header[0..4]);
        if (labels_magic != 2049) return error.InvalidMagicNumber;

        const num_labels = readU32BE(labels_header[4..8]);
        if (num_labels != num_images) return error.MismatchedCounts;

        const raw_labels = try allocator.alloc(u8, num_labels);
        defer allocator.free(raw_labels);

        total_read = 0;
        while (total_read < raw_labels.len) {
            const n = try labels_file.read(raw_labels[total_read..]);
            if (n == 0) return error.UnexpectedEOF;
            total_read += n;
        }

        const labels = try allocator.alloc(u32, num_labels);
        errdefer allocator.free(labels);

        for (raw_labels, labels) |l, *u| {
            u.* = @as(u32, l);
        }

        return .{
            .allocator = allocator,
            .images = images,
            .labels = labels,
            .num_samples = num_images,
        };
    }

    pub fn deinit(self: *MNISTDataset) void {
        self.allocator.free(self.images);
        self.allocator.free(self.labels);
    }

    /// Get a batch of data.
    /// Returns slices into the dataset (no copy).
    pub fn getBatch(
        self: *const MNISTDataset,
        batch_idx: usize,
        batch_size: usize,
    ) struct {
        images: []const f32,
        labels: []const u32,
    } {
        const start = batch_idx * batch_size;
        const end = @min(start + batch_size, self.num_samples);

        const image_size = 28 * 28;
        return .{
            .images = self.images[start * image_size .. end * image_size],
            .labels = self.labels[start..end],
        };
    }

    /// Shuffle dataset in-place using Fisher-Yates.
    pub fn shuffle(self: *MNISTDataset, rng: std.Random) void {
        const image_size = 28 * 28;

        var i: usize = self.num_samples - 1;
        while (i > 0) : (i -= 1) {
            const j = rng.intRangeAtMost(usize, 0, i);

            // Swap labels
            const tmp_label = self.labels[i];
            self.labels[i] = self.labels[j];
            self.labels[j] = tmp_label;

            // Swap images (swap each pixel)
            const img_i = self.images[i * image_size ..][0..image_size];
            const img_j = self.images[j * image_size ..][0..image_size];
            for (img_i, img_j) |*a, *b| {
                const tmp = a.*;
                a.* = b.*;
                b.* = tmp;
            }
        }
    }

    /// Get number of batches for given batch size.
    pub fn numBatches(self: *const MNISTDataset, batch_size: usize) usize {
        return self.num_samples / batch_size;
    }
};

/// Read big-endian u32 from bytes.
fn readU32BE(bytes: *const [4]u8) u32 {
    return std.mem.readInt(u32, bytes, .big);
}

// ============================================================================
// Tests
// ============================================================================

test "read u32 big endian" {
    const bytes = [4]u8{ 0x00, 0x00, 0x00, 0x05 };
    try std.testing.expectEqual(@as(u32, 5), readU32BE(&bytes));

    const bytes2 = [4]u8{ 0x00, 0x00, 0x08, 0x03 }; // 2051 = 0x0803
    try std.testing.expectEqual(@as(u32, 2051), readU32BE(&bytes2));
}

test "batch calculation" {
    // Mock a dataset manually for batch testing
    const allocator = std.testing.allocator;

    // Create tiny "dataset"
    const num_samples: usize = 10;
    const images = try allocator.alloc(f32, num_samples * 28 * 28);
    defer allocator.free(images);
    @memset(images, 0.5);

    const labels = try allocator.alloc(u32, num_samples);
    defer allocator.free(labels);
    for (labels, 0..) |*l, i| l.* = @intCast(i % 10);

    var dataset = MNISTDataset{
        .allocator = allocator,
        .images = images,
        .labels = labels,
        .num_samples = num_samples,
    };

    // Test batch access
    const batch_size: usize = 3;
    try std.testing.expectEqual(@as(usize, 3), dataset.numBatches(batch_size));

    const batch0 = dataset.getBatch(0, batch_size);
    try std.testing.expectEqual(@as(usize, 3 * 28 * 28), batch0.images.len);
    try std.testing.expectEqual(@as(usize, 3), batch0.labels.len);

    const batch1 = dataset.getBatch(1, batch_size);
    try std.testing.expectEqual(@as(u32, 3), batch1.labels[0]);
    try std.testing.expectEqual(@as(u32, 4), batch1.labels[1]);

    // Don't call deinit since we manually allocated
}

test "shuffle preserves data" {
    const allocator = std.testing.allocator;

    const num_samples: usize = 5;
    const images = try allocator.alloc(f32, num_samples * 28 * 28);
    defer allocator.free(images);

    // Mark each image with its index
    for (0..num_samples) |i| {
        images[i * 28 * 28] = @floatFromInt(i);
    }

    const labels = try allocator.alloc(u32, num_samples);
    defer allocator.free(labels);
    for (labels, 0..) |*l, i| l.* = @intCast(i);

    var dataset = MNISTDataset{
        .allocator = allocator,
        .images = images,
        .labels = labels,
        .num_samples = num_samples,
    };

    var prng = std.Random.DefaultPrng.init(42);
    dataset.shuffle(prng.random());

    // Check that each label still matches its image marker
    for (0..num_samples) |i| {
        const label = dataset.labels[i];
        const marker = dataset.images[i * 28 * 28];
        try std.testing.expectEqual(@as(f32, @floatFromInt(label)), marker);
    }
}
