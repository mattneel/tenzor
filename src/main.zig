//! tenzor CLI - Train and run tensor models
//!
//! Subcommands:
//!   train   Train LeNet-5 on MNIST dataset
//!   embed   Generate text embeddings with Arctic-embed-xs

const std = @import("std");
const clap = @import("clap");
const tenzor = @import("tenzor");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    run(allocator) catch |err| {
        if (err != error.Help) {
            std.debug.print("Error: {}\n", .{err});
            std.process.exit(1);
        }
    };
}

fn run(allocator: std.mem.Allocator) !void {
    const parsers = comptime .{
        .PATH = clap.parsers.string,
        .NUM = clap.parsers.int(u32, 10),
        .FLOAT = clap.parsers.float(f32),
        .SEED = clap.parsers.int(u64, 10),
        .TEXT = clap.parsers.string,
    };

    const params = comptime clap.parseParamsComptime(
        \\-h, --help             Show this help message
        \\-d, --data-dir <PATH>  Path to MNIST data directory (default: data/mnist)
        \\-e, --epochs <NUM>     Number of training epochs (default: 10)
        \\-b, --batch-size <NUM> Batch size (default: 64)
        \\-l, --lr <FLOAT>       Learning rate (default: 0.01)
        \\    --momentum <FLOAT> SGD momentum (default: 0.9)
        \\-s, --seed <SEED>      Random seed (default: 42)
        \\-m, --model <PATH>     Path to model.safetensors
        \\<TEXT>...
        \\
    );

    var res = clap.parse(clap.Help, &params, parsers, .{
        .allocator = allocator,
    }) catch |err| {
        std.debug.print("Error parsing arguments: {}\n", .{err});
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        return printHelp(&params);
    }

    // Get subcommand from positional args (first element of the variadic TEXT positional)
    const positionals = res.positionals[0];
    if (positionals.len == 0) {
        return printHelp(&params);
    }

    const subcommand = positionals[0];

    if (std.mem.eql(u8, subcommand, "train")) {
        try runTrain(allocator, .{
            .data_dir = res.args.@"data-dir" orelse "data/mnist",
            .epochs = res.args.epochs orelse 10,
            .batch_size = res.args.@"batch-size" orelse 64,
            .learning_rate = res.args.lr orelse 0.01,
            .momentum = res.args.momentum orelse 0.9,
            .seed = res.args.seed orelse 42,
        });
    } else if (std.mem.eql(u8, subcommand, "embed")) {
        const model_path = res.args.model orelse {
            std.debug.print("Error: --model is required for embed command\n\n", .{});
            return printHelp(&params);
        };

        // Remaining positionals are texts
        const texts = positionals[1..];

        if (texts.len == 0) {
            std.debug.print("Error: At least one text argument is required\n\n", .{});
            return printHelp(&params);
        }

        try runEmbed(allocator, .{
            .model_path = model_path,
            .texts = texts,
        });
    } else if (std.mem.eql(u8, subcommand, "help")) {
        return printHelp(&params);
    } else {
        std.debug.print("Unknown command: {s}\n\n", .{subcommand});
        return printHelp(&params);
    }
}

fn printHelp(params: anytype) error{Help} {
    const help_text =
        \\tenzor - A comptime tensor library for Zig
        \\
        \\USAGE:
        \\  tenzor <COMMAND> [OPTIONS]
        \\
        \\COMMANDS:
        \\  train    Train LeNet-5 on MNIST dataset
        \\  embed    Generate text embeddings with Arctic-embed-xs
        \\  help     Show this help message
        \\
        \\TRAIN OPTIONS:
        \\  -d, --data-dir <PATH>   MNIST data directory (default: data/mnist)
        \\  -e, --epochs <NUM>      Training epochs (default: 10)
        \\  -b, --batch-size <NUM>  Batch size (default: 64)
        \\  -l, --lr <FLOAT>        Learning rate (default: 0.01)
        \\      --momentum <FLOAT>  SGD momentum (default: 0.9)
        \\  -s, --seed <SEED>       Random seed (default: 42)
        \\
        \\EMBED OPTIONS:
        \\  -m, --model <PATH>      Path to model.safetensors (required)
        \\  <TEXT>...               Text(s) to embed
        \\
        \\EXAMPLES:
        \\  tenzor train --data-dir data/mnist --epochs 10 --lr 0.01
        \\  tenzor embed --model models/arctic-embed-xs/model.safetensors "Hello world"
        \\
    ;
    _ = params;
    std.debug.print("{s}", .{help_text});
    return error.Help;
}

// ============================================================================
// Train Command
// ============================================================================

const TrainArgs = struct {
    data_dir: []const u8,
    epochs: u32,
    batch_size: u32,
    learning_rate: f32,
    momentum: f32,
    seed: u64,
};

fn runTrain(allocator: std.mem.Allocator, args: TrainArgs) !void {
    const lenet = tenzor.model.lenet;
    const mnist = tenzor.io.mnist;

    std.debug.print("LeNet-5 MNIST Training\n", .{});
    std.debug.print("======================\n", .{});
    std.debug.print("Data directory: {s}\n", .{args.data_dir});
    std.debug.print("Epochs: {d}\n", .{args.epochs});
    std.debug.print("Batch size: {d}\n", .{args.batch_size});
    std.debug.print("Learning rate: {d:.4}\n", .{args.learning_rate});
    std.debug.print("Momentum: {d:.2}\n", .{args.momentum});
    std.debug.print("Seed: {d}\n\n", .{args.seed});

    // Build file paths
    var train_images_buf: [256]u8 = undefined;
    var train_labels_buf: [256]u8 = undefined;
    var test_images_buf: [256]u8 = undefined;
    var test_labels_buf: [256]u8 = undefined;

    const train_images = std.fmt.bufPrint(&train_images_buf, "{s}/train-images-idx3-ubyte", .{args.data_dir}) catch return error.PathTooLong;
    const train_labels = std.fmt.bufPrint(&train_labels_buf, "{s}/train-labels-idx1-ubyte", .{args.data_dir}) catch return error.PathTooLong;
    const test_images = std.fmt.bufPrint(&test_images_buf, "{s}/t10k-images-idx3-ubyte", .{args.data_dir}) catch return error.PathTooLong;
    const test_labels = std.fmt.bufPrint(&test_labels_buf, "{s}/t10k-labels-idx1-ubyte", .{args.data_dir}) catch return error.PathTooLong;

    // Load data
    std.debug.print("Loading training data...\n", .{});
    var train_data = mnist.MNISTDataset.load(allocator, train_images, train_labels) catch |err| {
        std.debug.print("Error loading training data: {}\n", .{err});
        std.debug.print("Make sure MNIST files exist in {s}/\n", .{args.data_dir});
        std.debug.print("\nDownload from: http://yann.lecun.com/exdb/mnist/\n", .{});
        std.debug.print("Expected files:\n", .{});
        std.debug.print("  train-images-idx3-ubyte\n", .{});
        std.debug.print("  train-labels-idx1-ubyte\n", .{});
        std.debug.print("  t10k-images-idx3-ubyte\n", .{});
        std.debug.print("  t10k-labels-idx1-ubyte\n", .{});
        return err;
    };
    defer train_data.deinit();
    std.debug.print("  Loaded {d} training samples\n", .{train_data.num_samples});

    std.debug.print("Loading test data...\n", .{});
    var test_data = mnist.MNISTDataset.load(allocator, test_images, test_labels) catch |err| {
        std.debug.print("Error loading test data: {}\n", .{err});
        return err;
    };
    defer test_data.deinit();
    std.debug.print("  Loaded {d} test samples\n\n", .{test_data.num_samples});

    // Initialize model
    const config = lenet.LeNetConfig{ .batch_size = args.batch_size };
    var model = try lenet.LeNet.init(allocator, config);
    defer model.deinit();

    // Initialize weights
    var prng = std.Random.DefaultPrng.init(args.seed);
    model.weights.initKaiming(prng.random());

    std.debug.print("Training...\n", .{});
    const num_batches = train_data.numBatches(args.batch_size);

    for (0..args.epochs) |epoch| {
        // Shuffle training data
        train_data.shuffle(prng.random());

        var epoch_loss: f32 = 0;
        var epoch_correct: usize = 0;
        var epoch_total: usize = 0;

        for (0..num_batches) |batch_idx| {
            const batch = train_data.getBatch(batch_idx, args.batch_size);
            const actual_batch = batch.labels.len;

            // Forward pass
            _ = model.forward(batch.images, actual_batch);

            // Compute loss and accuracy
            const metrics = model.computeLoss(batch.labels, actual_batch);
            epoch_loss += metrics.loss;
            epoch_correct += @intFromFloat(metrics.accuracy * @as(f32, @floatFromInt(actual_batch)));
            epoch_total += actual_batch;

            // Backward pass
            model.grads.zero();
            model.computeLossGradient(batch.labels, actual_batch);
            model.backward(batch.images, actual_batch);

            // SGD update
            sgdUpdate(&model, args.learning_rate);
        }

        // Evaluate on test set
        var test_correct: usize = 0;
        var test_total: usize = 0;
        const test_batches = test_data.numBatches(args.batch_size);

        for (0..test_batches) |batch_idx| {
            const batch = test_data.getBatch(batch_idx, args.batch_size);
            const actual_batch = batch.labels.len;

            _ = model.forward(batch.images, actual_batch);
            const metrics = model.computeLoss(batch.labels, actual_batch);
            test_correct += @intFromFloat(metrics.accuracy * @as(f32, @floatFromInt(actual_batch)));
            test_total += actual_batch;
        }

        const train_acc = @as(f32, @floatFromInt(epoch_correct)) / @as(f32, @floatFromInt(epoch_total)) * 100.0;
        const test_acc = @as(f32, @floatFromInt(test_correct)) / @as(f32, @floatFromInt(test_total)) * 100.0;

        std.debug.print("Epoch {d:2}/{d}: loss={d:.4}, train_acc={d:.1}%, test_acc={d:.1}%\n", .{
            epoch + 1,
            args.epochs,
            epoch_loss / @as(f32, @floatFromInt(num_batches)),
            train_acc,
            test_acc,
        });
    }

    std.debug.print("\nTraining complete!\n", .{});
}

fn sgdUpdate(model: *tenzor.model.lenet.LeNet, lr: f32) void {
    updateParams(model.weights.conv1_weight, model.grads.conv1_weight, lr);
    updateParams(model.weights.conv1_bias, model.grads.conv1_bias, lr);
    updateParams(model.weights.conv2_weight, model.grads.conv2_weight, lr);
    updateParams(model.weights.conv2_bias, model.grads.conv2_bias, lr);
    updateParams(model.weights.fc1_weight, model.grads.fc1_weight, lr);
    updateParams(model.weights.fc1_bias, model.grads.fc1_bias, lr);
    updateParams(model.weights.fc2_weight, model.grads.fc2_weight, lr);
    updateParams(model.weights.fc2_bias, model.grads.fc2_bias, lr);
    updateParams(model.weights.fc3_weight, model.grads.fc3_weight, lr);
    updateParams(model.weights.fc3_bias, model.grads.fc3_bias, lr);
}

fn updateParams(params: []f32, grads: []const f32, lr: f32) void {
    for (params, grads) |*p, g| {
        p.* -= lr * g;
    }
}

// ============================================================================
// Embed Command
// ============================================================================

const EmbedArgs = struct {
    model_path: []const u8,
    texts: []const []const u8,
};

fn runEmbed(allocator: std.mem.Allocator, args: EmbedArgs) !void {
    const arctic = tenzor.model.arctic;
    const safetensors = tenzor.io.safetensors;

    std.debug.print("Arctic-embed-xs Text Embedding\n", .{});
    std.debug.print("==============================\n", .{});
    std.debug.print("Model: {s}\n", .{args.model_path});
    std.debug.print("Texts: {d}\n\n", .{args.texts.len});

    // Load model weights
    std.debug.print("Loading model...\n", .{});
    var loaded = safetensors.load(allocator, args.model_path) catch |err| {
        std.debug.print("Error loading model: {}\n", .{err});
        return err;
    };
    defer loaded.st.deinit();
    defer allocator.free(loaded.data);

    const config = arctic.arctic_embed_xs_config;
    var weights = try arctic.ModelWeights.fromSafeTensors(allocator, loaded.st, config);
    defer weights.deinit(allocator);

    // Create inference context
    const max_seq_len: usize = 128;
    var ctx = try arctic.InferenceContext.init(allocator, config, max_seq_len);
    defer ctx.deinit();

    std.debug.print("Generating embeddings...\n\n", .{});
    std.debug.print("[\n", .{});

    for (args.texts, 0..) |text, i| {
        // Simple tokenization: for demonstration, we use character codes
        // In production, you'd use a proper WordPiece tokenizer
        var token_ids: [128]u32 = undefined;
        token_ids[0] = 101; // [CLS]

        var seq_len: usize = 1;
        for (text) |c| {
            if (seq_len >= 127) break;
            // Map ASCII to vocab IDs (simplified - real tokenizer needed)
            token_ids[seq_len] = @as(u32, c) + 1000;
            seq_len += 1;
        }
        token_ids[seq_len] = 102; // [SEP]
        seq_len += 1;

        // Generate embedding
        var embedding: [384]f32 = undefined;
        arctic.forward(&embedding, token_ids[0..seq_len], weights, &ctx);

        // Output as JSON
        std.debug.print("  {{\n    \"text\": \"{s}\",\n    \"embedding\": [", .{text});
        for (embedding, 0..) |v, j| {
            if (j > 0) std.debug.print(", ", .{});
            if (j % 10 == 0 and j > 0) std.debug.print("\n      ", .{});
            std.debug.print("{d:.6}", .{v});
        }
        std.debug.print("]\n  }}", .{});
        if (i < args.texts.len - 1) std.debug.print(",", .{});
        std.debug.print("\n", .{});
    }

    std.debug.print("]\n", .{});
}

test "main imports" {
    // Verify imports work
    _ = tenzor;
    _ = clap;
}
