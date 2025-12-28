//! tenzor CLI - Train and run tensor models
//!
//! Subcommands:
//!   train    Train LeNet-5 on MNIST dataset
//!   embed    Generate text embeddings with Arctic-embed-xs
//!   onnx     Load and run ONNX models
//!   convert  Convert model formats (safetensors -> .tenzor)

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
        \\    --scheduler <TEXT> LR scheduler: constant|step|cosine (default: cosine)
        \\    --warmup <NUM>     Warmup steps (default: 0)
        \\    --patience <NUM>   Early stopping patience (default: 0 = disabled)
        \\    --checkpoint <PATH> Checkpoint directory (default: none)
        \\    --no-tui           Disable TUI dashboard
        \\-s, --seed <SEED>      Random seed (default: 42)
        \\-m, --model <PATH>     Path to model.safetensors
        \\-i, --input <PATH>     Read text from file instead of args
        \\-o, --output <PATH>    Output path for convert command
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
            .scheduler = res.args.scheduler orelse "cosine",
            .warmup = res.args.warmup orelse 0,
            .patience = res.args.patience orelse 0,
            .checkpoint_dir = res.args.checkpoint,
            .use_tui = res.args.@"no-tui" == 0,
        });
    } else if (std.mem.eql(u8, subcommand, "eval")) {
        // Evaluate model checkpoint on test data
        if (positionals.len < 2) {
            std.debug.print("Error: eval requires checkpoint path\n", .{});
            std.debug.print("Usage: tenzor eval <checkpoint.tenzor> [-d data_dir]\n\n", .{});
            return printHelp(&params);
        }

        try runEval(allocator, .{
            .checkpoint_path = positionals[1],
            .data_dir = res.args.@"data-dir" orelse "data/mnist",
            .batch_size = res.args.@"batch-size" orelse 64,
        });
    } else if (std.mem.eql(u8, subcommand, "embed")) {
        const model_path = res.args.model orelse {
            std.debug.print("Error: --model is required for embed command\n\n", .{});
            return printHelp(&params);
        };

        // Get texts from --input file or positional args
        const input_path = res.args.input;
        const positional_texts = positionals[1..];

        if (input_path == null and positional_texts.len == 0) {
            std.debug.print("Error: Provide text via --input <file> or as arguments\n\n", .{});
            return printHelp(&params);
        }

        try runEmbed(allocator, .{
            .model_path = model_path,
            .texts = positional_texts,
            .input_path = input_path,
        });
    } else if (std.mem.eql(u8, subcommand, "info")) {
        // Show .tenzor file info
        if (positionals.len < 2) {
            std.debug.print("Error: info requires file path\n", .{});
            std.debug.print("Usage: tenzor info <file.tenzor>\n\n", .{});
            return printHelp(&params);
        }

        try runInfo(allocator, positionals[1]);
    } else if (std.mem.eql(u8, subcommand, "download")) {
        // Download model from HuggingFace
        if (positionals.len < 2) {
            std.debug.print("Error: download requires model ID\n", .{});
            std.debug.print("Usage: tenzor download <model_id> [-o output.tenzor]\n\n", .{});
            return printHelp(&params);
        }

        const model_id = positionals[1];
        try runDownload(allocator, model_id, res.args.output);
    } else if (std.mem.eql(u8, subcommand, "convert")) {
        // Convert safetensors -> .tenzor
        if (positionals.len < 2) {
            std.debug.print("Error: convert requires input path\n", .{});
            std.debug.print("Usage: tenzor convert <input.safetensors> [-o output.tenzor]\n\n", .{});
            return printHelp(&params);
        }

        const input_path = positionals[1];
        const output_path = res.args.output orelse blk: {
            // Default: replace .safetensors with .tenzor
            var buf: [512]u8 = undefined;
            if (std.mem.endsWith(u8, input_path, ".safetensors")) {
                const base_len = input_path.len - ".safetensors".len;
                @memcpy(buf[0..base_len], input_path[0..base_len]);
                @memcpy(buf[base_len .. base_len + 7], ".tenzor");
                break :blk buf[0 .. base_len + 7];
            } else {
                const len = input_path.len;
                @memcpy(buf[0..len], input_path);
                @memcpy(buf[len .. len + 7], ".tenzor");
                break :blk buf[0 .. len + 7];
            }
        };

        try runConvert(allocator, input_path, output_path);
    } else if (std.mem.eql(u8, subcommand, "onnx")) {
        // ONNX subcommands
        if (positionals.len < 2) {
            std.debug.print("Error: onnx requires a subcommand\n", .{});
            std.debug.print("Usage: tenzor onnx <info|download|run> [OPTIONS]\n\n", .{});
            return printHelp(&params);
        }

        const onnx_cmd = positionals[1];

        if (std.mem.eql(u8, onnx_cmd, "info")) {
            if (positionals.len < 3) {
                std.debug.print("Error: onnx info requires file path\n", .{});
                std.debug.print("Usage: tenzor onnx info <model.onnx>\n\n", .{});
                return printHelp(&params);
            }
            try runOnnxInfo(allocator, positionals[2]);
        } else if (std.mem.eql(u8, onnx_cmd, "download")) {
            if (positionals.len < 4) {
                std.debug.print("Error: onnx download requires model_id and onnx_file\n", .{});
                std.debug.print("Usage: tenzor onnx download <model_id> <onnx_file>\n", .{});
                std.debug.print("Example: tenzor onnx download ResembleAI/chatterbox-turbo-ONNX embed_tokens.onnx\n\n", .{});
                return printHelp(&params);
            }
            try runOnnxDownload(allocator, positionals[2], positionals[3]);
        } else if (std.mem.eql(u8, onnx_cmd, "run")) {
            if (positionals.len < 3) {
                std.debug.print("Error: onnx run requires file path\n", .{});
                std.debug.print("Usage: tenzor onnx run <model.onnx>\n\n", .{});
                return printHelp(&params);
            }
            try runOnnxRun(allocator, positionals[2]);
        } else {
            std.debug.print("Unknown onnx command: {s}\n", .{onnx_cmd});
            std.debug.print("Available: info, download, run\n\n", .{});
            return printHelp(&params);
        }
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
        \\  eval     Evaluate model checkpoint on test data
        \\  embed    Generate text embeddings with Arctic-embed-xs
        \\  onnx     Load and run ONNX models
        \\  download Download model from HuggingFace Hub
        \\  convert  Convert safetensors to .tenzor format
        \\  info     Show .tenzor file information
        \\  help     Show this help message
        \\
        \\TRAIN OPTIONS:
        \\  -d, --data-dir <PATH>   MNIST data directory (default: data/mnist)
        \\  -e, --epochs <NUM>      Training epochs (default: 10)
        \\  -b, --batch-size <NUM>  Batch size (default: 64)
        \\  -l, --lr <FLOAT>        Learning rate (default: 0.01)
        \\      --momentum <FLOAT>  SGD momentum (default: 0.9)
        \\      --scheduler <TYPE>  LR scheduler: constant|step|cosine (default: cosine)
        \\      --warmup <NUM>      Warmup steps (default: 0)
        \\      --patience <NUM>    Early stopping patience (default: disabled)
        \\      --checkpoint <PATH> Checkpoint directory
        \\      --no-tui            Disable TUI dashboard
        \\  -s, --seed <SEED>       Random seed (default: 42)
        \\
        \\EVAL OPTIONS:
        \\  <CHECKPOINT>            Path to checkpoint.tenzor file
        \\  -d, --data-dir <PATH>   MNIST data directory (default: data/mnist)
        \\  -b, --batch-size <NUM>  Batch size (default: 64)
        \\
        \\EMBED OPTIONS:
        \\  -m, --model <PATH>      Path to model.safetensors (required)
        \\  -i, --input <PATH>      Read text from file (for large inputs)
        \\  <TEXT>...               Text(s) to embed (alternative to --input)
        \\
        \\DOWNLOAD OPTIONS:
        \\  <MODEL_ID>              HuggingFace model ID (e.g., Snowflake/snowflake-arctic-embed-xs)
        \\  -o, --output <PATH>     Output .tenzor file (default: cache directory)
        \\
        \\ONNX OPTIONS:
        \\  onnx info <FILE>        Show ONNX model structure
        \\  onnx download <ID> <F>  Download ONNX model from HuggingFace
        \\  onnx run <FILE>         Run inference on ONNX model
        \\
        \\CONVERT OPTIONS:
        \\  <INPUT>                 Input .safetensors file
        \\  -o, --output <PATH>     Output .tenzor file (default: <input>.tenzor)
        \\
        \\EXAMPLES:
        \\  tenzor train --data-dir data/mnist --epochs 10 --lr 0.01
        \\  tenzor train -e 20 --scheduler cosine --warmup 500 --checkpoint ckpt/
        \\  tenzor eval ckpt/best.tenzor -d data/mnist
        \\  tenzor embed --model models/arctic-embed-xs/model.safetensors "Hello world"
        \\  tenzor onnx info model.onnx
        \\  tenzor onnx download ResembleAI/chatterbox-turbo-ONNX embed_tokens.onnx
        \\  tenzor download Snowflake/snowflake-arctic-embed-xs
        \\  tenzor convert model.safetensors -o model.tenzor
        \\  tenzor info model.tenzor
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
    scheduler: []const u8,
    warmup: u32,
    patience: u32,
    checkpoint_dir: ?[]const u8,
    use_tui: bool,
};

fn runTrain(allocator: std.mem.Allocator, args: TrainArgs) !void {
    const lenet = tenzor.model.lenet;
    const mnist = tenzor.io.mnist;
    const Trainer = tenzor.training.Trainer;
    const TrainerConfig = tenzor.training.TrainerConfig;
    const threading = tenzor.backend.cpu.threading;
    const checkpoint = tenzor.nn.checkpoint;

    // Build file paths
    var train_images_buf: [256]u8 = undefined;
    var train_labels_buf: [256]u8 = undefined;
    var test_images_buf: [256]u8 = undefined;
    var test_labels_buf: [256]u8 = undefined;

    const train_images = std.fmt.bufPrint(&train_images_buf, "{s}/train-images-idx3-ubyte", .{args.data_dir}) catch return error.PathTooLong;
    const train_labels = std.fmt.bufPrint(&train_labels_buf, "{s}/train-labels-idx1-ubyte", .{args.data_dir}) catch return error.PathTooLong;
    const test_images = std.fmt.bufPrint(&test_images_buf, "{s}/t10k-images-idx3-ubyte", .{args.data_dir}) catch return error.PathTooLong;
    const test_labels = std.fmt.bufPrint(&test_labels_buf, "{s}/t10k-labels-idx1-ubyte", .{args.data_dir}) catch return error.PathTooLong;

    // Load data (before initializing trainer so we print loading messages)
    if (!args.use_tui) {
        std.debug.print("LeNet-5 MNIST Training\n", .{});
        std.debug.print("======================\n", .{});
        std.debug.print("Data directory: {s}\n", .{args.data_dir});
        std.debug.print("Epochs: {d}\n", .{args.epochs});
        std.debug.print("Batch size: {d}\n", .{args.batch_size});
        std.debug.print("Learning rate: {d:.4}\n", .{args.learning_rate});
        std.debug.print("Scheduler: {s}\n", .{args.scheduler});
        std.debug.print("Seed: {d}\n\n", .{args.seed});
        std.debug.print("Loading training data...\n", .{});
    }

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

    if (!args.use_tui) {
        std.debug.print("  Loaded {d} training samples\n", .{train_data.num_samples});
        std.debug.print("Loading test data...\n", .{});
    }

    var test_data = mnist.MNISTDataset.load(allocator, test_images, test_labels) catch |err| {
        std.debug.print("Error loading test data: {}\n", .{err});
        return err;
    };
    defer test_data.deinit();

    if (!args.use_tui) {
        std.debug.print("  Loaded {d} test samples\n\n", .{test_data.num_samples});
    }

    // Parse scheduler type
    const scheduler_type: TrainerConfig.SchedulerType = if (std.mem.eql(u8, args.scheduler, "constant"))
        .constant
    else if (std.mem.eql(u8, args.scheduler, "step"))
        .step
    else if (std.mem.eql(u8, args.scheduler, "cosine"))
        .cosine
    else
        .cosine; // default

    // Calculate total batches for scheduler
    const num_batches = train_data.numBatches(args.batch_size);
    const total_steps = @as(u64, args.epochs) * num_batches;

    // Initialize trainer with TUI dashboard
    var trainer = try Trainer.init(allocator, .{
        .epochs = args.epochs,
        .batch_size = args.batch_size,
        .learning_rate = args.learning_rate,
        .scheduler = scheduler_type,
        .warmup_steps = args.warmup,
        .min_lr = 0.0001,
        .total_steps = total_steps,
        .checkpoint_dir = args.checkpoint_dir,
        .use_tui = args.use_tui,
        .early_stopping_patience = args.patience,
        .seed = args.seed,
        .model_name = "LeNet-5/MNIST",
    });
    defer trainer.deinit();

    // Initialize thread pool for parallel execution
    var pool = try threading.ThreadPool.create(allocator, .{}); // Use all available cores
    defer pool.destroy();

    // Initialize model with thread pool for parallel kernels
    const model_config = lenet.LeNetConfig{ .batch_size = args.batch_size };
    var model = try lenet.LeNet.init(allocator, model_config, pool);
    defer model.deinit();

    // Initialize weights
    var prng = std.Random.DefaultPrng.init(args.seed);
    model.weights.initKaiming(prng.random());

    // Training loop
    var epoch: u32 = 1;
    while (epoch <= args.epochs and trainer.shouldContinue()) : (epoch += 1) {
        trainer.beginEpoch(epoch, @intCast(num_batches));

        // Shuffle training data
        train_data.shuffle(prng.random());

        var epoch_loss: f32 = 0;
        var epoch_correct: usize = 0;
        var epoch_total: usize = 0;
        const epoch_start = std.time.Instant.now() catch null;

        for (0..num_batches) |batch_idx| {
            const batch_start = std.time.Instant.now() catch null;

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
            try model.backward(batch.images, actual_batch);

            // SGD update with learning rate from scheduler
            const lr = trainer.getLR();
            sgdUpdate(&model, lr);

            // Record batch metrics
            const batch_time_ms: f32 = if (batch_start) |bs| blk: {
                const batch_end = std.time.Instant.now() catch bs;
                break :blk @as(f32, @floatFromInt(batch_end.since(bs))) / std.time.ns_per_ms;
            } else 50.0;

            const running_acc = @as(f32, @floatFromInt(epoch_correct)) / @as(f32, @floatFromInt(epoch_total));
            try trainer.recordBatch(metrics.loss, running_acc, batch_time_ms);

            // Check if user wants to quit
            if (!trainer.shouldContinue()) break;
        }

        // Evaluate on test set
        var test_correct: usize = 0;
        var test_total: usize = 0;
        var test_loss: f32 = 0;
        const test_batches = test_data.numBatches(args.batch_size);

        for (0..test_batches) |batch_idx| {
            const batch = test_data.getBatch(batch_idx, args.batch_size);
            const actual_batch = batch.labels.len;

            _ = model.forward(batch.images, actual_batch);
            const metrics = model.computeLoss(batch.labels, actual_batch);
            test_loss += metrics.loss;
            test_correct += @intFromFloat(metrics.accuracy * @as(f32, @floatFromInt(actual_batch)));
            test_total += actual_batch;
        }

        const train_acc = @as(f32, @floatFromInt(epoch_correct)) / @as(f32, @floatFromInt(epoch_total));
        const val_acc = @as(f32, @floatFromInt(test_correct)) / @as(f32, @floatFromInt(test_total));
        const val_loss = test_loss / @as(f32, @floatFromInt(test_batches));

        const epoch_time_sec: f32 = if (epoch_start) |es| blk: {
            const epoch_end = std.time.Instant.now() catch es;
            break :blk @as(f32, @floatFromInt(epoch_end.since(es))) / std.time.ns_per_s;
        } else 0;

        try trainer.endEpoch(val_loss, val_acc, epoch_time_sec);

        // Save checkpoint if this is a new best and checkpoint_dir is set
        if (args.checkpoint_dir != null and trainer.state.best_epoch == epoch) {
            const ckpt_dir = args.checkpoint_dir.?;

            // Create checkpoint directory if it doesn't exist
            std.fs.cwd().makePath(ckpt_dir) catch {};

            // Build checkpoint path
            var path_buf: [512]u8 = undefined;
            const ckpt_path = std.fmt.bufPrint(&path_buf, "{s}/best.tenzor", .{ckpt_dir}) catch continue;

            // Save checkpoint using CheckpointWriter
            var writer = checkpoint.CheckpointWriter.init(allocator, ckpt_path) catch |err| {
                try trainer.log("Failed to create checkpoint");
                std.debug.print("Checkpoint error: {}\n", .{err});
                continue;
            };
            defer writer.deinit();

            // Add all weights with their shapes
            writer.addWeight("conv1_weight", model.weights.conv1_weight, &[_]usize{ 6, 5, 5, 1 }) catch {};
            writer.addWeight("conv1_bias", model.weights.conv1_bias, &[_]usize{6}) catch {};
            writer.addWeight("conv2_weight", model.weights.conv2_weight, &[_]usize{ 16, 5, 5, 6 }) catch {};
            writer.addWeight("conv2_bias", model.weights.conv2_bias, &[_]usize{16}) catch {};
            writer.addWeight("fc1_weight", model.weights.fc1_weight, &[_]usize{ 120, 256 }) catch {};
            writer.addWeight("fc1_bias", model.weights.fc1_bias, &[_]usize{120}) catch {};
            writer.addWeight("fc2_weight", model.weights.fc2_weight, &[_]usize{ 84, 120 }) catch {};
            writer.addWeight("fc2_bias", model.weights.fc2_bias, &[_]usize{84}) catch {};
            writer.addWeight("fc3_weight", model.weights.fc3_weight, &[_]usize{ 10, 84 }) catch {};
            writer.addWeight("fc3_bias", model.weights.fc3_bias, &[_]usize{10}) catch {};

            // Add metadata
            writer.setMetadata(.{
                .epoch = epoch,
                .global_step = trainer.state.global_step,
                .best_val_loss = trainer.state.best_val_loss,
                .best_val_acc = trainer.state.best_val_acc,
                .best_epoch = trainer.state.best_epoch,
                .learning_rate = trainer.state.lr,
                .model_name = "lenet",
                .timestamp = 0, // TODO: get real timestamp when Zig API stabilizes
            }) catch {};

            writer.finish() catch |err| {
                try trainer.log("Failed to save checkpoint");
                std.debug.print("Checkpoint write error: {}\n", .{err});
                continue;
            };

            try trainer.log("Saved best model checkpoint");
        }

        // Print progress if no TUI
        if (!args.use_tui) {
            std.debug.print("Epoch {d:2}/{d}: loss={d:.4}, train_acc={d:.1}%, val_acc={d:.1}%\n", .{
                epoch,
                args.epochs,
                epoch_loss / @as(f32, @floatFromInt(num_batches)),
                train_acc * 100,
                val_acc * 100,
            });
        }
    }

    // Print final stats
    const stats = trainer.getStats();
    if (!args.use_tui) {
        std.debug.print("\nTraining complete!\n", .{});
    } else {
        std.debug.print("\n\nTraining complete!\n", .{});
    }
    std.debug.print("  Best val_acc: {d:.1}% (epoch {d})\n", .{ stats.best_val_acc * 100, stats.best_epoch });
    std.debug.print("  Total time: {d:.1}s\n", .{stats.total_time_sec});
    if (stats.stopped_early) {
        std.debug.print("  Stopped early due to early stopping\n", .{});
    }
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
// Eval Command
// ============================================================================

const EvalArgs = struct {
    checkpoint_path: []const u8,
    data_dir: []const u8,
    batch_size: u32,
};

fn runEval(allocator: std.mem.Allocator, args: EvalArgs) !void {
    const lenet = tenzor.model.lenet;
    const mnist = tenzor.io.mnist;
    const checkpoint = tenzor.nn.checkpoint;
    const threading = tenzor.backend.cpu.threading;

    std.debug.print("LeNet-5 Evaluation\n", .{});
    std.debug.print("==================\n", .{});
    std.debug.print("Checkpoint: {s}\n", .{args.checkpoint_path});
    std.debug.print("Data directory: {s}\n", .{args.data_dir});
    std.debug.print("Batch size: {d}\n\n", .{args.batch_size});

    // Load checkpoint
    std.debug.print("Loading checkpoint...\n", .{});
    var ckpt = checkpoint.Checkpoint.load(allocator, args.checkpoint_path) catch |err| {
        std.debug.print("Error loading checkpoint: {}\n", .{err});
        return err;
    };
    defer ckpt.deinit();

    std.debug.print("  Loaded {d} tensors\n", .{ckpt.tensorCount()});

    // Load test data
    var test_images_buf: [256]u8 = undefined;
    var test_labels_buf: [256]u8 = undefined;
    const test_images = std.fmt.bufPrint(&test_images_buf, "{s}/t10k-images-idx3-ubyte", .{args.data_dir}) catch return error.PathTooLong;
    const test_labels = std.fmt.bufPrint(&test_labels_buf, "{s}/t10k-labels-idx1-ubyte", .{args.data_dir}) catch return error.PathTooLong;

    std.debug.print("Loading test data...\n", .{});
    var test_data = mnist.MNISTDataset.load(allocator, test_images, test_labels) catch |err| {
        std.debug.print("Error loading test data: {}\n", .{err});
        std.debug.print("Make sure MNIST files exist in {s}/\n", .{args.data_dir});
        return err;
    };
    defer test_data.deinit();

    std.debug.print("  Loaded {d} test samples\n\n", .{test_data.num_samples});

    // Initialize model
    var pool = try threading.ThreadPool.create(allocator, .{});
    defer pool.destroy();

    const model_config = lenet.LeNetConfig{ .batch_size = args.batch_size };
    var model = try lenet.LeNet.init(allocator, model_config, pool);
    defer model.deinit();

    // Load weights from checkpoint into model
    std.debug.print("Loading weights from checkpoint...\n", .{});

    if (ckpt.getTensor("conv1_weight")) |data| {
        @memcpy(model.weights.conv1_weight, data);
    } else {
        std.debug.print("  Warning: conv1_weight not found in checkpoint\n", .{});
    }
    if (ckpt.getTensor("conv1_bias")) |data| {
        @memcpy(model.weights.conv1_bias, data);
    }
    if (ckpt.getTensor("conv2_weight")) |data| {
        @memcpy(model.weights.conv2_weight, data);
    }
    if (ckpt.getTensor("conv2_bias")) |data| {
        @memcpy(model.weights.conv2_bias, data);
    }
    if (ckpt.getTensor("fc1_weight")) |data| {
        @memcpy(model.weights.fc1_weight, data);
    }
    if (ckpt.getTensor("fc1_bias")) |data| {
        @memcpy(model.weights.fc1_bias, data);
    }
    if (ckpt.getTensor("fc2_weight")) |data| {
        @memcpy(model.weights.fc2_weight, data);
    }
    if (ckpt.getTensor("fc2_bias")) |data| {
        @memcpy(model.weights.fc2_bias, data);
    }
    if (ckpt.getTensor("fc3_weight")) |data| {
        @memcpy(model.weights.fc3_weight, data);
    }
    if (ckpt.getTensor("fc3_bias")) |data| {
        @memcpy(model.weights.fc3_bias, data);
    }

    std.debug.print("  Weights loaded successfully\n\n", .{});

    // Evaluate on test set
    std.debug.print("Evaluating...\n", .{});
    const start_time = std.time.Instant.now() catch null;

    var test_correct: usize = 0;
    var test_total: usize = 0;
    var test_loss: f32 = 0;
    const test_batches = test_data.numBatches(args.batch_size);

    for (0..test_batches) |batch_idx| {
        const batch = test_data.getBatch(batch_idx, args.batch_size);
        const actual_batch = batch.labels.len;

        _ = model.forward(batch.images, actual_batch);
        const metrics = model.computeLoss(batch.labels, actual_batch);
        test_loss += metrics.loss;
        test_correct += @intFromFloat(metrics.accuracy * @as(f32, @floatFromInt(actual_batch)));
        test_total += actual_batch;
    }

    const elapsed_sec: f32 = if (start_time) |st| blk: {
        const end = std.time.Instant.now() catch st;
        break :blk @as(f32, @floatFromInt(end.since(st))) / std.time.ns_per_s;
    } else 0;

    const accuracy = @as(f32, @floatFromInt(test_correct)) / @as(f32, @floatFromInt(test_total));
    const avg_loss = test_loss / @as(f32, @floatFromInt(test_batches));

    std.debug.print("\nResults:\n", .{});
    std.debug.print("  Test accuracy: {d:.2}%\n", .{accuracy * 100});
    std.debug.print("  Test loss: {d:.4}\n", .{avg_loss});
    std.debug.print("  Samples evaluated: {d}\n", .{test_total});
    std.debug.print("  Evaluation time: {d:.2}s\n", .{elapsed_sec});
}

// ============================================================================
// Embed Command
// ============================================================================

const EmbedArgs = struct {
    model_path: []const u8,
    texts: []const []const u8,
    input_path: ?[]const u8,
};

fn runEmbed(allocator: std.mem.Allocator, args: EmbedArgs) !void {
    const arctic = tenzor.model.arctic;
    const safetensors = tenzor.io.safetensors;
    const Tokenizer = tenzor.io.tokenizer.Tokenizer;
    const ThreadPool = tenzor.backend.cpu.threading.ThreadPool;

    std.debug.print("Arctic-embed-xs Text Embedding\n", .{});
    std.debug.print("==============================\n", .{});
    std.debug.print("Model: {s}\n", .{args.model_path});

    // Load text from file if --input provided
    var file_text: ?[]u8 = null;
    defer if (file_text) |t| allocator.free(t);

    if (args.input_path) |input_path| {
        std.debug.print("Input: {s}\n", .{input_path});
        const file = std.fs.cwd().openFile(input_path, .{}) catch |err| {
            std.debug.print("Error opening input file: {}\n", .{err});
            return err;
        };
        defer file.close();

        const stat = try file.stat();
        file_text = try allocator.alloc(u8, stat.size);

        var total_read: usize = 0;
        while (total_read < stat.size) {
            const n = try file.read(file_text.?[total_read..]);
            if (n == 0) break;
            total_read += n;
        }
        file_text = file_text.?[0..total_read];
    } else {
        std.debug.print("Texts: {d}\n", .{args.texts.len});
    }
    std.debug.print("\n", .{});

    // Derive vocab path from model path (same directory)
    var vocab_path_buf: [512]u8 = undefined;
    const model_dir = std.fs.path.dirname(args.model_path) orelse ".";
    const vocab_path = std.fmt.bufPrint(&vocab_path_buf, "{s}/vocab.txt", .{model_dir}) catch return error.PathTooLong;

    // Load tokenizer
    std.debug.print("Loading tokenizer...\n", .{});
    var tokenizer = Tokenizer.init(allocator);
    defer tokenizer.deinit();
    tokenizer.loadVocab(vocab_path) catch |err| {
        std.debug.print("Error loading vocab.txt: {}\n", .{err});
        std.debug.print("Expected at: {s}\n", .{vocab_path});
        return err;
    };

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

    const max_seq_len: usize = 512;
    const chunk_size: usize = 510; // Leave room for [CLS] and [SEP]
    const chunk_overlap: usize = 50;

    // Create thread pool for parallel execution
    const cpu_count: u32 = @intCast(std.Thread.getCpuCount() catch 4);
    var pool = try ThreadPool.create(allocator, .{ .thread_count = cpu_count });
    defer pool.destroy();

    std.debug.print("Generating embeddings ({d} threads)...\n\n", .{cpu_count});

    // Process either file text or positional args
    if (file_text) |text| {
        // Tokenize entire document (without truncation for chunking)
        var all_tokens: std.ArrayList(u32) = .empty;
        defer all_tokens.deinit(allocator);

        // Tokenize without [CLS]/[SEP] for chunking
        try tokenizer.encodeRaw(text, &all_tokens, allocator);
        const total_tokens = all_tokens.items.len;

        // Calculate number of chunks needed
        const num_chunks = if (total_tokens <= chunk_size)
            1
        else
            1 + (total_tokens - chunk_size + chunk_overlap - 1) / (chunk_size - chunk_overlap);

        std.debug.print("Total tokens: {d}, Chunks: {d} (size={d}, overlap={d})\n", .{
            total_tokens, num_chunks, chunk_size, chunk_overlap,
        });

        // Allocate embeddings for all chunks
        const chunk_embeddings = try allocator.alloc([384]f32, num_chunks);
        defer allocator.free(chunk_embeddings);

        // Parallel embedding - limit workers to avoid cache thrashing
        // Each InferenceContext uses ~5.5MB, so cap at 8 workers (~44MB working set)
        const max_workers: usize = 8;
        const num_workers: usize = @min(@min(cpu_count, num_chunks), max_workers);

        if (num_workers > 1) {
            std.debug.print("Using {d} worker threads...\n", .{num_workers});

            // One context per worker - no sharing, no contention
            // Don't pass pool to contexts - avoid nested parallelism overhead
            const contexts = try allocator.alloc(arctic.InferenceContext, num_workers);
            defer allocator.free(contexts);
            for (contexts) |*ctx| {
                ctx.* = try arctic.InferenceContext.init(allocator, config, max_seq_len);
            }
            defer for (contexts) |*ctx| ctx.deinit();

            const WorkCtx = struct {
                all_tokens: []const u32,
                chunk_embeddings: [][384]f32,
                weights: arctic.ModelWeights,
                contexts: []arctic.InferenceContext,
                num_chunks: usize,
                num_workers: usize,
                chunk_size: usize,
                chunk_overlap: usize,
                cls_id: u32,
                sep_id: u32,
            };

            const work_ctx = WorkCtx{
                .all_tokens = all_tokens.items,
                .chunk_embeddings = chunk_embeddings,
                .weights = weights,
                .contexts = contexts,
                .num_chunks = num_chunks,
                .num_workers = num_workers,
                .chunk_size = chunk_size,
                .chunk_overlap = chunk_overlap,
                .cls_id = tokenizer.cls_id,
                .sep_id = tokenizer.sep_id,
            };

            // Create a smaller pool sized for our actual worker count
            var chunk_pool = try ThreadPool.create(allocator, .{ .thread_count = @intCast(num_workers) });
            defer chunk_pool.destroy();

            // Compute chunks per worker for context assignment
            const chunks_per_worker = (num_chunks + num_workers - 1) / num_workers;

            // Parallelize over chunks using parallelForBatch (bypasses MIN_PARALLEL_SIZE threshold)
            chunk_pool.parallelForBatch(num_chunks, work_ctx, struct {
                fn work(ctx: WorkCtx, start: usize, end: usize) void {
                    // Determine which context to use based on chunk range
                    const chunks_per_ctx = (ctx.num_chunks + ctx.num_workers - 1) / ctx.num_workers;
                    const worker_id = start / chunks_per_ctx;
                    const ctx_idx = worker_id % ctx.num_workers;
                    const inf_ctx = &ctx.contexts[ctx_idx];

                    for (start..end) |chunk_idx| {
                        const stride = ctx.chunk_size - ctx.chunk_overlap;
                        const token_start = chunk_idx * stride;
                        const token_end = @min(token_start + ctx.chunk_size, ctx.all_tokens.len);

                        var chunk_tokens: [512]u32 = undefined;
                        chunk_tokens[0] = ctx.cls_id;
                        const content_len = token_end - token_start;
                        @memcpy(chunk_tokens[1 .. 1 + content_len], ctx.all_tokens[token_start..token_end]);
                        chunk_tokens[1 + content_len] = ctx.sep_id;

                        arctic.forward(&ctx.chunk_embeddings[chunk_idx], chunk_tokens[0 .. content_len + 2], ctx.weights, inf_ctx);
                    }
                }
            }.work);
            _ = chunks_per_worker;
        } else {
            // Single chunk - still use parallel internal ops
            var ctx = try arctic.InferenceContext.initWithPool(allocator, config, max_seq_len, pool);
            defer ctx.deinit();

            var chunk_tokens: [512]u32 = undefined;
            chunk_tokens[0] = tokenizer.cls_id;
            @memcpy(chunk_tokens[1 .. 1 + total_tokens], all_tokens.items);
            chunk_tokens[1 + total_tokens] = tokenizer.sep_id;

            arctic.forward(&chunk_embeddings[0], chunk_tokens[0 .. total_tokens + 2], weights, &ctx);
        }

        // Mean pool across all chunk embeddings
        var final_embedding: [384]f32 = [_]f32{0.0} ** 384;
        for (chunk_embeddings) |chunk_emb| {
            for (&final_embedding, chunk_emb) |*f, c| {
                f.* += c;
            }
        }
        const scale = 1.0 / @as(f32, @floatFromInt(num_chunks));
        for (&final_embedding) |*f| {
            f.* *= scale;
        }

        // L2 normalize the final embedding
        var norm: f32 = 0.0;
        for (final_embedding) |v| norm += v * v;
        norm = @sqrt(norm);
        if (norm > 0) {
            for (&final_embedding) |*f| f.* /= norm;
        }

        // Output embedding
        std.debug.print("Embedding: [", .{});
        for (final_embedding, 0..) |v, j| {
            if (j > 0) std.debug.print(", ", .{});
            if (j % 10 == 0 and j > 0) std.debug.print("\n  ", .{});
            std.debug.print("{d:.6}", .{v});
        }
        std.debug.print("]\n", .{});
    } else {
        // Multiple texts from args - use parallel batch processing
        const num_texts = args.texts.len;

        if (num_texts > 1 and cpu_count > 1) {
            // Tokenize all texts first
            const token_ids_storage = try allocator.alloc([512]u32, num_texts);
            defer allocator.free(token_ids_storage);

            const seq_lens = try allocator.alloc(usize, num_texts);
            defer allocator.free(seq_lens);

            for (args.texts, 0..) |text, i| {
                seq_lens[i] = tokenizer.encode(text, &token_ids_storage[i]) catch |err| {
                    std.debug.print("Error tokenizing \"{s}\": {}\n", .{ text, err });
                    return err;
                };
            }

            // Create batch context pool for parallel processing
            const num_contexts = @min(cpu_count, num_texts);
            var ctx_pool = try arctic.BatchContextPool.init(allocator, config, max_seq_len, num_contexts, pool);
            defer ctx_pool.deinit();

            // Allocate output embeddings
            const embeddings = try allocator.alloc([384]f32, num_texts);
            defer allocator.free(embeddings);

            // Build slice array for forwardBatchParallel
            const token_slices = try allocator.alloc([]const u32, num_texts);
            defer allocator.free(token_slices);
            for (0..num_texts) |i| {
                token_slices[i] = token_ids_storage[i][0..seq_lens[i]];
            }

            // Process all texts in parallel
            arctic.forwardBatchParallel(
                @as([*]f32, @ptrCast(embeddings.ptr))[0 .. num_texts * 384],
                token_slices,
                weights,
                &ctx_pool,
                pool,
            );

            // Output results
            std.debug.print("[\n", .{});
            for (args.texts, 0..) |text, i| {
                std.debug.print("  {{\n    \"text\": \"{s}\",\n    \"tokens\": {d},\n    \"embedding\": [", .{ text, seq_lens[i] });
                for (embeddings[i], 0..) |v, j| {
                    if (j > 0) std.debug.print(", ", .{});
                    if (j % 10 == 0 and j > 0) std.debug.print("\n      ", .{});
                    std.debug.print("{d:.6}", .{v});
                }
                std.debug.print("]\n  }}", .{});
                if (i < args.texts.len - 1) std.debug.print(",", .{});
                std.debug.print("\n", .{});
            }
            std.debug.print("]\n", .{});
        } else {
            // Single text or single thread - use simple path with parallel internal ops
            var ctx = try arctic.InferenceContext.initWithPool(allocator, config, max_seq_len, pool);
            defer ctx.deinit();

            std.debug.print("[\n", .{});
            for (args.texts, 0..) |text, i| {
                var token_ids: [512]u32 = undefined;
                const seq_len = tokenizer.encode(text, &token_ids) catch |err| {
                    std.debug.print("Error tokenizing \"{s}\": {}\n", .{ text, err });
                    return err;
                };

                var embedding: [384]f32 = undefined;
                arctic.forward(&embedding, token_ids[0..seq_len], weights, &ctx);

                std.debug.print("  {{\n    \"text\": \"{s}\",\n    \"tokens\": {d},\n    \"embedding\": [", .{ text, seq_len });
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
    }
}

// ============================================================================
// Convert Command
// ============================================================================

fn runConvert(allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8) !void {
    const tenzor_format = tenzor.io.tenzor_format;

    std.debug.print("Converting SafeTensors to .tenzor\n", .{});
    std.debug.print("=================================\n", .{});
    std.debug.print("Input:  {s}\n", .{input_path});
    std.debug.print("Output: {s}\n\n", .{output_path});

    const start_time = std.time.Instant.now() catch null;

    tenzor_format.convertFromSafetensors(allocator, input_path, output_path) catch |err| {
        std.debug.print("Error converting file: {}\n", .{err});
        return err;
    };

    const elapsed_ns: u64 = if (start_time) |start| blk: {
        const end = std.time.Instant.now() catch break :blk 0;
        break :blk end.since(start);
    } else 0;
    const elapsed_ms = elapsed_ns / std.time.ns_per_ms;

    // Get file sizes
    const input_stat = try std.fs.cwd().statFile(input_path);
    const output_stat = try std.fs.cwd().statFile(output_path);

    std.debug.print("Conversion complete!\n", .{});
    std.debug.print("  Input size:  {d:.2} MB\n", .{@as(f64, @floatFromInt(input_stat.size)) / (1024 * 1024)});
    std.debug.print("  Output size: {d:.2} MB\n", .{@as(f64, @floatFromInt(output_stat.size)) / (1024 * 1024)});
    std.debug.print("  Time: {d} ms\n", .{elapsed_ms});
}

// ============================================================================
// Download Command
// ============================================================================

fn runDownload(allocator: std.mem.Allocator, model_id: []const u8, output_path: ?[]const u8) !void {
    const HuggingFace = tenzor.io.huggingface.HuggingFace;

    std.debug.print("HuggingFace Model Download\n", .{});
    std.debug.print("==========================\n", .{});
    std.debug.print("Model: {s}\n\n", .{model_id});

    var hf = HuggingFace.init(allocator, null);
    defer hf.deinit();

    const start_time = std.time.Instant.now() catch null;

    const tenzor_path = hf.downloadAndConvert(model_id, output_path) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        return err;
    };
    defer allocator.free(tenzor_path);

    const elapsed_ns: u64 = if (start_time) |start| blk: {
        const end = std.time.Instant.now() catch break :blk 0;
        break :blk end.since(start);
    } else 0;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    // Get file size
    const stat = try std.fs.cwd().statFile(tenzor_path);

    std.debug.print("\nDownload complete!\n", .{});
    std.debug.print("  Output: {s}\n", .{tenzor_path});
    std.debug.print("  Size:   {d:.2} MB\n", .{@as(f64, @floatFromInt(stat.size)) / (1024 * 1024)});
    std.debug.print("  Time:   {d:.1} s\n", .{elapsed_s});
}

// ============================================================================
// Info Command
// ============================================================================

fn runInfo(allocator: std.mem.Allocator, path: []const u8) !void {
    const tenzor_format = tenzor.io.tenzor_format;

    std.debug.print(".tenzor File Info\n", .{});
    std.debug.print("=================\n", .{});
    std.debug.print("Path: {s}\n\n", .{path});

    var file = tenzor_format.TenzorFile.open(allocator, path) catch |err| {
        std.debug.print("Error opening file: {}\n", .{err});
        return err;
    };
    defer file.close();

    // File stats
    const stat = try std.fs.cwd().statFile(path);

    std.debug.print("Header:\n", .{});
    std.debug.print("  Version:      {d}\n", .{file.header.version});
    std.debug.print("  Tensors:      {d}\n", .{file.header.tensor_count});
    std.debug.print("  Index offset: 0x{x}\n", .{file.header.index_offset});
    std.debug.print("  Data offset:  0x{x}\n", .{file.header.data_offset});
    std.debug.print("  File size:    {d:.2} MB\n\n", .{@as(f64, @floatFromInt(stat.size)) / (1024 * 1024)});

    if (file.metadata_json.len > 2) {
        std.debug.print("Metadata: {s}\n\n", .{file.metadata_json});
    }

    std.debug.print("Tensors:\n", .{});
    var total_params: usize = 0;
    for (file.index, 0..) |entry, i| {
        const numel = entry.numel();
        total_params += numel;

        std.debug.print("  [{d:3}] hash=0x{x:016} dtype={s} shape=[", .{ i, entry.name_hash, entry.dtype.toString() });
        for (0..entry.ndim) |j| {
            if (j > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{entry.shape[j]});
        }
        std.debug.print("] numel={d}\n", .{numel});
    }

    std.debug.print("\nTotal parameters: {d} ({d:.2} M)\n", .{ total_params, @as(f64, @floatFromInt(total_params)) / 1_000_000 });
}

// ============================================================================
// ONNX Commands
// ============================================================================

fn runOnnxInfo(allocator: std.mem.Allocator, path: []const u8) !void {
    const onnx = tenzor.onnx;

    std.debug.print("ONNX Model Info\n", .{});
    std.debug.print("===============\n", .{});
    std.debug.print("Path: {s}\n\n", .{path});

    // Read file
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("Error opening file: {}\n", .{err});
        return err;
    };
    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);

    var total_read: usize = 0;
    while (total_read < stat.size) {
        const n = try file.read(data[total_read..]);
        if (n == 0) break;
        total_read += n;
    }

    std.debug.print("File size: {d:.2} KB\n\n", .{@as(f64, @floatFromInt(stat.size)) / 1024});

    // Parse model
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const model = onnx.parser.parseModel(arena.allocator(), data[0..total_read]) catch |err| {
        std.debug.print("Error parsing ONNX: {}\n", .{err});
        return err;
    };

    // Print model info
    std.debug.print("IR version: {}\n", .{model.ir_version});
    if (model.producer_name) |name| {
        std.debug.print("Producer: {s}\n", .{name});
    }
    if (model.producer_version) |ver| {
        std.debug.print("Producer version: {s}\n", .{ver});
    }

    // Opset imports
    std.debug.print("\nOpset imports: {}\n", .{model.opset_import.len});
    for (model.opset_import) |opset| {
        if (opset.domain) |domain| {
            if (domain.len > 0) {
                std.debug.print("  {s}: v{}\n", .{ domain, opset.version });
            } else {
                std.debug.print("  (default): v{}\n", .{opset.version});
            }
        } else {
            std.debug.print("  (default): v{}\n", .{opset.version});
        }
    }

    // Graph info
    if (model.graph) |graph| {
        if (graph.name) |name| {
            std.debug.print("\nGraph: {s}\n", .{name});
        } else {
            std.debug.print("\nGraph:\n", .{});
        }

        std.debug.print("  Inputs:  {}\n", .{graph.input.len});
        for (graph.input) |input| {
            if (input.name) |name| {
                std.debug.print("    - {s}\n", .{name});
            }
        }

        std.debug.print("  Outputs: {}\n", .{graph.output.len});
        for (graph.output) |output| {
            if (output.name) |name| {
                std.debug.print("    - {s}\n", .{name});
            }
        }

        std.debug.print("  Nodes:   {}\n", .{graph.node.len});

        // Count ops
        var op_counts = std.StringHashMap(u32).init(allocator);
        defer op_counts.deinit();

        for (graph.node) |node| {
            if (node.op_type) |op| {
                const entry = op_counts.getOrPut(op) catch continue;
                if (!entry.found_existing) {
                    entry.value_ptr.* = 0;
                }
                entry.value_ptr.* += 1;
            }
        }

        std.debug.print("  Op distribution:\n", .{});
        var it = op_counts.iterator();
        while (it.next()) |entry| {
            std.debug.print("    {s}: {}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        // Initializers (weights)
        std.debug.print("  Initializers: {}\n", .{graph.initializer.len});

        var external_count: usize = 0;
        var inline_count: usize = 0;
        var total_bytes: usize = 0;

        for (graph.initializer) |tensor| {
            if (tensor.isExternal()) {
                external_count += 1;
            } else {
                inline_count += 1;
                if (tensor.raw_data) |rd| {
                    total_bytes += rd.len;
                }
            }
        }

        std.debug.print("    Inline:   {} ({d:.2} KB)\n", .{ inline_count, @as(f64, @floatFromInt(total_bytes)) / 1024 });
        std.debug.print("    External: {}\n", .{external_count});

        if (external_count > 0) {
            std.debug.print("  External data files:\n", .{});
            var seen = std.StringHashMap(void).init(allocator);
            defer seen.deinit();

            for (graph.initializer) |tensor| {
                if (tensor.getExternalLocation()) |loc| {
                    if (!seen.contains(loc)) {
                        seen.put(loc, {}) catch continue;
                        std.debug.print("    - {s}\n", .{loc});
                    }
                }
            }
        }
    }
}

fn runOnnxDownload(allocator: std.mem.Allocator, model_id: []const u8, onnx_file: []const u8) !void {
    const HuggingFace = tenzor.io.huggingface.HuggingFace;

    std.debug.print("ONNX Model Download\n", .{});
    std.debug.print("===================\n", .{});
    std.debug.print("Model: {s}\n", .{model_id});
    std.debug.print("File:  {s}\n\n", .{onnx_file});

    var hf = HuggingFace.init(allocator, null);
    defer hf.deinit();

    const start_time = std.time.Instant.now() catch null;

    var info = hf.downloadOnnxModelAuto(model_id, onnx_file) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        return err;
    };
    defer info.deinit();

    const elapsed_ns: u64 = if (start_time) |start| blk: {
        const end = std.time.Instant.now() catch break :blk 0;
        break :blk end.since(start);
    } else 0;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;

    // Get file sizes
    const stat = try std.fs.cwd().statFile(info.model_path);

    std.debug.print("\nDownload complete!\n", .{});
    std.debug.print("  Model:    {s}\n", .{info.model_path});
    std.debug.print("  Size:     {d:.2} KB\n", .{@as(f64, @floatFromInt(stat.size)) / 1024});
    std.debug.print("  External: {} files\n", .{info.external_files.len});
    for (info.external_files) |f| {
        const ext_stat = std.fs.cwd().statFile(f) catch continue;
        std.debug.print("    - {s} ({d:.2} MB)\n", .{
            std.fs.path.basename(f),
            @as(f64, @floatFromInt(ext_stat.size)) / (1024 * 1024),
        });
    }
    std.debug.print("  Time:     {d:.1} s\n", .{elapsed_s});
}

fn runOnnxRun(allocator: std.mem.Allocator, path: []const u8) !void {
    const onnx = tenzor.onnx;

    std.debug.print("ONNX Model Run\n", .{});
    std.debug.print("==============\n", .{});
    std.debug.print("Path: {s}\n\n", .{path});

    // Read file
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("Error opening file: {}\n", .{err});
        return err;
    };
    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);

    var total_read: usize = 0;
    while (total_read < stat.size) {
        const n = try file.read(data[total_read..]);
        if (n == 0) break;
        total_read += n;
    }

    // Build runtime graph
    std.debug.print("Building graph...\n", .{});
    var graph = onnx.buildFromBytes(allocator, data[0..total_read]) catch |err| {
        std.debug.print("Error building graph: {}\n", .{err});
        return err;
    };
    defer graph.deinit();

    std.debug.print("  Nodes:   {}\n", .{graph.nodes.items.len});
    std.debug.print("  Tensors: {}\n", .{graph.tensors.items.len});
    std.debug.print("  Inputs:  {}\n", .{graph.inputs.len});
    std.debug.print("  Outputs: {}\n", .{graph.outputs.len});

    // Create executor
    std.debug.print("\nInitializing executor...\n", .{});
    var exec = onnx.Executor.init(allocator, &graph) catch |err| {
        std.debug.print("Error creating executor: {}\n", .{err});
        return err;
    };
    defer exec.deinit();

    // Load weights
    std.debug.print("Loading weights...\n", .{});
    try exec.loadWeights();

    // Try loading external weights
    const model_dir = std.fs.path.dirname(path) orelse ".";
    exec.loadExternalWeights(model_dir) catch |err| {
        std.debug.print("  External weights: {} (skipped)\n", .{err});
    };

    // Print input requirements
    std.debug.print("\nModel requires {} input(s):\n", .{graph.inputs.len});
    for (graph.inputs) |input_idx| {
        const info = graph.tensors.items[input_idx];
        std.debug.print("  - {s}: {s} [", .{ info.name, @tagName(info.dtype) });
        for (info.shape, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            if (dim < 0) {
                std.debug.print("?", .{});
            } else {
                std.debug.print("{}", .{dim});
            }
        }
        std.debug.print("]\n", .{});
    }

    // Run inference with test input based on what the model expects
    std.debug.print("\nRunning inference with test data...\n", .{});

    // Detect input type and provide appropriate test data
    const first_input_idx = graph.inputs[0];
    const first_input = graph.tensors.items[first_input_idx];

    if (std.mem.eql(u8, first_input.name, "input_ids")) {
        // Token-based model (embed_tokens)
        const test_tokens = [_]i64{ 1, 2, 3, 4, 5 };
        const input_shape = [_]i64{ 1, 5 };
        exec.setInputFromSlice("input_ids", i64, &test_tokens, &input_shape) catch |err| {
            std.debug.print("Error setting input_ids: {}\n", .{err});
            return err;
        };
        std.debug.print("  input_ids: [1, 2, 3, 4, 5] shape=[1, 5]\n", .{});
    } else if (std.mem.eql(u8, first_input.name, "inputs_embeds")) {
        // Embedding-based model (language_model)
        // Create dummy embeddings: [batch=1, seq=1, dim=1024]
        var test_embeds: [1024]f32 = undefined;
        for (&test_embeds, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) * 0.001;
        }
        const embed_shape = [_]i64{ 1, 1, 1024 };
        exec.setInputFromSlice("inputs_embeds", f32, &test_embeds, &embed_shape) catch |err| {
            std.debug.print("Error setting inputs_embeds: {}\n", .{err});
            return err;
        };
        std.debug.print("  inputs_embeds: shape=[1, 1, 1024]\n", .{});

        // Set attention_mask
        const attn_mask = [_]i64{1};
        const mask_shape = [_]i64{ 1, 1 };
        exec.setInputFromSlice("attention_mask", i64, &attn_mask, &mask_shape) catch {};

        // Set position_ids
        const pos_ids = [_]i64{0};
        exec.setInputFromSlice("position_ids", i64, &pos_ids, &mask_shape) catch {};

        // Set empty KV caches (24 layers) - use f16 to match model expectations
        var kv_cache_f16: [0]f16 = .{};
        const kv_shape = [_]i64{ 1, 16, 0, 64 }; // batch, heads, seq=0, head_dim
        for (0..24) |layer| {
            var key_buf: [64]u8 = undefined;
            var val_buf: [64]u8 = undefined;
            const key_name = std.fmt.bufPrint(&key_buf, "past_key_values.{}.key", .{layer}) catch continue;
            const val_name = std.fmt.bufPrint(&val_buf, "past_key_values.{}.value", .{layer}) catch continue;
            exec.setInputFromSlice(key_name, f16, &kv_cache_f16, &kv_shape) catch |err| {
                std.debug.print("Failed to set {s}: {}\n", .{key_name, err});
            };
            exec.setInputFromSlice(val_name, f16, &kv_cache_f16, &kv_shape) catch |err| {
                std.debug.print("Failed to set {s}: {}\n", .{val_name, err});
            };
        }
        std.debug.print("  Set all 51 inputs for language model\n", .{});
    } else if (std.mem.eql(u8, first_input.name, "audio_values")) {
        // Audio waveform input (speech_encoder)
        // Create dummy audio: [batch=1, samples=80000] (5 seconds at 16kHz - required for voice cloning)
        var test_audio: [80000]f32 = undefined;
        for (&test_audio, 0..) |*v, i| {
            // Simple sine wave at 440Hz
            const t: f32 = @as(f32, @floatFromInt(i)) / 16000.0;
            v.* = @sin(t * 440.0 * 2.0 * std.math.pi);
        }
        const audio_shape = [_]i64{ 1, 80000 };
        exec.setInputFromSlice("audio_values", f32, &test_audio, &audio_shape) catch |err| {
            std.debug.print("Error setting audio_values: {}\n", .{err});
            return err;
        };
        std.debug.print("  audio_values: shape=[1, 80000] (5sec @ 16kHz)\n", .{});
    } else if (std.mem.eql(u8, first_input.name, "prompt")) {
        // Conditional decoder input
        // prompt is the speech encoder output: [batch=1, seq=~50, dim=1024]
        var test_prompt: [50 * 1024]f32 = undefined;
        for (&test_prompt, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 1024)) * 0.001;
        }
        const prompt_shape = [_]i64{ 1, 50, 1024 };
        exec.setInputFromSlice("prompt", f32, &test_prompt, &prompt_shape) catch |err| {
            std.debug.print("Error setting prompt: {}\n", .{err});
            return err;
        };
        std.debug.print("  prompt: shape=[1, 50, 1024]\n", .{});
    } else {
        std.debug.print("Unknown input type: {s}\n", .{first_input.name});
        std.debug.print("Skipping inference.\n", .{});
        return;
    }

    // Execute
    const start_time = std.time.Instant.now() catch null;
    exec.run() catch |err| {
        std.debug.print("Error during inference: {}\n", .{err});
        return err;
    };
    const elapsed_ns: u64 = if (start_time) |start| blk: {
        const end = std.time.Instant.now() catch break :blk 0;
        break :blk end.since(start);
    } else 0;

    std.debug.print("  Inference complete in {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000});

    // Get first output
    if (graph.outputs.len > 0) {
        const out_idx = graph.outputs[0];
        const out_info = graph.tensors.items[out_idx];
        if (exec.getOutput(out_info.name)) |output| {
            std.debug.print("\nOutput '{s}':\n", .{out_info.name});
            std.debug.print("  dtype: {s}\n", .{@tagName(output.dtype)});
            std.debug.print("  shape: [", .{});
            for (output.shape, 0..) |dim, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{dim});
            }
            std.debug.print("]\n", .{});
            std.debug.print("  numel: {}\n", .{output.numel});

            // Print first few values
            if (output.dtype == .f32) {
                if (output.asSlice(f32)) |values| {
                    const num_to_show = @min(values.len, 10);
                    std.debug.print("  first {} values: [", .{num_to_show});
                    for (values[0..num_to_show], 0..) |v, i| {
                        if (i > 0) std.debug.print(", ", .{});
                        std.debug.print("{d:.4}", .{v});
                    }
                    std.debug.print("]\n", .{});
                }
            }
        }
    }
}

test "main imports" {
    // Verify imports work
    _ = tenzor;
    _ = clap;
}
