const std = @import("std");
const Translator = @import("translate_c").Translator;

pub fn addBlas(
    b: *std.Build,
    translate_c_dep: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    // Create a stub header that includes cblas.h
    const wrapper = b.addWriteFiles();
    const cblas_wrapper = wrapper.add("cblas_wrapper.h",
        \\#include <cblas.h>
    );

    // Use translate-c to generate Zig bindings
    var t: Translator = .init(translate_c_dep, .{
        .name = "cblas",
        .c_source_file = cblas_wrapper,
        .target = target,
        .optimize = optimize,
    });

    // Link OpenBLAS directly for high-performance BLAS
    // OpenBLAS provides the CBLAS interface
    t.mod.linkSystemLibrary("openblas", .{ .preferred_link_mode = .dynamic });

    return t.mod;
}
