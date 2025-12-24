const std = @import("std");

pub fn addBlas(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    // Create a stub header that includes cblas.h
    const wrapper = b.addWriteFiles();
    const cblas_wrapper = wrapper.add("cblas_wrapper.h",
        \\#include <cblas.h>
    );

    const translate_c = b.addTranslateC(.{
        .root_source_file = cblas_wrapper,
        .target = target,
        .optimize = optimize,
    });

    const mod = translate_c.createModule();
    mod.linkSystemLibrary("openblas", .{ .preferred_link_mode = .dynamic });
    return mod;
}
