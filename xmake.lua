add_rules("mode.debug", "mode.release")

target("pbrt-cuda")
    set_kind("binary")
    add_files("src/*.cu")
    add_cugencodes("native")
