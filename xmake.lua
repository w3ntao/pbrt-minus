add_rules("mode.debug", "mode.release")

add_requires("png", {system = true})

target("pbrt-cuda")
    set_kind("binary")
    add_includedirs("src")
    add_files("src/*/*.cu")
    add_files("src/main.cu")
    add_cugencodes("native")
    add_packages("png")
