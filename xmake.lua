add_rules("mode.debug", "mode.release")

add_requires("png", { system = true })

target("pbrt-minus")
set_kind("binary")
add_cugencodes("native")
add_cuflags("--expt-relaxed-constexpr")
add_packages("png")

add_includedirs("src")
add_files(
        "src/pbrt/main.cu",
        "src/pbrt/base/shape.cu",
        "src/pbrt/euclidean_space/squared_matrix.cu",
        "src/ext/lodepng/lodepng.cpp"
)
target_end()
