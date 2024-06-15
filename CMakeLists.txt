cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
# CUDA_ARCHITECTURES native is available in CMake since 3.24

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    # required by CLion
endif ()

message("CUDA compiler: ${CMAKE_CUDA_COMPILER}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJ_NAME "pbrt-minus")

project(${PROJ_NAME} LANGUAGES CUDA CXX C)

option(PBRT_FLOAT_AS_DOUBLE "use 64-bit floats" OFF)

if (PBRT_FLOAT_AS_DOUBLE)
    list(APPEND PBRT_DEFINITIONS "PBRT_FLOAT_AS_DOUBLE")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/ext)

enable_language(CUDA)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

find_package(Threads REQUIRED)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
add_definitions(${PNG_DEFINITIONS})

add_executable(${PROJ_NAME}
        src/pbrt/main.cu

        src/pbrt/base/bsdf.cu
        src/pbrt/base/bxdf.cu
        src/pbrt/base/camera.cu
        src/pbrt/base/film.cu
        src/pbrt/base/filter.cu
        src/pbrt/base/integrator.cu
        src/pbrt/base/interaction.cu
        src/pbrt/base/light.cu
        src/pbrt/base/material.cu
        src/pbrt/base/primitive.cu
        src/pbrt/base/sampler.cu
        src/pbrt/base/shape.cu
        src/pbrt/base/spectrum.cu
        src/pbrt/base/texture.cu

        src/pbrt/accelerator/hlbvh.cu

        src/pbrt/bxdfs/conductor_bxdf.cu
        src/pbrt/bxdfs/dielectric_bxdf.cu

        src/pbrt/euclidean_space/squared_matrix.cu

        src/pbrt/integrators/integrator_base.cu
        src/pbrt/integrators/random_walk.cu
        src/pbrt/integrators/simple_path.cu

        src/pbrt/lights/diffuse_area_light.cu
        src/pbrt/lights/distant_light.cu
        src/pbrt/lights/image_infinite_light.cu

        src/pbrt/materials/conductor_material.cu
        src/pbrt/materials/coated_diffuse_material.cu
        src/pbrt/materials/dielectric_material.cu
        src/pbrt/materials/diffuse_material.cu

        src/pbrt/primitives/geometric_primitive.cu

        src/pbrt/scene/parameter_dictionary.cu
        src/pbrt/scene/scene_builder.cu

        src/pbrt/shapes/disk.cu
        src/pbrt/shapes/loop_subdivide.cu
        src/pbrt/shapes/sphere.cu
        src/pbrt/shapes/tri_quad_mesh.cu

        src/pbrt/spectra/densely_sampled_spectrum.cu
        src/pbrt/spectra/piecewise_linear_spectrum.cu
        src/pbrt/spectra/rgb_illuminant_spectrum.cu
        src/pbrt/spectra/rgb_albedo_spectrum.cu

        src/pbrt/spectrum_util/global_spectra.cu
        src/pbrt/spectrum_util/rgb_to_spectrum_data.cu

        src/pbrt/textures/gpu_image.cu
        src/pbrt/textures/image_texture_base.cu
        src/pbrt/textures/spectrum_image_texture.cu
        src/pbrt/textures/spectrum_scale_texture.cu

        src/pbrt/util/sampling.cu
        src/pbrt/util/math.cu

        src/ext/lodepng/lodepng.cpp
        src/ext/rply/rply.cpp
        src/ext/tinyexr/deps/miniz/miniz.c
)

target_include_directories(
        ${PROJ_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${ZLIB_INCLUDE_DIRS}
        ${TINYEXR_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJ_NAME} PRIVATE
        Threads::Threads
        ${PNG_LIBRARY}
        ${ZLIB_LIBRARIES}
)

set_target_properties(
        ${PROJ_NAME} PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# CUDA_SEPARABLE_COMPILATION ON: member function declaration and implementation can be separate
# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

# CUDA_ARCHITECTURES native: CUDA_ARCHITECTURES is detected automatically
# https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures

target_compile_options(
        ${PROJ_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        >
)

target_compile_definitions(
        ${PROJ_NAME} PRIVATE
        ${PBRT_DEFINITIONS}
)
