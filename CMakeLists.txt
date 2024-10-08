cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
# CUDA_ARCHITECTURES native is available in CMake since 3.24

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    # required by CLion
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJ_NAME "pbrt-minus")

project(${PROJ_NAME} LANGUAGES CUDA CXX C)

option(PBRT_FLOAT_AS_DOUBLE "use 64-bit floats" OFF)

if (PBRT_FLOAT_AS_DOUBLE)
    list(APPEND PBRT_DEFINITIONS "PBRT_FLOAT_AS_DOUBLE")
endif ()

enable_language(CUDA)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

message(STATUS "nvcc path: ${CMAKE_CUDA_COMPILER}")
message(STATUS "nvcc version = ${CMAKE_CUDA_COMPILER_VERSION}")

if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "12.5.0")
    message(FATAL_ERROR "you need NVCC >= 12.5.0 to build this project (required by cuda::std::optional)")
endif ()

add_subdirectory("src/ext")

include_directories("src/ext/glad-3.3-core/include")

include_directories("${GLFW_SOURCE_DIR}/deps")
link_libraries(glfw)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
link_libraries(${OPENGL_LIBRARIES})

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
link_libraries(${PNG_LIBRARY})
add_definitions(${PNG_DEFINITIONS})

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

add_executable(${PROJ_NAME}
        src/pbrt/main.cu

        src/pbrt/base/bsdf.cu
        src/pbrt/base/bxdf.cu
        src/pbrt/base/camera.cu
        src/pbrt/base/film.cu
        src/pbrt/base/filter.cu
        src/pbrt/base/float_texture.cu
        src/pbrt/base/integrator_base.cu
        src/pbrt/base/interaction.cu
        src/pbrt/base/light.cu
        src/pbrt/base/material.cu
        src/pbrt/base/megakernel_integrator.cu
        src/pbrt/base/primitive.cu
        src/pbrt/base/sampler.cu
        src/pbrt/base/shape.cu
        src/pbrt/base/spectrum.cu
        src/pbrt/base/spectrum_texture.cu

        src/pbrt/accelerator/hlbvh.cu

        src/pbrt/bxdfs/conductor_bxdf.cu
        src/pbrt/bxdfs/dielectric_bxdf.cu

        src/pbrt/cameras/perspective.cu

        src/pbrt/euclidean_space/squared_matrix.cu

        src/pbrt/films/rgb_film.cu

        src/pbrt/gpu/renderer.cu

        src/pbrt/integrators/ambient_occlusion.cu
        src/pbrt/integrators/path.cu
        src/pbrt/integrators/random_walk.cu
        src/pbrt/integrators/simple_path.cu
        src/pbrt/integrators/surface_normal.cu

        src/pbrt/lights/diffuse_area_light.cu
        src/pbrt/lights/distant_light.cu
        src/pbrt/lights/image_infinite_light.cu

        src/pbrt/light_samplers/power_light_sampler.cu

        src/pbrt/materials/conductor_material.cu
        src/pbrt/materials/coated_conductor_material.cu
        src/pbrt/materials/coated_diffuse_material.cu
        src/pbrt/materials/dielectric_material.cu
        src/pbrt/materials/diffuse_material.cu
        src/pbrt/materials/mix_material.cu

        src/pbrt/primitives/geometric_primitive.cu
        src/pbrt/primitives/transformed_primitive.cu

        src/pbrt/scene/parameter_dictionary.cu
        src/pbrt/scene/scene_builder.cu

        src/pbrt/shapes/disk.cu
        src/pbrt/shapes/loop_subdivide.cu
        src/pbrt/shapes/sphere.cu
        src/pbrt/shapes/triangle_mesh.cu
        src/pbrt/shapes/tri_quad_mesh.cu

        src/pbrt/spectra/densely_sampled_spectrum.cu
        src/pbrt/spectra/piecewise_linear_spectrum.cu
        src/pbrt/spectra/rgb_albedo_spectrum.cu
        src/pbrt/spectra/rgb_illuminant_spectrum.cu
        src/pbrt/spectra/rgb_unbounded_spectrum.cu

        src/pbrt/spectrum_util/global_spectra.cu
        src/pbrt/spectrum_util/rgb_to_spectrum_data.cu

        src/pbrt/textures/float_scaled_texture.cu
        src/pbrt/textures/gpu_image.cu
        src/pbrt/textures/image_texture_base.cu
        src/pbrt/textures/spectrum_image_texture.cu
        src/pbrt/textures/spectrum_scaled_texture.cu
        src/pbrt/textures/texture_mapping_2d.cu

        src/pbrt/util/distribution_2d.cu
        src/pbrt/util/hash_map.cu
        src/pbrt/util/sampling.cu

        src/pbrt/wavefront_integrators/path.cu

        src/ext/glad-3.3-core/src/glad.c
        src/ext/lodepng/lodepng.cpp
        src/ext/rply/rply.cpp
        src/ext/tinyexr/deps/miniz/miniz.c
)


target_include_directories(
        ${PROJ_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${TINYEXR_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJ_NAME} PRIVATE
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
