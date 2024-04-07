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

project(${PROJ_NAME} LANGUAGES CUDA CXX)

find_package(Threads REQUIRED)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
add_definitions(${PNG_DEFINITIONS})
set(LIBRARIES ${LIBRARIES} ${PNG_LIBRARIES})

add_executable(${PROJ_NAME}
        src/pbrt/main.cu
        src/pbrt/base/camera.cu
        src/pbrt/base/film.cu
        src/pbrt/base/filter.cu
        src/pbrt/base/integrator.cu
        src/pbrt/base/sampler.cu
        src/pbrt/base/shape.cu
        src/pbrt/base/spectrum.cu
        src/pbrt/accelerator/hlbvh.cu
        src/pbrt/euclidean_space/squared_matrix.cu
        src/pbrt/shapes/loop_subdivide.cu
        src/pbrt/spectra/densely_sampled_spectrum.cu
        src/pbrt/spectra/rgb_to_spectrum_data.cu
        src/ext/lodepng/lodepng.cpp
)

target_include_directories(
        ${PROJ_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(
        ${PROJ_NAME} PRIVATE
        ${PNG_LIBRARY}
        Threads::Threads
)

set_target_properties(
        ${PROJ_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
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
