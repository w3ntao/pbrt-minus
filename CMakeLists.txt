cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    # required by CLion
endif ()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(pbrt-cuda LANGUAGES CUDA CXX)

find_package(PNG QUIET)
if (PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIRS})
    add_definitions(${PNG_DEFINITIONS})
    set(LIBRARIES ${LIBRARIES} ${PNG_LIBRARIES})
else ()
    find_package(PNG)
    message(FATAL_ERROR "libpng-dev is required to build the project.\nPlease install libpng-dev.")
endif ()

set(EXEC_PROGRAM "pbrt-cuda")

add_executable(${EXEC_PROGRAM}
        src/main.cu
        src/util/common.h
        src/base/vector3.h
        src/base/point.h
        src/base/ray.h
        src/base/color.h
        src/util/image.cu
        src/util/image.h
        src/base/shape.h
        src/shapes/sphere.h
        src/base/world.h
        src/base/camera.h
        src/base/material.h
        src/cameras/perspective.h
        src/base/integrator.h
        src/integrators/path.h)

target_include_directories(${EXEC_PROGRAM} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(${EXEC_PROGRAM} PRIVATE ${PNG_LIBRARY})

target_compile_options(${EXEC_PROGRAM} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        >)
