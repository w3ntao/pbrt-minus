cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

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
        src/ext/lodepng/lodepng.cpp
)

target_include_directories(${PROJ_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(${PROJ_NAME} PRIVATE ${PNG_LIBRARY} Threads::Threads)

set_property(TARGET ${PROJ_NAME} PROPERTY CUDA_ARCHITECTURES native)
target_compile_options(${PROJ_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        >)
