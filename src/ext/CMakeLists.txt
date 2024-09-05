cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only")
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "GLFW lib only")
add_subdirectory(glfw)

###########################################################################
# zlib

find_package(ZLIB)
if (NOT ZLIB_FOUND)
    # Build zlib
    set(ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
    set(ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
    add_subdirectory(zlib)

    set(ZLIB_LIBRARIES zlibstatic)
    set(ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)

    # try to make openexr happy about this...
    set(ZLIB_LIBRARY zlibstatic)
    set(ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)
    set(ZLIB_FOUND TRUE)

    set_property(TARGET zlibstatic PROPERTY FOLDER "ext")

    add_library(ZLIB::ZLIB ALIAS zlibstatic)
    include_directories(${ZLIB_INCLUDE_DIRS})  # yuck, but so openexr/ptex can find zlib.h...
endif ()

set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS} PARENT_SCOPE)
set(ZLIB_LIBRARIES ${ZLIB_LIBARIES} PARENT_SCOPE)


set(TINYEXR_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr/deps/miniz PARENT_SCOPE)
