# pbrt-cuda

A simpler, less performant physically based ray tracer rewritten from PBRT-v4.

## requisite

* C++ (>= 17)
* CUDA (>= 12)
* CMake or [Xmake](https://xmake.io/)


## build

### clone repository

clone this repository with its submodule:
```
$ git clone --recursive https://github.com/w3ntao/pbrt-cuda.git
```

### build with CMake

```
$ mkdir build
$ cd build
$ cmake ..
$ make -j16
```

You should find executable at `pbrt-cuda/build/pbrt-cuda`

### build with Xmake

```
$xmake
```

You should find executable at `pbrt-cuda/build/linux/x86_64/release/pbrt-cuda`
