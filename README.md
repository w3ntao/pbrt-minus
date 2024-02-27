# pbrt-minus

A simpler, less performant physically based ray tracer rewritten from PBRT-v4.

## requisite

* C++ (>= 17)
* CUDA (>= 12)
* CMake or [Xmake](https://xmake.io/)
* PNG
  library ([for Debian](https://packages.debian.org/search?keywords=libpng-dev), [for Arch](https://archlinux.org/packages/extra/x86_64/libpng/))

## build

### clone repository

clone this repository with its submodule:

```
$ git clone --recursive https://github.com/w3ntao/pbrt-minus.git
```

### build with CMake

```
$ mkdir build
$ cd build
$ cmake ..
$ make -j16
```

You should find executable at `pbrt-minus/build/pbrt-minus`

### build with Xmake

```
$ xmake
```

You should find executable at `pbrt-minus/build/linux/x86_64/release/pbrt-minus`
