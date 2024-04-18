# pbrt-minus

A simpler, less performant physically based GPU ray tracer rewritten from PBRT-v4.

## requisite

* C++ >= 17
* CUDA (compute capability >= 7.5, runtime version >= 11)
* CMake (>= 3.24)
* PNG library ([for Debian](https://packages.debian.org/search?keywords=libpng-dev), [for Arch](https://archlinux.org/packages/extra/x86_64/libpng/))

## build

### clone repository

clone this repository with its submodules:

```
$ git clone --recursive https://github.com/w3ntao/pbrt-minus.git
```

### build with CMake

```
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```

You should find executable at `pbrt-minus/build/pbrt-minus`


### preview

scenes: https://github.com/w3ntao/pbrt-minus-scenes \
(currently most scenes are borrowed from https://github.com/mmp/pbrt-v4-scenes)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/cornell-box-ao-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/cornell-box-sn-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/killeroo-simple-ao-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/killeroo-simple-sn-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/killeroo-coated-gold-ao-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/killeroo-coated-gold-sn-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/lte-orb-rough-glass-ao-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/lte-orb-rough-glass-sn-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/ganesha-ao-1024.png)

![](https://github.com/w3ntao/pbrt-minus-preview/blob/main/ganesha-sn-1024.png)
