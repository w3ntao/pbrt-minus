# pbrt-minus

A simpler, less performant, physically based, GPU ray tracer rewritten from PBRT-v4.

## feature

* CUDA acceleration
* spectral rendering
* HLBVH with work queues ([Pantaleoni et al. 2010](https://research.nvidia.com/publication/2010-06_hlbvh-hierarchical-lbvh-construction-real-time-ray-tracing), [Garanzha et al. 2011](https://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues))
* integrator: AmbientOcclusion, SurfaceNormal, PathTracing

## requisite

* C++ >= 17
* CUDA (compute capability >= 7.5, runtime version >= 11)
* CMake (>= 3.24)
* PNG library ([for Debian](https://packages.debian.org/search?keywords=libpng-dev), [for Arch](https://archlinux.org/packages/extra/x86_64/libpng/))


## build and render

```
# clone this repository and its submodules
$ git clone --recursive https://github.com/w3ntao/pbrt-minus.git

# build with CMake
$ cd pbrt-minus
$ mkdir build
$ cd build
$ cmake ..
$ make -j

# render
$ ./pbrt-minus ../example/cornell-box-specular.pbrt --spp 4
```

More scenes could be found at https://github.com/w3ntao/pbrt-minus-scenes.

## gallery

All rendered with scenes from https://github.com/w3ntao/pbrt-minus-scenes \
(which is basically a subset of https://github.com/mmp/pbrt-v4-scenes).

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/ganesha-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/lte-orb-silver-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/lte-orb-rough-glass-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/cornell-box-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/cornell-box-specular-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/veach-mis-colorized-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/killeroo-gold-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/killeroo-simple-simplepath-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/dragon_10-ambientocclusion-stratified-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/crown-surfacenormal-stratified-1024.png)
