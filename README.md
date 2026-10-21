# pbrt-minus

[![cuda build](https://github.com/w3ntao/pbrt-minus/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/w3ntao/pbrt-minus/actions/workflows/cuda-build.yml)

A simpler, less performant, physically based, GPU ray tracer rewritten from PBRT-v4.


## feature

* CUDA acceleration
* wavefront path tracing, BDPT, MLT-path, MLT-BDPT
* HLBVH with work queues
* spectral rendering, multiple importance sampling, light sampling (by power), stratified sampler, russian roulett


## set up environment

### Linux

Debian/Ubuntu:
```
$ sudo apt install -y \
  cmake libglfw3-dev libglu1-mesa-dev libpng-dev libx11-dev xorg-dev libxrandr-dev
```

Setting up for other distros should be similar.

### Windows

* It's recommended to build with [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install).
* [CUDA on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/#getting-started-with-cuda-on-wsl-2) might help set up CUDA.


## build and render

```
$ git clone --recursive https://github.com/w3ntao/pbrt-minus.git

$ cd pbrt-minus
$ mkdir build; cd build
$ cmake ..; make -j

$ ./pbrt-minus ../example/cornell-box-specular.pbrt --preview --spp 16
```


## gallery

More scenes at https://github.com/w3ntao/pbrt-minus-scenes.

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame300-wavefrontpath-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame300-diag-1024-0.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame675-wavefrontpath-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/book-wavefrontpath-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/bathroom-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/staircase-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/staircase2-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/material-testball-sequence-4096.png)

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/caustic-glass-v4-1024-mlt.png" width="48%">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/caustic-glass-v4-1024-density.png" width="48%">
</p>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/volumetric-caustic-1024-mlt.png" width="48%">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/volumetric-caustic-1024-density.png" width="48%">
</p>
