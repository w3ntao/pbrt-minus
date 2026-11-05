# pbrt-minus

[![cuda build](https://github.com/w3ntao/pbrt-minus/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/w3ntao/pbrt-minus/actions/workflows/cuda-build.yml)

A simpler, less performant, physically based, GPU ray tracer rewritten from PBRT-v4.


## Features

* CUDA acceleration (without OptiX)
* HLBVH with work queues ([HLBVH](https://research.nvidia.com/publication/2010-06_hlbvh-hierarchical-lbvh-construction-real-time-ray-tracing), [HLBVH with Work Queues](https://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues))
* [wavefront path tracing](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus), BDPT (CPU), MLT-path (CPU), MLT-BDPT (CPU)， selective-MLT
* spectral rendering, volumetric rendering, multiple importance sampling, ray split (trace reflection and refraction separately), light sampling (by power)
* Sampler: stratified sampler
* Denoising ([Statistical MC Denoising](https://www.cg.tuwien.ac.at/research/publications/2024/sakai-2024-asa/sakai-2024-asa-paper.pdf), [Edge-Avoiding à-trous](https://jo.dreggn.org/home/2010_atrous.pdf), [Reweighting Firefly Samples](https://cg.ivd.kit.edu/publications/2018/rwmc/reweighting-fireflies-preprint.pdf))


## Set up environment

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


## Gallery

More scenes at https://github.com/w3ntao/pbrt-minus-scenes.

![](https://github.com/w3ntao/pbrt-minus-gallery/raw/main/frame300-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/raw/main/frame300-gbuffer-diag.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/raw/main/frame675-wavefrontpath-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/raw/main/book-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/raw/main/bathroom-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/raw/main/staircase-1024.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/raw/main/material-testball-sequence-1024.png)

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/caustic-glass-v4-1024-before-denoise.png" width="49%">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/caustic-glass-v4-1024-density.png" width="49%">
</p>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/volumetric-caustic-1024.png" width="49%">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/volumetric-caustic-1024-density.png" width="49%">
</p>
