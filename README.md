# pbrt-minus

[![cuda build](https://github.com/w3ntao/pbrt-minus/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/w3ntao/pbrt-minus/actions/workflows/cuda-build.yml)

A simpler, less performant, physically based, GPU ray tracer rewritten from PBRT-v4.


## Features

* CUDA acceleration (without OptiX)
* [HLBVH](https://research.nvidia.com/publication/2010-06_hlbvh-hierarchical-lbvh-construction-real-time-ray-tracing) with [Work Queues](https://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues)
* [wavefront path tracing](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus), BDPT (CPU), MLT-path (CPU), MLT-BDPT (CPU), [selective-MLT](https://benedikt-bitterli.me/smlt/)
* spectral rendering, volumetric rendering, multiple importance sampling, ray split (trace both reflection and refraction), light sampling (by power)
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


## Build and render

```
$ git clone --recursive https://github.com/w3ntao/pbrt-minus.git

$ cd pbrt-minus
$ mkdir build; cd build
$ cmake ..; make -j

$ ./pbrt-minus ../example/cornell-box-specular.pbrt --preview --spp 16
```


## Gallery

source: https://github.com/w3ntao/pbrt-minus-scenes

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/zero-day/frame300-1024.png" alt="">
</p>
<p align="center">zero-day/frame300.pbrt</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/zero-day/frame300-gbuffer-diag.png" alt="">
</p>
<p align="center">denoiser input: normal, albedo, luminance</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/transparent-machines/frame675-1024-before-denoise.png" alt="">
</p>
<p align="center">transparent-machines/frame675.pbrt</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/book-1024.png" alt="">
</p>
<p align="center">pbrt-book/book.pbrt</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/chopper-titan-v4-1024.png" alt="">
</p>
<p align="center">chopper-titan/chopper-titan-v4.pbrt</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/staircase-1024.png" alt="">
</p>
<p align="center">staircase/staircase.pbrt</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/material-testball-sequence-1024.png" alt="">
</p>
<p align="center">material-testball/material-testball-sequence.pbrt</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/caustic-glass-v4-1024-before-denoise.png" width="49%" alt="">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/caustic-glass-v4-1024-density.png" width="49%" alt="">
</p>
<p align="center">caustic-glass/caustic-glass-v4.pbrt &nbsp;|&nbsp; MLT sampling density</p>
<br>

<p align="center">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/volumetric-caustic-1024.png" width="49%" alt="">
  <img src="https://github.com/w3ntao/pbrt-minus-gallery/raw/main/volumetric-caustic-1024-density.png" width="49%" alt="">
</p>
<p align="center">volumetric-caustic/volumetric-caustic.pbrt &nbsp;|&nbsp; MLT sampling density</p>
