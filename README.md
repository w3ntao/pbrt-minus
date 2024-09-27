# pbrt-minus

A simpler, less performant, physically based, GPU ray tracer rewritten from PBRT-v4.


## feature

* CUDA acceleration
* HLBVH with work queues ([Pantaleoni et al. 2010](https://research.nvidia.com/publication/2010-06_hlbvh-hierarchical-lbvh-construction-real-time-ray-tracing), [Garanzha et al. 2011](https://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues))
* wavefront path tracing ([Laine et al. 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf))
* spectral rendering
* multiple importance sampling
* power light sampler
* stratified sampler


## set up environment

### Linux

Debian/Ubuntu:
```
$ sudo apt install -y libglu1-mesa-dev libpng-dev libx11-dev xorg-dev
```

Setting up for other distros should be similar.

### Windows

It's recommended to build it with [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl) might help set up CUDA.


## build and render

```
$ git clone --recursive https://github.com/w3ntao/pbrt-minus.git

$ cd pbrt-minus
$ mkdir build; cd build
$ cmake ..; make -j

$ ./pbrt-minus ../example/cornell-box-specular.pbrt --spp 4
```


## gallery

More scenes at https://github.com/w3ntao/pbrt-minus-scenes.

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/book-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/chopper-titan-v4-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/crown-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/ganesha-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/ganesha-coated-gold-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/lte-orb-rough-glass-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/lte-orb-silver-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/veach-mis-colorized-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame25-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame35-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame52-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame85-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame120-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame180-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame210-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame300-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame542-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame675-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame812-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame888-path-4096.png)

![](https://github.com/w3ntao/pbrt-minus-gallery/blob/main/frame1266-path-4096.png)

