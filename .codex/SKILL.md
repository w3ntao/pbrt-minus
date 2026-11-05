---
name: pbrt-minus
description: 用中文理解和导航 pbrt-minus CUDA 光谱渲染器，说明构建入口、场景解析、GPU 积分器及模块关系，为代码阅读与修改定位文件，并以 pbrt-minus/pbrt-v4 代码为基础解释渲染论文与相关问题。适用于工程结构说明、执行流程分析、功能定位和渲染论文解读；默认只关注 src/pbrt，不展开 src/ext。
---

# pbrt-minus 工程结构与代码导航

## 维护元规则

- 修改代码前先评估与已有架构的适配性，尽量遵循现有架构、模块职责和接口约定；若无法匹配已有架构，或匹配后会造成明显不合理的设计，则不进行相关代码修改，并向用户说明原因。
- 回答用户问题时（尤其是论文相关问题），默认以 pbrt-minus 作为用户的知识边界，从本工程已有的概念和实现出发解释；超出此边界的术语、背景或算法，先补充必要的前置知识，再建立联系。用户明确说明其知识背景时，以该说明为准。
- 解释时尽量以 pbrt-minus/pbrt-v4 的相关代码为基础，优先引用 pbrt-minus 中已核实的文件、类或函数；解读论文时，将关键概念、公式符号和算法步骤对应到具体实现。借助 pbrt-v4 补充或只能类比时，明确说明与本工程实现的差异；没有对应实现时如实指出，不暗示工程已支持。
- Skill 只记录工程结构、模块职责、关键语义和代码定位，不记录具体算法的数值参数（如采样数量、滤波半径、标准差、阈值或距离比例）；需要这些数值时读取当前源码。
- CUDA kernel 启动（如 `<<<...>>>`）一般不应放在循环里：单层循环内启动可能是设计失误；双重或更深层嵌套循环内启动大概率是设计错误。
- 阅读、审查或生成代码时，若发现循环内启动 CUDA kernel，必须指出代码位置并提醒作者；若位于双重或更深层嵌套循环内，必须明确警告作者，并说明可能的设计问题。
- 修改代码后不主动修改根目录 `README.md`；只有用户明确要求时才编辑。可以向用户指出已核实的明显偏差，并说明对应的代码位置。
- 主要参考实现是 [pbrt-v4](https://github.com/mmp/pbrt-v4/)。对照实现时优先查阅本地 clone `~/Desktop/pbrt-v4`；本地不可用或需要核对上游时再查阅 GitHub。
- 场景文件主要来自 [pbrt-minus-scenes](https://github.com/w3ntao/pbrt-minus-scenes)。查找场景时优先使用本地 clone `~/Desktop/pbrt-minus-scenes`；本地不可用或需要核对上游时再查阅 GitHub。

## 使用范围

这是一份轻量的工程地图。解释工程或定位改动时，先沿下面的主流程找到负责模块，再读取相关实现；具体接口和支持范围以当前源码为准。

- 仓库根目录以本文件所在目录的上一级为准；在其他 checkout 中也按本文件位置解析，不使用固定绝对路径。
- 构建入口是根目录 `CMakeLists.txt`，项目说明是根目录 `README.md`；当前仓库没有单独的 `pbrt/CMakeLists.txt`。
- 代码阅读默认限定在 `src/pbrt/`。`src/ext/` 是第三方依赖边界，仅根据根 CMake 和调用位置了解其作用，不展开其实现；用户明确调整范围时按新要求处理。
- 本文记录结构，不代表已经编译或验证渲染结果。代码变化后重新核对工厂、类型分发和积分器调度。

## 项目与构建

这是从 PBRT-v4 重写、简化的物理 GPU 光线追踪器，使用 CUDA 加速并在光谱域计算辐射，支持单向路径追踪、双向路径追踪、MLT 和 OpenGL 渐进预览。

根 `CMakeLists.txt` 定义一个可执行目标 `pbrt-minus`，显式列出需要编译的 `.cu` 源文件，因此新增编译单元时要更新该清单。头文件中的实现无需单独列入。

构建要求 CMake ≥ 3.25.2，启用 CUDA、C++、C，C++/CUDA 标准均为 20；默认 `Release`，CUDA 架构设为 `native`。`PBRT_FLOAT_AS_DOUBLE` 控制 `gpu/macro.h` 中的 `Real`，默认 `float`，打开后为 `double`。目标启用 CUDA 分离编译和设备符号解析，并导出 `compile_commands.json`。

依赖包括 CUDA 工具链、GLFW、OpenGL、PNG、Threads，以及 CMake 接入的第三方库。具体安装说明参考 `README.md`。

在仓库根目录可使用：

```bash
cmake -S . -B build
cmake --build build -j
./build/pbrt-minus ./example/cornell-box-specular.pbrt --spp 16 --preview
```

`--preview` 启用窗口预览。命令行还支持 `--integrator`、`--outfile`；解析入口在 `src/pbrt/scene/command_line_option.h`。

## 主执行链

```text
src/pbrt/main.cu::main
  → CommandLineOption
  → SceneBuilder::render_pbrt
      → parse_file → parse_pbrt_into_token → Lexer / Token
      → parse_tokens / parse_keyword → 场景对象与资源
      → preprocess → 根 HLBVH、光源预处理、build_integrator
      → render → 对应积分器 → Film::write_to_png
```

`main` 先检查 CUDA 配置、设置设备栈并输出 GPU 信息，再进入场景构建；结束时检查错误、同步并重置设备。

`scene/parser.h` 主要收集 token，`lexer.h` 负责分词，`tokenizer.h` 定义 token。场景语义解释集中在 `scene/scene_builder.cu`，而非 `parser.h`。`ParameterDictionary` 将 PBRT 参数转换成数值、几何量、光谱及命名纹理/材质引用，供各对象工厂使用。

`SceneBuilder` 维护命名资源表和 `GraphicsState`。后者记录当前变换、材质、面光源和介质边界；`AttributeBegin/End` 保存和恢复状态。遇到 `WorldBegin` 时按 **filter → film → camera** 的顺序构建对象并设置渲染坐标系，随后解析世界中的形状和光源。`ObjectBegin/End` 收集实例定义，多个图元通过实例内部 BVH 聚合，`ObjectInstance` 复用或包装变换后的图元。

`preprocess` 建立场景根 BVH，根据场景尺度设置射线偏移，预处理光源，构建 GPU 可访问的灯光数组和 `PowerLightSampler`，最后选择积分器。`IntegratorBase` 汇总这些共享数据并提供求交、遮挡和透射率接口。

## 目录职责

以下路径均相对于 `src/pbrt/`。

| 目录 | 主要职责与连接关系 |
|---|---|
| `scene/` | 命令行、分词、参数字典、场景状态与对象构建；全局调度中心是 `SceneBuilder`。 |
| `base/` | Camera、Film、Filter、Material、BxDF、Shape、Primitive、Light、Sampler、Spectrum、Texture 等统一接口、工厂和类型分发；也包含 Ray、Interaction、BSDF 和积分器共享上下文。 |
| `integrators/` | 路径追踪、BDPT 和 MLT；G-buffer 诊断属性由 `base/gbuffer_integrator.*` 统一计算。 |
| `accelerator/` | 自实现 HLBVH：构建层次包围盒，提供详细求交和快速遮挡测试。 |
| `gpu/` | 统一内存分配与释放、CPU/GPU 函数标记、浮点精度和 CUDA 错误检查。 |
| `shapes/`、`primitives/` | Shape 负责几何求交、面积与表面采样；Primitive 将几何绑定到材质、面光源、介质或实例变换。 |
| `materials/`、`bxdfs/` | Material 从参数和纹理生成散射模型；BxDF 实现散射求值、采样及 PDF；BSDF 负责局部着色坐标与渲染坐标的转换。 |
| `lights/`、`light_samplers/` | 远光、聚光、面光源和环境光；光源采样器决定选择哪盏灯，共享上下文使用功率采样器。 |
| `textures/` | 浮点/光谱纹理、求值上下文、坐标映射、图像加载及 MIPMap 过滤。 |
| `spectra/`、`spectrum_util/` | 前者实现具体光谱表示；后者提供采样光谱/波长、RGB/XYZ、颜色空间、RGB 到光谱数据及 CIE 等常量。 |
| `cameras/` | 透视相机、相机光线与采样/PDF，及相机、世界、渲染坐标间的变换。 |
| `film/`、`filters/` | RGB 样本和 splat 累积、传感器颜色转换、图像重建滤波；灰度 Film 用于 MLT 热力图。 |
| `samplers/`、`distribution/` | 独立、分层、MLT 样本流；alias table、1D/2D 与分段常量分布提供重要性采样。 |
| `medium/` | 均匀介质的吸收/散射光谱、HG 相函数和内外介质边界。 |
| `euclidean_space/`、`util/` | 点、向量、法线、包围盒、矩阵、变换、误差区间，以及 RNG、数学、采样、哈希、容器、线程池和计时。 |
| `gui/` | `GLHelper` / `Shader` 提供 GLFW/OpenGL 预览，显示 Film 转换出的 RGB 帧缓冲。 |

## 接口分发与 CPU/GPU 边界

`base/` 的主要扩展方式是类型分发：

- `Material`、`Shape`、`Primitive`、`Light`、`Sampler`、`Spectrum`、`FloatTexture`、`SpectrumTexture` 使用 `enum Type` 加 `void*` 或 `const void*`，通过 `switch` / `static_cast` 调用具体实现。新增类型通常需要同时更新类型标签、构造/工厂和各个分发入口。
- `BxDF`、`Camera`、`Film`、`Filter` 使用 `cuda::std::variant` 和 `visit`，新增实现需要扩展对应 variant。当前 Camera 只有 `PerspectiveCamera`，Film 只有 `RGBFilm`；`GreyScaleFilm` 是独立的热力图辅助对象。
- `IntegratorBase` 是共享场景上下文结构，积分器选择与总体调度在 `SceneBuilder`。

CPU 负责文本读取、命名资源和状态栈、host 调度以及部分排序/图像累积；CUDA kernel 负责主要渲染计算。`.cu` 文件经常同时包含两侧代码，按函数标记和调用方式判断执行位置。

`GPUMemoryAllocator::allocate/create` 使用 `cudaMallocManaged`，其中 `create` 在统一内存上执行 placement new，并集中记录、释放分配。`SceneBuilder::gpu_primitives/gpu_lights` 是 CPU vector，里面收集指向 GPU 可访问对象的指针；渲染使用的连续指针数组另行构建。

`gpu/macro.h` 中 `PBRT_CPU_GPU` 表示 `__host__ __device__`，`PBRT_GPU` 表示 `__device__`。新增渲染数据时遵循现有分配方式，并检查其成员是否可被设备代码访问。

HLBVH 构建也横跨 CPU/GPU：GPU 计算 Morton 编码并建立下层 treelet，CPU 的 `build_upper_sah` 和线程池处理上层聚合。

## 积分器与输出

名称映射以 `SceneBuilder::build_integrator` 为准：

| 场景或 `--integrator` 名称 | 实现与执行方式 |
|---|---|
| `path`、`volpath` | `WavefrontPathIntegrator`；`volpath` 映射为 `path`。路径状态按字段存放在数组中，按射线、介质、无材质界面和具体材质分队列，分阶段启动 kernel。 |
| `megakernelpath` | `MegakernelPathIntegrator` 独立调度；一个线程完成一个像素的样本与完整路径计算。 |
| `bdpt` | `BDPTIntegrator` 生成相机/光源子路径并连接，使用分批 GPU 计算与 CPU 累积。 |
| `mlt`、`mltbdpt` | `MLTBDPTIntegrator`，以 bootstrap、变异和接受/拒绝流程驱动采样，复用 BDPT 子路径与连接函数。 |
| `mltpath` | `MLTPathIntegrator`，复用 `MegakernelPathIntegrator::evaluate_Li_volume` 计算路径贡献。 |
| `gbuffer` | `base/gbuffer_integrator.*` 的 `GBufferIntegrator` 通过单次渲染 kernel 同时计算环境遮蔽、表面法线、albedo 和深度；主 PNG 显示 AO，不提供窗口预览。 |

G-buffer 原始值保存在 `film/gbuffer.*` 的浮点数组及 `*-gbuffer.exr` 中；独立生成 `*-ao.png`、`*-normal.png`、`*-albedo.png`、`*-depth.png` 可视化，绕过 Film 的光谱辐射转换。法线为世界空间 shading normal，深度为相机空间 Z；albedo 模仿 pbrt-v4，在每个首表面交点固定观察方向 `wo`，使用相同的固定 BSDF 样本组，以 Radiance 模式同时采样反射和透射并平均 `f * abs_cos / pdf`，失败样本贡献 0，仍按完整样本组数量求平均，再按 reference white 转为线性 RGB；玻璃包含折射率对应的 Radiance 缩放，结果不是简单的 `R + T = 1`，低 SPP 仍有波长采样噪声。未命中深度为正无穷，并记录 coverage。

AO 的最大射线距离按世界空间场景包围盒的尺度自动确定，用于估计局部遮蔽；不再读取场景中的 `maxdistance` 参数。环境遮蔽和表面法线不再使用独立积分器。

BDPT 内部的 `wavefront_render` kernel 是其批处理实现，阅读时不要据此推断它复用了 `WavefrontPathIntegrator` 的材质队列。

目前默认积分器是 `path`。场景 `Sampler` 指定采样器和样本数，命令行 `--spp` 可以覆盖样本数；当前支持 `independent` 和 `stratified`，默认使用分层采样，其他采样器名称回退为分层采样。分层采样要求样本数为完全平方数；MLT 使用自己的采样器，`--spp` 在 MLT 中表示每像素变异次数。

当前传入 `--integrator` 会让 `parse_keyword` 跳过场景中的整条 `Integrator` 配置，包括其附带参数。比较自定义积分器参数时，优先在场景中修改积分器名称，或核对命令行覆盖后实际使用的默认参数。

渲染贡献沿 `SampledSpectrum + SampledWavelengths → PixelSensor → RGBFilm` 转为输出 RGB；`Film::write_to_png` 编码为 sRGB PNG，`copy_to_frame_buffer` 为预览提供 RGB 数据。BDPT 的 splat 按 `1 / spp` 缩放，MLT 按返回亮度与变异次数归一化，并额外输出 `heatmap-` 前缀的 PNG。

所有 radiance integrator（`path`、`megakernelpath`、`bdpt`、`mltbdpt`、`mltpath`、`selectivemlt`）默认在最终输出前执行完整 StatMC 图像降噪：`RGBFilm` 对完整 camera sample 的未乘重建 filter 权重、未乘输出 RGB 矩阵的 CIE 1931 sensor XYZ 做 Box–Cox 变换，在线保存均值、二阶和三阶中心矩；负数或非有限 XYZ 样本会报错，转换后的负 RGB 分量不截断。这里使用 XYZ 检验是对论文 RGB 检验的光谱适配。Megakernel 和 wavefront path 使用实际逐像素 camera sample；BDPT 将统计指针传给独占像素的 worker film，但统计不包含其 light-to-camera splats；MLT 和 Selective MLT 使用额外的独立相机路径追踪 pass 生成逐像素统计引导，Markov-chain splats 不作为独立样本。后两类统计对最终图像是近似，且增加一次 PT pass 开销。`SceneBuilder::render` 用 `GBufferIntegrator::render_denoising_guides` 收集 normal、albedo、coverage，跳过 AO 和 depth。`film/statistical_denoiser.*` 组合联合双边权重与偏度修正统计检验，再滤原始线性 RGB；albedo 引导保持 RGB。非 Box filter、分层采样及 MLT 引导下统计检验为近似处理。输出保留 `*-before-denoise.png`，完整版本还输出 `*-denoise-neighbors.png` 和 `*-denoise-weights.png` 诊断图。

调用 StatMC 前，`SceneBuilder::render` 使用 `film/firefly_filter.*` 的 `firefly::Prefilter` 在完整线性 RGB（含已归一化的 splat）上抑制孤立异常亮点，以 normal、albedo 和 coverage 限制可参考的邻居，并保留有稳定相机样本统计支持的亮点。这是图像空间预处理，不是论文的逐样本重加权；小而孤立的真实高光仍可能被衰减。处理时统一缩放像素 RGB，并通过 `TransformedStatistics::rescale` 对 Box–Cox 均值及中心矩做相应变换，随后从处理后的图像更新 luminance 引导，再调用降噪器。`*-before-firefly.png` 保留原始渲染，`*-before-denoise.png` 保存实际送入 StatMC 的清理后图像。

## 常见阅读或修改入口

| 任务 | 优先检查 |
|---|---|
| PBRT 语法、参数、Include、命名资源、实例或变换 | `scene/lexer.h`、`tokenizer.h`、`parser.h`、`parameter_dictionary.*`、`scene_builder.*`；相对资源路径由 `SceneBuilder::root` 参与解析。 |
| 新增材质或调整散射 | `materials/` → `bxdfs/` → `base/material.*` / `base/bxdf.h`；涉及 wavefront 时还检查 `Material::get_basic_material_types()` 与材质队列/分发。 |
| 新增几何或调整求交 | `shapes/` → `base/shape.*` → `primitives/`；PLY/网格导入和细分最终生成 Triangle，运行时 Shape 类型是 disk/sphere/triangle。 |
| 光源采样、发光行为或灯光选择 | `lights/`、`base/light.*`、`light_samplers/`；按积分器检查入射光与发射光接口，面光源绑定还涉及 `GeometricPrimitive`。 |
| 路径估计、MIS、体积散射或 MLT | 对应 `integrators/`，并检查 `base/integrator_base.*`、`medium/`、`samplers/mlt.*`。 |
| 纹理或颜色问题 | `textures/`、`spectra/`、`spectrum_util/`；输出偏色沿 `PixelSensor → RGBFilm::get_pixel_rgb → Film` 检查。 |
| 相机、滤波、预览或图片输出 | `cameras/`、`filters/`、`film/`、`base/film.*`、`gui/`。 |
| GPU 内存、队列或 BVH 性能 | `gpu/`、`integrators/wavefront_path.*`、`accelerator/hlbvh.*`；检查 host/device 数据流与同步位置。 |

定位到具体实现后，沿“场景参数入口 → 工厂与分发 → 实现 → 积分器调用”核对连接关系。涉及新 `.cu` 文件时同步检查根 `CMakeLists.txt`；涉及功能支持时以实际分支为准，不从目录名推断完整 PBRT-v4 兼容性。
