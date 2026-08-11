# ks-ascend-survey

在一台华为昇腾（Ascend）机器上，一条命令测出：软件栈实际装了什么、Triton 在这里能做什么
以及代价多少、十个 KernelSwift 竞赛算子的参考实现基线是多少。

写这套东西的场景是：机器只能人工登录，无法从开发机脚本化访问，所以需要一个上传后
自己跑完、产出一份可直接带回的报告的工具。

算子参考实现来自 [DeepLink-org/DLBlas](https://github.com/DeepLink-org/DLBlas) 的
KernelSwift 赛题，`auto_bench.py` 是其官方评分脚本，**未做任何修改**。

## 用法

```bash
git clone <this repo> ks-ascend-survey
cd ks-ascend-survey
bash run_all.sh
```

产出：

- `results/REPORT.txt` — 三个阶段的完整汇总，这是要带回去的文件
- `results/stage*.log` — 各阶段独立日志
- `ks-ascend-results.tar.gz` — 上面两者的打包

预计半小时到两小时，主要取决于 Triton 编译速度。**报告每完成一个阶段就重写一次**，
所以中途 `Ctrl-C` 也没关系，已完成的部分就在文件里。

只跑其中某些阶段：

```bash
bash run_all.sh --stage 1
bash run_all.sh --stage 2 --stage 3
```

## 三个阶段

**1. environment** — Python / torch / torch_npu / triton 版本，CANN 与驱动信息，
`npu-smi info`，设备属性，AI Core 数量。刻意用老语法写成，即使解释器版本过低、
后面的阶段跑不起来，它也能跑完并说明原因。

**2. triton** — Triton 能否编译运行；单次 launch 的开销（对比 torch 原生与一次内存分配）；
`CompiledKernel.run` 能否直接调用；拷贝带宽；`tl.dot` 对比厂商 BLAS；
各 dtype 的逐元素吞吐；以及三个决定移植策略的问题：

- **并发执行的 program 数量**，也就是 AI Core 数。这是最关键的一个数字：DaVinci 是
  几十个 AI Core 而不是几千个 SM，「一个 program 处理一小块、剩下交给调度器」的
  CUDA 式分解在这里不成立，所有 block 尺寸的选择都取决于它。torch API 查不到，只能实测。
- **单个 program 能装下的最大 tile**，也就是片上缓冲的上限，决定算子能否常驻还是必须
  拆成多次 launch。
- **`num_warps` 还有没有意义**，被忽略是无用参数，被拒绝则是编译错误。

**3. baselines** — 十个算子参考实现的耗时，以及每个输入张量的 shape / dtype / **stride
和内存格式**。记 stride 的理由：曾经在别的后端上遇到过张量搬到设备时内存格式被静默改变，
导致 kernel 的守卫拒绝输入、退回参考实现、拿到 1.00x，而这在耗时表里完全看不出来。

## 几处设计取舍

**每个用例跑在独立子进程里。** 不是过度防御：在此前测过的一个后端上，`range()` 循环里
携带 tile 做归约不是抛异常，而是直接 SIGSEGV 杀掉进程。不隔离的话一次崩溃会带走整轮
测量；现在会记成 `*** SIGSEGV ***` 然后继续下一个。超时同样处理。

**`auto_bench.py` 一个字节没改。** 它本身已经支持昇腾——`_iter_accelerators()` 覆盖
`cuda/npu/mlu/gcu`，计时用 `perf_counter` 加设备同步而不是 CUDA event。唯一的问题是
`torch.npu` 要等 `import torch_npu` 之后才存在，而它自己不 import。所以用
`sitecustomize.py` 在解释器启动时完成注册（`run_all.sh` 把仓库目录放进 `PYTHONPATH`），
不动评分脚本本身。

**部分赛题源码硬编码了 `device="cuda"`。** `auto_bench.py` 的设备字面量重写只处理
`npu → 其他后端`，在昇腾上会直接跳过，所以这些文件原样跑会报错。阶段 3 在生成
`baselines/*.py` 时替换这个字面量，参考实现和待测实现同等处理，对比仍然公平；
转换后的文件留在 `baselines/` 下可直接查。这个改写有测试：

```bash
python3 tests/test_device_rewrite.py
```

不需要 torch 或加速器，任何机器上都能跑。

## 环境前提

- Python ≥ 3.10（`auto_bench.py` 用了 `float | None` 这类语法）
- `torch` 与 `torch_npu`
- `triton-ascend`
- CANN。`run_all.sh` 会在存在时自动 source
  `/usr/local/Ascend/ascend-toolkit/set_env.sh`——非交互 shell 里没加载这个的话，
  症状是 `import torch_npu` 失败，看着像装坏了，其实只是环境没加载。

阶段 1 会检查上述各项并在没有可用加速器时停下，不会继续产生无意义的日志。
