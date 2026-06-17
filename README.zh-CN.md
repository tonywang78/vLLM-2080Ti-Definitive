<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM Tesla T10 Definitive Edition

![vLLM Tesla T10 Definitive Edition 题图](docs/assets/vllm-2080ti-cover.jpg)

面向多张 Tesla T10 / SM75 推理的终极版 vLLM 运行时。

这是一个硬件定向的 vLLM fork，用来保存已经跑通的 Tesla T10 vLLM
栈：补丁源码、启动 profile、运行时说明和稳定环境记录。

Fork 发布版本：`v0.2.0`
基础 vLLM：`0.21.0`

核心目标：在多张 Tesla T10 16GB 上以可配置张量并行（TP=2/4/8）运行
Qwen3.6 27B。吞吐数字需在具体卡数与 PCIe 拓扑上重新 benchmark。

语言：[English](README.md) | 简体中文

![单请求实时测速演示](docs/assets/vllmspeed.gif)

## 💡 为什么用 Tesla T10 做 LLM 推理？

Tesla T10（TU102、SM75）是面向数据中心的 Turing 显卡，16GB GDDR6、约
403 GB/s 显存带宽。它与 RTX 2080 Ti 等同属 SM75，因此本 fork 可直接复用
已验证的 SM75 内核栈（Marlin、FlashQLA、FlashInfer、TurboQuant/INT8 KV、
MTP、CUDAGraph）。

| 指标 | Tesla T10 16GB | RTX 2080 Ti 22GB（旧验证目标） | 说明 |
|---|---:|---:|---|
| 架构 | TU102 / SM75 | TU102 / SM75 | 相同 CUDA arch（`7.5`） |
| 显存 | 16GB | 22GB（改显存） | T10 profile 使用更紧的 `GPU_UTIL` |
| 显存带宽 | 403 GB/s | 616 GB/s | 吞吐预期更低 |
| Tensor Core | 448 | 544 | 算力略低 |
| 互联 | 通常 PCIe P2P | 常见 NVLink | 多卡 TP 依赖 P2P |
| TDP | 150W | 250W | 更易部署多卡 |

本项目面向**多张 Tesla T10**，张量并行度可配置。TP=2 在 32GB 总显存上可跑
27B INT4/INT8KV；TP=4/TP=8 可换更长上下文，但 PCIe all-reduce 开销更大。

本 fork 的价值：在已验证的 SM75 运行时之上，调整 GPU 检测、profile 与文档，
适配 Tesla T10 多卡主机。

## 🧩 核心路线

服务形态：

- 本项目追求的是多张 Tesla T10 上的极限单并发性能：一个个人 agent 场景、
  一个足够强的 27B/31B 模型，以及所选 TP 规模下能稳定承载的最大实用上下文。
- 它不是多租户 serving 集群。多 agent 使用更适合作为排队式工作区隔离，
  而不是并行长 prefill 吞吐。长上下文并发在调好参数后可以安全排队，
  但多卡 TP profile 下实际会被 runtime scheduler 串行化。

状态：🟢 已验证支持；🟡 实验或部分支持；🔴 已知失败或明显退化；⚪ 非目标预设或尚未验证。

### Qwen3.6 27B 成熟主线

Qwen 系 27B 是这个 fork 的主要生产路线，在 FP8/INT4/NVFP4 权重、MTP、
FP16/INT8/TurboQuant KV、256K 原生上下文、YaRN 容量和图像多模态上覆盖最完整。

| 功能 | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin 权重路线 | 🟢 FP8/INT4/NVFP4 | 🟢 FP8/INT4/NVFP4 | 🟢 FP8/INT4/NVFP4 |
| MTP 解码 | 🟢 支持 | 🟢 支持 | 🟢 支持 |
| 原生 256K 上下文 | 🟢 支持 | 🟢 支持 | 🟢 支持 |
| YaRN 扩展 | ⚪ 非目标路线 | 🟢 支持 | ⚪ 非目标预设 |
| No-eager / CUDAGraph | 🟢 支持 | 🟡 部分支持 | 🟢 支持 |
| 快速 prefill 路线 | 🟢 FlashQLA / FlashInfer | 🟢 FlashQLA / FlashInfer | 🟢 FlashQLA / FlashInfer |
| 图像多模态 | 🟢 支持 | 🟢 支持 | 🟢 支持 |
| 当前预设状态 | 🟢 normal / fast / safe | 🟢 normal / safe | 🟢 fast |

### Gemma4 31B 实验路线

Gemma4 31B 保留为第二路线和实验路线。目前最值得继续推进的是官方 QAT
target 搭配对应的 QAT assistant，这条路线相比早期 Gemma 变体有更好的
FP16/default KV 空间。

| 功能 | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin 权重路线 | 🟢 GPTQ / QAT | 🟡 GPTQ / QAT | 🟡 GPTQ / QAT |
| MTP 解码 | 🟡 QAT assistant MTP3 | ⚪ 无预设 | ⚪ 无预设 |
| 实测上下文 | 🟡 约 170K KV 空间 | 🔴 初始化问题 | 🔴 容量不足 |
| No-eager / CUDAGraph | 🟢 支持 | 🟡 fallback 问题 | 🟡 admission 受限 |
| 快速 prefill 路线 | 🟢 FlashInfer | 🟡 FlashInfer | 🟡 FlashInfer |
| 图像多模态 | ⚪ 无已验证预设 | ⚪ 无已验证预设 | ⚪ 无已验证预设 |
| 当前预设状态 | 🟡 仅实验 | ⚪ 无预设 | ⚪ 无预设 |

## 🧪 已测试模型权重

这一节记录 checkpoint 级别的验证结果。这里的标准比“vLLM 能加载”更严格：
支持表示可以启动并生成；推荐表示在 Tesla T10 上、对应 TP 规模下同时具备
有意义的速度 / 上下文权衡。

| 模型路线 | 权重路线 | 模型卡 | 状态 |
|---|---|---|---|
| Qwen3.6 27B FP8 | FP8 | [Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)<br>[Jackrong/Qwopus3.6-27B-v2-FP8](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-FP8) | 🟢 推荐 |
| Qwen3.6 27B AWQ | AWQ-INT4 | [QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ)<br>[mconcat/Qwopus3.6-27B-v2-AWQ-4bit](https://huggingface.co/mconcat/Qwopus3.6-27B-v2-AWQ-4bit) | 🟢 推荐 |
| Qwen3.6 27B GPTQ | GPTQ-INT4 | [llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) | 🟢 推荐 |
| Qwen3.6 27B NVFP4 | NVFP4 | [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 🟡 支持 |
| Qwen3.6 27B Quark INT8 | Quark-INT8 | [nameistoken/Qwen3.6-27B-Quark-W8A8-INT8](https://huggingface.co/nameistoken/Qwen3.6-27B-Quark-W8A8-INT8) | 🟡 支持 |
| Qwen3.6 27B AutoRound | AutoGPTQ-INT8 | [Minachist/Qwen3.6-27B-INT8-AutoRound](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound)<br>[Minachist/Qwen3.6-27B-INT8-AutoRound W8A16-GS128](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound/tree/W8A16-GS128) | 🟡 支持 |
| Gemma4 31B QAT | QAT + QAT assistant draft | [google/gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)<br>[google/gemma-4-31B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant) | 🟡 支持 |
| Gemma4 31B GPTQ | GPTQ-INT4 + assistant draft | [ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ](https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ) | 🟡 支持 |

## 🛠️ 目标硬件与运行环境

- 已验证 GPU profile：多张 Tesla T10 16GB，SM75，PCIe P2P，tensor parallel
  size 2/4/8（可配置；TP 必须等于所选 GPU 数量）
- 已验证主机系统：Ubuntu 22.04/24.04 LTS 或 Debian 12，Linux kernel 6.x
- CUDA/PyTorch：CUDA 12.8，`torch 2.11.0+cu128`
- Fork 发布版本：`v0.2.0`
- 基础 vLLM：`0.21.0`
- 仓库身份：`vllm-tesla-t10-definitive`
- 运行时身份：`vllm-sm75-tesla-t10-cu128`
- 兼容目标：NVIDIA Turing / SM75 显卡。Tesla T4 可编译运行，但不是主要
  profile 目标。其它 Turing 显卡仍需要按显存容量、P2P/NVLink 行为、模型
  head_dim、KV dtype、CUDAGraph/MTP 设置重新验证 profile。

## 🚀 如何使用

源码 checkout 后分两步使用。

1. 编译 runtime：

```bash
./build.sh
```

`build.sh` 会创建本地 `.venv`、安装依赖、编译 CUDA 扩展，并在结束时明确提示
成功或失败，同时给出 build log 路径。

2. 启动并管理服务：

```bash
./launcher.sh
```

`launcher.sh` 是交互式服务管理器。你可以在菜单里选择 checkpoint 目录、套用或
修改 profile、选择 `safe` / `normal` / `fast` 模式、选择 GPU/TP、设置端口、
切换仅本地或局域网访问、配置 chat template 和工具调用、启动服务、停止服务，
也可以保存自定义 profile。

支持 OpenAI-compatible 工具调用。launcher 提供自动工具选择、tool parser 选择
和严格结构化 tool 输出等全局运行参数。

启动成功后，状态区会显示 `RUNNING`、服务模型名、PID、OpenAI-compatible API
地址、日志文件位置，以及 vLLM 能上报时的 cache 容量。

非交互启动示例：

```bash
MODEL_DIR=/path/to/qwen-or-gemma-checkpoint \
PROFILE=qwen27b/tp2/normal/int4/int8kv-96K-mtp3-text-only.env \
MODE=normal \
PORT=8000 \
SERVICE_SCOPE=lan \
GPU_DEVICES=0,1,2,3 \
TP_SIZE=4 \
./launcher.sh --non-interactive
```

Profile 只声明兼容模式，不再提供推荐启动模式。需要指定模式时，显式传
`MODE=safe`、`MODE=normal` 或 `MODE=fast`；launcher 会根据 profile 做二次校验。

3. 更新已有 checkout：

```bash
./update.sh
```

`update.sh` 会检查 GitHub 最新 Release 和本地 fork 版本；有新版本时下载
release archive，并保留本地 `.venv`、`.deps`、日志、结果、缓存和用户 profile
等运行状态，更新完成后会询问是否立刻运行 `build.sh`。

## 🧭 Profile 与推荐路线

从 [Profile 导引](profiles/README.zh-CN.md) 开始选。Tesla T10 路线位于
`profiles/<model>/tp<N>/<mode>/<weight>/<route>.env`，例如
`qwen27b/tp2/normal/int4/int8kv-96K-mtp3-text-only.env` 和
`qwen27b/tp4/normal/int4/fp16kv-128K-mtp3-text-only.env`。

原 2080 Ti 验证路线仍保留在 `profiles/qwen27b/normal/...` 供参考，不作为
Tesla T10 推荐部署预设。

可用模式：

- `normal`：默认推荐模式，适合日常生产部署。
- `fast`：高性能模式，但不推荐用于稳定生产部署。
- `aggressive`：更加激进的模式，性能与质量风险最高。
- `safe`：安全模式，速度较慢，但输出质量高度稳定，用于排障。

后续目标 profile 和实测进度记录在
[Profile 蓝图草案](docs/profile-blueprint.zh-CN.md)。

## 🚀 MTP 与 KV 精度

优先使用项目自带 profile，不建议一开始手动调 MTP 和 KV 参数。每条路线的
MTP 已按当前实测选择了更适合部署的值。KV 先按目标选择：FP16/default KV
追求质量，INT8 KV 用于平衡型长上下文服务，TurboQuant K8V4 用于 fast
压缩路线。

详细 benchmark 记录见
[MTP 任务敏感性](docs/mtp-task-sensitivity.md) 和
[Qwen3.6 KV 吞吐 Sweep](docs/qwen36-kv-throughput-sweep.zh-CN.md)。

## ❓ 硬件 Q&A

**Q：需要什么样的卡间互联？**

A：多张 Tesla T10 张量并行的底线是 PCIe P2P。T10 没有 NVLink。推广 TP=4/8
路线前请用 `nvidia-smi topo -m` 确认拓扑。P2P 不可用时，多卡 TP 可能变慢或
无法启动。

**Q：需要很强的 CPU 或很多内存吗？**

A：不需要高端 CPU，但更推荐单核性能强、平台延迟低的现代 CPU。更多内存主要
帮助 build、下载和 compile cache。由于 vLLM 有 Python / 服务化控制面，很老的
CPU 平台可能更适合 llama.cpp 这类极简 C++ runtime。

**Q：哪些 Turing 显卡值得尝试？Tesla T4 呢？**

A：主要验证目标是**多张 Tesla T10 16GB**，TP 等于所选 GPU 数。Tesla T4 与
T10 同为 SM75，可跑小模型或短上下文，但不是 27B/31B 主路线。原双 2080 Ti
22GB 路线见 `profiles/qwen27b/normal/...` 仅供参考。

**Q：如何选择 GPU 和 TP？**

A：默认 `TARGET_GPU_PATTERN=t10`，launcher 会自动选中所有 Tesla T10；也可显式
设置 `GPU_DEVICES`。`TP_SIZE` 必须等于所选 GPU 数量。可在菜单第 3 项或非交互
模式下同时传入。

**Q：已验证的 CUDA、PyTorch 和驱动版本是什么？**

A：已验证 runtime 是 CUDA 12.8 + `torch 2.11.0+cu128`，参考验证主机使用
NVIDIA driver `590.48.01`。请使用支持目标 GPU、并且兼容该 CUDA runtime 的
较新 NVIDIA driver。不要随意混用 build/runtime 假设：PyTorch CUDA 版本、
本地 CUDA toolkit、FlashInfer/FlashQLA 构建和启动 profile 应保持一致。

**Q：还有哪些硬件风险需要注意？**

A：散热、供电稳定性，以及给模型文件和 compile cache 留够 SSD 空间。长 prefill
或反复 CUDAGraph/AOT 编译时，降频很容易伪装成软件性能回退。

## 🔗 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双
  2080 Ti 模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。
  本仓库则聚焦于 vLLM 运行时源码、补丁和启动配置本身。

## 🙏 致谢 / 上游项目

本仓库是基于上游 [vLLM](https://github.com/vllm-project/vllm) 的硬件定向
fork，遵循 Apache-2.0 license。仓库保留上游项目结构，并加入面向 Tesla T10
多卡 / SM75 路线的本地运行时补丁、启动 profile 和验证记录。

当前 runtime 使用或集成的加速组件包括：

- [vLLM](https://github.com/vllm-project/vllm)：基础推理引擎和 serving
  框架。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：vLLM 使用的
  attention、sampling 和量化 kernel 路线。
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)：上游 FlashQLA
  Gated DeltaNet / Qwen3.5 linear-attention 实现。
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)：
  面向 SM70/SM75 的适配版本，已验证 Qwen3.6 prefill profile 会用到。
- TurboQuant、Marlin、CUTLASS、Triton 以及 vLLM
  相关加速 kernel：这些都是已有开源加速工作，本项目将它们整合、适配并在
  目标硬件上验证。

本仓库不会严格遵循上游主线vLLM的版本更新节奏，但其每次版本迭代所引入的功能与修复，都将
在SM75架构视角下重新评估.
