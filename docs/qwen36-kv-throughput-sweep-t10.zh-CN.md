# Qwen3.6 27B KV / 吞吐 Sweep（Tesla T10）

语言：[English](qwen36-kv-throughput-sweep-t10.md) | 简体中文

本文说明如何在 Tesla T10 16GB 主机上重新验证 Qwen3.6 27B 路线。方法与
[Qwen3.6 KV 吞吐 Sweep](qwen36-kv-throughput-sweep.zh-CN.md) 相同，但容量
假设不同于旧版双 RTX 2080 Ti 22GB 验证目标。

## 硬件假设

- GPU：Tesla T10 16GB，SM75（TU102）
- 互联：TP > 1 时需要 PCIe P2P
- 张量并行：TP 必须等于所选 GPU 数量（`GPU_DEVICES` 长度）
- 起始 profile：见 `profiles/qwen27b/tp2/...` 与 `profiles/qwen27b/tp4/...`

## 容量说明（benchmark 前）

| TP | 总显存 | 起始路线 | 状态 |
|---:|---:|---|---|
| 2 | 32GB | INT8KV 96K、TQK8V4 128K | experimental |
| 4 | 64GB | FP16KV 128K、INT8KV 256K | experimental |
| 8 | 128GB | 尚未提供 | not promoted |

27B GPTQ-INT4 权重约 19GB。TP=2 在 16GB 单卡上 FP16 KV 长上下文空间很紧；
在收集 benchmark 证据前优先 INT8KV 或 TQK8V4。

## Sweep 步骤

1. 确认拓扑：`nvidia-smi topo -m`
2. 编译 runtime：`./build.sh`
3. 从 `profiles/qwen27b/tp<N>/...` 选择起始 profile
4. 显式指定 GPU/TP 启动：

```bash
MODEL_DIR=/path/to/gptq-int4-checkpoint \
PROFILE=qwen27b/tp4/normal/int4/int8kv-256K-mtp3-text-only.env \
MODE=normal \
GPU_DEVICES=0,1,2,3 \
TP_SIZE=4 \
./launcher.sh --print-config
```

5. 跑容量 smoke（OOM 边界），再在 `4096/128` 口径下测吞吐
6. 记录 prefill/decode tok/s 与中文质量 smoke 结果
7. 仅 capacity 与质量均通过的路线可 promote 为 `validated`

## 推广规则

- 无 benchmark 数字的路线保持 `experimental`
- 不要直接复制 2080 Ti 吞吐表；16GB 显存与 PCIe 带宽都会改变容量与 decode 速度
- promote 为 `validated` 时同步更新 `profiles/README.zh-CN.md`
