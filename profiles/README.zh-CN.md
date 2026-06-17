# Profile 导引

语言：[English](README.md) | 简体中文

这里是 vLLM Tesla T10 Definitive 自带的启动 profile。一个 profile 只是运行参数
的 `.env` 预设，不包含模型权重路径；权重目录通过 `launcher.sh` 或
`MODEL_DIR=...` 单独选择。

Tesla T10 路线按张量并行规模组织在 `tp<N>/` 下。启动时 `GPU_DEVICES` 与
`TP_SIZE` 必须一致（例如 TP=4 对应 `GPU_DEVICES=0,1,2,3`）。

## Tesla T10 profile（experimental）

面向 Tesla T10 16GB 的起始预设。吞吐数字**尚未**验证；推广任何路线前请先按
[Tesla T10 KV 吞吐 Sweep](../docs/qwen36-kv-throughput-sweep-t10.zh-CN.md) 跑
完整 benchmark。

测试权重目标：GPTQ-INT4，约 19G。

| Profile | TP | 兼容模式 | 上下文 | KV | MTP | 状态 |
|---|---|---:|---|---:|---:|---|
| `qwen27b/tp2/normal/int4/int8kv-96K-mtp3-text-only.env` | 2 | normal | 96K | INT8 | 3 | experimental |
| `qwen27b/tp2/fast/int4/tqk8v4-128K-mtp3-text-only.env` | 2 | fast | 128K | TQK8V4 | 3 | experimental |
| `qwen27b/tp4/normal/int4/fp16kv-128K-mtp3-text-only.env` | 4 | normal | 128K | FP16 | 3 | experimental |
| `qwen27b/tp4/normal/int4/int8kv-256K-mtp3-text-only.env` | 4 | normal | 256K | INT8 | 3 | experimental |

目录结构：

```text
profiles/
  templates/
  qwen27b/
    tp2/
      normal/int4/
      fast/int4/
    tp4/
      normal/int4/
    normal/          # 2080 Ti 历史验证路线
      fp8/
      int4/
    fast/
      fp8/
      int4/
    user/
```

启动模式：

- `safe`：保守回退模式，优先保证可用性。
- `normal`：推荐日常模式，适合稳定部署。
- `fast`：高性能模式，适合追求更高吞吐的场景。
- `aggressive`：更加激进的模式，性能与质量风险最高。

`profiles/templates/` 存放可选 chat template 预设。它们通过 launcher 作为全局
服务设置选择；具体 route profile 不保存 chat template、GPU、端口、reasoning
默认值或工具调用默认值。

文件名描述路线：

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV 精度定位：

- `fp16kv`：质量路线。
- `int8kv`：容量 / 平衡路线。
- `tqk8v4`：TurboQuant K8V4 压缩路线，用于 `fast` profile。

## 历史验证 profile（双 RTX 2080 Ti 22GB，TP=2）

下表保留原 2080 Ti fork 的 benchmark 证据。在 Tesla T10 上**不要**直接部署，
除非重新验证。

### FP8

测试权重：Jackrong/Qwopus3.6-27B-v2-FP8，约 29G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

### AWQ/GPTQ-INT4

测试权重：GPTQ-INT4，约 19G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 每工作区 250K | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 每工作区 250K | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |

表中的吞吐性能是在 `4096/128` 口径下测试，格式为
`prefill tok/s / decode tok/s`。测试前会先跑中文质量 smoke；质量失败的路线不作为
profile 保留。
