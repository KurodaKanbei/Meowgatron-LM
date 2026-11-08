<p align="center">
  <img src="assets/meowgatron.svg" alt="Meowgatron-LM，一只橘猫和一个小小的训练内核" width="900">
</p>

Meowgatron-LM 是一个从 PyTorch native 起步的大模型训练框架，希望为国产卡提供简洁、易用的开源训练实现。
同一份模型和训练代码运行在 NVIDIA GPU 与 Ascend NPU 上，用实验比较精度和性能。

我们遵循 **least necessary**原则：只实现当前需要的能力，一点点做加法。

## 开始训练

默认配置训练 16 层、32 个 attention head、约 0.38B 参数的 Meow Transformer，使用 FineWeb-Edu 10BT、GPT-2 tokenizer 和 bf16。
模型、batch、学习率等配置都在 [configs/meow.yaml](configs/meow.yaml) 中调整。

使用 Python 3.10+，准备好对应设备的运行环境，选择一项安装：

```bash
# NVIDIA GPU
python -m pip install -e '.[cuda]'

# Ascend NPU（需要配套的 torch、torch_npu、CANN 与驱动）
python -m pip install -e '.[npu]'
```

按[数据准备](docs/data.md)下载语料和 tokenizer，然后指定数据目录：

```bash
export MEOWGATRON_DATA_ROOT=/path/to/meowgatron-data
```

在 YAML 中设置 `device`：NVIDIA GPU 使用 `cuda:0`，Ascend NPU 使用 `npu:0`。
单卡训练：

```bash
python -m examples.meow --config configs/meow.yaml
```

八卡数据并行训练：

```bash
torchrun --standalone --nproc_per_node=8 -m examples.meow --config configs/meow.yaml
```

各进程按 `LOCAL_RANK` 绑定设备。
`global_batch_size` 是所有 DP 副本合计的 batch，须为 `micro_batch_size × DP` 的整数倍。
950DT 多卡训练使用 HCCL 9.2.0，包含 [OneShot 规约顺序修复](https://github.com/hicann/hccl/commit/f54ff6beb1dd3cda558481b839e2979b9031d3e9)。

流水线并行与激活重算也在 YAML 中配置。例如八卡使用 PP4 × DP2：

```yaml
distributed:
  pipeline_layout: [4, 4, 4, 4]
training:
  recompute: selective
  recompute_modules: [attention, mlp_act]
```

`pipeline_layout` 依次指定每个 stage 的 Transformer 层数，列表长度就是 PP 数；
也可写成 `[3, 5, 5, 3]` 等不均分布局。embedding 在首段，final norm 与 LM head 在末段。
流水线使用 1F1B 调度。`recompute` 可选 `none`、逐层重算的 `full`，
或通过 `recompute_modules` 选择区域的 `selective`。可选 `attention`（attention core）、
`norm`（block 内两处 RMSNorm）、`mlp_act`（GELU）和 `mlp`（整个 MLP）。
可以组合；`mlp` 已包含 `mlp_act`。GBS 按 `micro_batch_size × DP × 梯度累积次数` 计算，启动命令不变。

续训时，在 YAML 中填写 checkpoint 路径，使用相同的启动命令、卡数和 batch 配置：

```yaml
checkpoint:
  interval: 1000
  resume: /path/to/checkpoint.pt
```

`interval` 按参数更新次数计数，设为 `null` 时仅在正常结束时保存；从头训练将 `resume` 设为 `null`。
恢复会新建输出目录，日志的 step 和累计 token 数接续原值。
PP 训练将各 stage 保存到 `checkpoint/` 目录，续训时将 `resume` 指向这个目录，保持相同的 layout 与 DP 配置。
`training.max_steps` 是累计更新终点，设为 `null` 时训练到数据耗尽。
数据分配与读取位置的恢复见[数据准备](docs/data.md#样本语义)。

用 TensorBoard 查看 loss、iteration time 和吞吐：

```bash
tensorboard --logdir runs
```
