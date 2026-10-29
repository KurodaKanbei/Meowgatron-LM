<p align="center">
  <img src="assets/meowgatron.svg" alt="Meowgatron-LM，一只橘猫和一个小小的训练内核" width="900">
</p>

Meowgatron-LM 是一个从 PyTorch native 起步的大模型训练框架，希望为国产卡提供简洁、易用的开源训练实现。
同一份模型和训练代码运行在 NVIDIA GPU 与 Ascend NPU 上，用实验比较精度和性能。

我们遵循 **least necessary**原则：只实现当前需要的能力，一点点做加法。

## 开始训练

默认配置训练约 0.5B 参数的 Meow Transformer，使用 FineWeb-Edu 10BT、GPT-2 tokenizer 和 bf16。
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

选择对应设备的命令启动训练：

```bash
# NVIDIA GPU
python -m examples.meow --config configs/meow.yaml device=cuda:0

# Ascend NPU
python -m examples.meow --config configs/meow.yaml device=npu:0
```

用 TensorBoard 查看 loss、iteration time 和吞吐：

```bash
tensorboard --logdir runs
```
