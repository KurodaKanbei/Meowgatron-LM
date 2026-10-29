# 数据准备

第一版使用 **FineWeb-Edu 的 sample-10BT** 与 **GPT-2 tokenizer**。
数据准备在训练之前完成，运行实验时只读取本地文件。

## 来源与固定版本

| 内容 | 来源 | Revision |
|---|---|---|
| 英文预训练文本 | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| GPT-2 byte-level BPE | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |

sample-10BT 共 14 个 Parquet 分片，约 26.56 GiB，包含 9,672,101 篇文档。
上游 `token_count` 合计 9,972,918,177；本地加入 EOS 后的 token 数会略有变化。
GPT-2 词表大小为 50,257，`<|endoftext|>` ID 为 50,256。

数据许可、署名和适用条款见 FineWeb-Edu 数据卡；tokenizer 来源及许可见 GPT-2 模型卡。

## 下载到自己的存储

先安装下载工具，然后选择有足够容量的存储位置：

```bash
python -m pip install huggingface_hub
export MEOWGATRON_DATA_ROOT=/path/to/meowgatron-data

hf download HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --include 'sample/10BT/*.parquet' README.md \
  --local-dir "$MEOWGATRON_DATA_ROOT/fineweb-edu-10bt"

hf download openai-community/gpt2 \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --include tokenizer.json tokenizer_config.json vocab.json merges.txt README.md \
  --local-dir "$MEOWGATRON_DATA_ROOT/tokenizers/gpt2"
```

默认 YAML 通过 `data.root: ${oc.env:MEOWGATRON_DATA_ROOT,../data}` 读取环境变量，
未设置时使用仓库的 `data/` 目录。也可以在 YAML 中直接填写路径，或通过
`data.root=/path/to/meowgatron-data` 临时覆盖。相对路径以 YAML 所在目录为基准。
预期目录结构如下：

```text
meowgatron-data/
├── fineweb-edu-10bt/
│   └── sample/10BT/
│       ├── 000_00000.parquet
│       └── ...
└── tokenizers/gpt2/
    └── tokenizer.json
```

实验还会读取可选的 `fineweb-edu-10bt/manifest.json`，将其中的 revision 和
manifest 指纹记录到 `run.json`。当前环境的持久化数据副本已经包含完整的来源和校验清单。

## 样本语义

1. 按文件名排序读取 Parquet，并保留文件内的文档顺序。
2. 用本地 tokenizer 编码 `text`，每篇文档追加一个 EOS。
3. 连续拼接 token，取长度为 `seq_len + 1` 的窗口。
4. `inputs = window[:-1]`，`targets = window[1:]`。标签只在这里错位一次。
5. 下一个窗口前进 `seq_len` 个 token，保持预测目标连续。

窗口允许跨文档，EOS 标记边界；attention 使用普通 causal mask。
末尾不足一个窗口的 token 和不足一个 batch 的样本会丢弃。
当前只准备训练数据，不划分 eval，不做 shuffle，也不支持多 worker 读取。

分词使用流式读取，每次处理一小批文档。后续离线 tokenization、shuffle 和多进程读取
可以分别引入，并与这一版的数据语义核对。
