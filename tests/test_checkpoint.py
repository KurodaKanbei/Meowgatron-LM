import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]


def make_experiment(tmp_path):
    shards = tmp_path / "data/fineweb-edu-10bt/sample/10BT"
    shards.mkdir(parents=True)
    texts = [
        " ".join("abcdef"[(i + offset) % 6] for i in range(length))
        for offset, length in enumerate((81, 75, 90))
    ]
    # 249 tokens including EOS -> 31 samples of length 8. With GBS=8,
    # rank 0 can form a fourth local update but rank 1 cannot: both must stop at 3.
    pq.write_table(
        pa.table({"text": texts[:2]}), shards / "a.parquet", row_group_size=1
    )
    pq.write_table(pa.table({"text": texts[2:]}), shards / "b.parquet")
    tokenizer_dir = tmp_path / "data/tokenizers/gpt2"
    tokenizer_dir.mkdir(parents=True)
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "<|endoftext|>": 1,
                **{c: i + 2 for i, c in enumerate("abcdef")},
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    config = tmp_path / "meow.yaml"
    config.write_text("""device: cpu
seed: 31
data:
  root: data
  micro_batch_size: 2
  global_batch_size: 8
model:
  vocab_size: 8
  hidden_size: 256
  num_heads: 4
  mlp_hidden_size: 1024
  max_seq_len: 8
  num_layers: 12
training:
  max_steps: null
  log_interval: 1
checkpoint:
  interval: 2
tensorboard:
  enabled: true
""")
    return config


def run_entry(config, output, dp, max_steps="null", resume=None):
    command = [sys.executable, "-m"]
    if dp > 1:
        command += [
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={dp}",
            "--module",
        ]
    command += [
        "examples.meow",
        "--config",
        str(config),
        f"output_root={output}",
        f"training.max_steps={max_steps}",
    ]
    if resume is not None:
        command.append(f"checkpoint.resume={resume}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        timeout=90,
    )
    assert result.returncode == 0, result.stdout
    return next(output.iterdir())


@pytest.mark.parametrize("dp", [1, 2])
def test_new_process_resume_matches_uninterrupted_training(tmp_path, dp):
    config = make_experiment(tmp_path)
    full = run_entry(config, tmp_path / "full", dp)
    first = run_entry(config, tmp_path / "first", dp, max_steps=1)
    resumed = run_entry(
        config, tmp_path / "resumed", dp, resume=first / "checkpoint.pt"
    )
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    # Includes optimizer moments/step and every rank's data position.
    for key in ("model", "optimizer"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in ("progress", "data"):
        assert actual[key] == expected[key]
    assert actual["progress"] == {"steps": 3, "tokens_seen": 3 * 8 * 8}
    assert len(actual["data"]) == dp
