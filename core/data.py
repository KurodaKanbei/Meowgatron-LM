"""Stream Parquet text into fixed-length, next-token training samples."""

from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset


class TokenDataset(IterableDataset):
    def __init__(self, files: Iterable[Path], tokenizer: Tokenizer, seq_len: int):
        super().__init__()
        self.files = tuple(sorted(Path(path) for path in files))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")

    def __iter__(self):
        pending = []
        for path in self.files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=64, columns=["text"]):
                texts = batch.column(0).to_pylist()
                for encoded in self.tokenizer.encode_batch(
                    texts, add_special_tokens=False
                ):
                    pending.extend(encoded.ids)
                    pending.append(self.eos_id)
                    offset = 0
                    while len(pending) - offset >= self.seq_len + 1:
                        window = torch.tensor(
                            pending[offset : offset + self.seq_len + 1],
                            dtype=torch.long,
                        )
                        yield window[:-1], window[1:]
                        offset += self.seq_len
                    pending = pending[offset:]
        # A final incomplete window is dropped; no padding or loss mask is needed.
