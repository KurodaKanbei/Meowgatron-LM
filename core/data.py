"""A resumable, sequential token stream, partitioned by packed sample number."""

from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset


class TokenDataset(IterableDataset):
    def __init__(
        self,
        files: Iterable[Path],
        tokenizer: Tokenizer,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.files = tuple(sorted(Path(path) for path in files))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")
        self.rank = rank
        self.world_size = world_size
        self.file_index = 0
        self.row_group = 0
        self.row = 0
        self.pending = []
        self.offset = 0
        self.sample_index = 0

    def state_dict(self):
        # The cursor points to the next document; pending holds its predecessor's
        # unconsumed tokens, including the one-token overlap between windows.
        return {
            "file_index": self.file_index,
            "row_group": self.row_group,
            "row": self.row,
            "pending": self.pending[self.offset :],
            "sample_index": self.sample_index,
        }

    def load_state_dict(self, state):
        self.file_index = state["file_index"]
        self.row_group = state["row_group"]
        self.row = state["row"]
        self.pending = state["pending"].copy()
        self.offset = 0
        self.sample_index = state["sample_index"]

    def _documents(self):
        while self.file_index < len(self.files):
            parquet = pq.ParquetFile(self.files[self.file_index])
            while self.row_group < parquet.num_row_groups:
                skipped = 0
                for batch in parquet.iter_batches(
                    batch_size=64, row_groups=[self.row_group], columns=["text"]
                ):
                    # Resume within this row group without tokenizing earlier rows.
                    if skipped + len(batch) <= self.row:
                        skipped += len(batch)
                        continue
                    texts = batch.column(0).slice(self.row - skipped).to_pylist()
                    for encoded in self.tokenizer.encode_batch(
                        texts, add_special_tokens=False
                    ):
                        self.row += 1
                        yield encoded.ids
                    skipped += len(batch)
                self.row_group += 1
                self.row = 0
            self.file_index += 1
            self.row_group = 0

    def __iter__(self):
        documents = self._documents()
        while True:
            while len(self.pending) - self.offset < self.seq_len + 1:
                self.pending = self.pending[self.offset :]
                self.offset = 0
                try:
                    self.pending.extend(next(documents))
                except StopIteration:
                    return
                self.pending.append(self.eos_id)

            sample_index = self.sample_index
            start = self.offset
            self.offset += self.seq_len
            self.sample_index += 1
            # Advance all state before yielding: a checkpoint taken while the
            # generator is suspended must continue with the next sample.
            if sample_index % self.world_size == self.rank:
                window = torch.tensor(
                    self.pending[start : start + self.seq_len + 1], dtype=torch.long
                )
                yield window[:-1], window[1:]
