"""Calibration data: fixed-length token windows from a text corpus.

Calibration passes (summaries, activation ranges, SmoothQuant) only need a
``DataLoader`` of ``{"input_ids", "attention_mask"}`` samples. This module builds
it from a token stream; ``wikitext2_loader`` is the WikiText-2 shortcut used by
the scripts, examples and benchmarks. The Hugging Face ``datasets`` package is
imported only when WikiText-2 is actually loaded.
"""
import random

import torch
from torch.utils.data import DataLoader

WIKITEXT2 = ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1")


def token_windows_loader(token_ids, num_samples: int, seq_len: int, *, seed: int = 42,
                         sequential: bool = False, labels: bool = False) -> DataLoader:
    """``DataLoader`` (batch size 1) of ``seq_len``-token windows of ``token_ids``.

    Random windows (seeded) by default, as used for calibration; ``sequential=True``
    takes consecutive non-overlapping windows from the start (the usual perplexity
    protocol). ``labels=True`` adds ``labels = input_ids`` for loss-based calibration.
    At most ``len(token_ids) // seq_len`` windows are returned.
    """
    count = min(num_samples, len(token_ids) // seq_len)
    rng = random.Random(seed)
    starts = [i * seq_len for i in range(count)] if sequential else \
             [rng.randint(0, len(token_ids) - seq_len - 1) for _ in range(count)]
    examples = []
    for start in starts:
        chunk = torch.tensor(token_ids[start:start + seq_len], dtype=torch.long)
        example = {"input_ids": chunk, "attention_mask": torch.ones(seq_len, dtype=torch.long)}
        if labels:
            example["labels"] = chunk.clone()
        examples.append(example)
    return DataLoader(examples, batch_size=1, shuffle=False)


def wikitext2_token_ids(tokenizer, split: str = "train") -> list[int]:
    """Token ids of the whole WikiText-2 ``split`` (non-empty pages joined by blank lines)."""
    from datasets import load_dataset

    pages = load_dataset(*WIKITEXT2, split=split)["page"]
    corpus = "\n\n".join(t for t in pages if isinstance(t, str) and t.strip())
    return tokenizer(corpus, truncation=False, padding=False, return_attention_mask=False)["input_ids"]


def wikitext2_loader(tokenizer, num_samples: int = 128, seq_len: int = 512, *, split: str = "train",
                     seed: int = 42, sequential: bool = False, labels: bool = False) -> DataLoader:
    """WikiText-2 windows: see :func:`token_windows_loader` for the options."""
    return token_windows_loader(wikitext2_token_ids(tokenizer, split), num_samples, seq_len,
                                seed=seed, sequential=sequential, labels=labels)


__all__ = ["token_windows_loader", "wikitext2_token_ids", "wikitext2_loader"]
