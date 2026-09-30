"""Dataset for HERO training.

Samples are stored as JSON Lines, one sample per line::

    {"id": "...", "source_id": "...", "text": "...", "label": "translated_fr",
     "generator": "Meta-Llama-3-8B-Instruct", "dataset": "goodnews", "split": "train"}

``source_id`` identifies the human-written article a sample was derived from, so
that all versions of one article (human, paraphrased, translated, ...) end up in
the same train/validation split.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger
from torch.utils.data import Dataset


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_samples(files, classes, root=None):
    """Load samples from JSONL files, keeping only the requested classes."""
    samples = []
    skipped = Counter()
    for file in files:
        path = Path(root) / file if root is not None else Path(file)
        for record in read_jsonl(path):
            if record["label"] not in classes:
                skipped[record["label"]] += 1
                continue
            if not record.get("text", "").strip():
                skipped["<empty>"] += 1
                continue
            samples.append(record)
    if skipped:
        logger.info(f"Skipped samples: {dict(skipped)}")
    return samples


def split_by_source(samples, val_fraction, seed):
    """Split samples into train/val so that no source article is in both."""
    sources = sorted({s["source_id"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_val = int(round(len(sources) * val_fraction))
    val_sources = set(sources[:n_val])
    train = [s for s in samples if s["source_id"] not in val_sources]
    val = [s for s in samples if s["source_id"] in val_sources]
    return train, val


def cap_per_class(samples, max_per_class, seed):
    """Randomly keep at most ``max_per_class`` samples of each class."""
    if max_per_class is None:
        return samples
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["label"]].append(s)
    rng = random.Random(seed)
    capped = []
    for label in sorted(by_class):
        group = by_class[label]
        rng.shuffle(group)
        capped.extend(group[:max_per_class])
    return capped


class TextClassificationDataset(Dataset):
    def __init__(self, samples, classes):
        self.samples = samples
        self.class_to_idx = {name: i for i, name in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {"text": s["text"], "label": self.class_to_idx[s["label"]]}

    def class_counts(self):
        return dict(Counter(s["label"] for s in self.samples))
