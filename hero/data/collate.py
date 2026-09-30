"""Tokenization with length cropping (Sec. 3.1.2).

Each length specialist sees at most ``max_length`` tokens. With probability
``crop_p`` a training document is additionally cropped to a shorter length drawn
from ``crop_lengths``, so that a specialist also learns from documents shorter
than the length it targets.
"""

import random

import torch


class LengthCropCollator:
    def __init__(
        self,
        tokenizer,
        max_length,
        crop_p=0.0,
        crop_lengths=(64, 128, 256, 512),
        crop_position="start",
        seed=None,
    ):
        if crop_position not in ("start", "random"):
            raise ValueError(f"crop_position must be 'start' or 'random', got '{crop_position}'")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.crop_p = crop_p
        self.crop_position = crop_position
        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)
        # Only lengths shorter than the specialist's own length change its input.
        self.crop_lengths = sorted(length for length in crop_lengths if length < max_length)
        self.rng = random.Random(seed)

    def crop(self, ids):
        """Crop a list of token ids (without special tokens)."""
        max_body = self.max_length - self.num_special
        length = max_body
        if self.crop_lengths and self.rng.random() < self.crop_p:
            length = min(max_body, self.rng.choice(self.crop_lengths) - self.num_special)
        if len(ids) <= length:
            return ids
        start = 0
        if self.crop_position == "random" and length < max_body:
            start = self.rng.randint(0, len(ids) - length)
        return ids[start : start + length]

    def __call__(self, batch):
        encoded = self.tokenizer(
            [example["text"] for example in batch],
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        input_ids = [
            self.tokenizer.build_inputs_with_special_tokens(self.crop(ids)) for ids in encoded
        ]
        padded = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        padded["labels"] = torch.tensor([example["label"] for example in batch], dtype=torch.long)
        return padded
