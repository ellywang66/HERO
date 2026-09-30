"""Transformer encoder with a fine-grained classification head."""

import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Detector(nn.Module):
    """Shared encoder ``g`` followed by one classifier over all categories.

    Subcategory Guidance operates on subsets of this classifier's logits, so the
    detector has no extra parameters that would need to be dropped at test time.
    """

    def __init__(self, backbone, classes, cache_dir=None):
        super().__init__()
        self.backbone = backbone
        self.classes = list(classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone,
            num_labels=len(self.classes),
            id2label=dict(enumerate(self.classes)),
            label2id={name: i for i, name in enumerate(self.classes)},
            # Detector backbones (e.g. OpenAI-D) ship with a binary head.
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir,
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def load_tokenizer(backbone, cache_dir=None):
    return AutoTokenizer.from_pretrained(backbone, cache_dir=cache_dir)
