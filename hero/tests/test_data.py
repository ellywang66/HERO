import json

import pytest

from hero.data import LengthCropCollator, cap_per_class, load_samples, split_by_source
from hero.data.labels import DEFAULT_CLASSES
from hero.data_process.convert_raw import convert, parse_args


class WhitespaceTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer: one token per word, [CLS] ... [SEP]."""

    cls_id, sep_id, pad_id = 1, 2, 0

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[10 + i for i, _ in enumerate(t.split())] for t in texts]}

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_id] + list(ids) + [self.sep_id]

    def pad(self, encoded, **kwargs):
        import torch

        seqs = encoded["input_ids"]
        width = max(len(s) for s in seqs)
        ids = [s + [self.pad_id] * (width - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def doc(n):
    return " ".join(["w"] * n)


def test_truncates_to_max_length_without_crop():
    collate = LengthCropCollator(WhitespaceTokenizer(), max_length=128, crop_p=0.0)
    batch = collate([{"text": doc(1000), "label": 0}, {"text": doc(10), "label": 3}])
    assert batch["input_ids"].shape == (2, 128)
    assert batch["attention_mask"][1].sum() == 12
    assert batch["labels"].tolist() == [0, 3]


def test_crop_always_uses_shorter_lengths():
    collate = LengthCropCollator(
        WhitespaceTokenizer(), max_length=512, crop_p=1.0, crop_lengths=[64, 128, 256, 512], seed=0
    )
    lengths = {
        int(collate([{"text": doc(2000), "label": 0}])["attention_mask"].sum()) for _ in range(50)
    }
    assert lengths == {64, 128, 256}


def test_smallest_specialist_crops_to_shorter_length_only():
    collate = LengthCropCollator(
        WhitespaceTokenizer(), max_length=128, crop_p=1.0, crop_lengths=[64, 128, 256, 512], seed=0
    )
    assert collate.crop_lengths == [64]
    assert int(collate([{"text": doc(2000), "label": 0}])["attention_mask"].sum()) == 64


def test_invalid_crop_position():
    with pytest.raises(ValueError):
        LengthCropCollator(WhitespaceTokenizer(), max_length=128, crop_position="middle")


def _samples():
    out = []
    for i in range(10):
        for label in ("human", "generated", "translated_fr"):
            out.append({"source_id": f"a{i}", "label": label, "text": f"{label} {i}"})
    return out


def test_split_keeps_sources_together():
    train, val = split_by_source(_samples(), val_fraction=0.2, seed=0)
    assert len(val) == 6 and len(train) == 24
    assert not {s["source_id"] for s in train} & {s["source_id"] for s in val}


def test_cap_per_class():
    capped = cap_per_class(_samples(), max_per_class=4, seed=0)
    assert len(capped) == 12


def test_convert_raw_links_translations_by_text(tmp_path):
    gp = {
        "idA": {
            "original_text": "Human A.",
            "generated_text": "Here is the text:\n\nGen A.",
            "paraphrased_text": "Para A.",
        },
        "idB": {"text": "Human B.", "generated_text": "Gen B.", "paraphrased_text": ""},
    }
    hum = {"idA": {"humanized_text": "Hum A."}}
    # Keyed by position, not article id; language taken from the record.
    zh = {
        "0": {
            "original_text": "Human B.",
            "round_trip_text": "RT B.",
            "translated_langauge": "Chinese",
        }
    }
    # Language taken from the filename; unsupported languages are skipped.
    fr = {"idA": {"original_text": "Human A.", "round_trip_text": "RT A."}}
    tr = {
        "idA": {
            "original_text": "Human A.",
            "round_trip_text": "RT A.",
            "translated_langauge": "Turkish",
        }
    }
    files = {}
    for name, content in [
        ("gp.json", gp),
        ("hum.json", hum),
        ("zh.json", zh),
        ("x-French.json", fr),
        ("tr.json", tr),
    ]:
        files[name] = tmp_path / name
        files[name].write_text(json.dumps(content))
    out = tmp_path / "out.jsonl"
    stats = convert(
        parse_args(
            [
                "--dataset",
                "goodnews",
                "--split",
                "train",
                "--generator",
                "gen",
                "--generate-paraphrase-files",
                str(files["gp.json"]),
                "--humanize-files",
                str(files["hum.json"]),
                "--translate-files",
                str(files["zh.json"]),
                str(files["x-French.json"]),
                str(files["tr.json"]),
                "--output",
                str(out),
            ]
        )
    )
    samples = load_samples([out], DEFAULT_CLASSES)
    by_id = {s["id"]: s for s in samples}
    assert by_id["idA:generated"]["text"] == "Gen A."
    assert by_id["idB:human"]["text"] == "Human B."
    assert "idB:paraphrased" not in by_id
    assert by_id["idB:translated_zh"]["source_id"] == "idB"
    assert by_id["idA:translated_fr"]["text"] == "RT A."
    assert by_id["idA:humanized"]["text"] == "Hum A."
    assert stats["skipped_language_Turkish"] == 1
    assert stats["empty_paraphrased"] == 1
