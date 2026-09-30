#!/usr/bin/env python3
"""Convert raw article-generation outputs into HERO JSONL training files.

The generation scripts write one JSON dict per (dataset, generator, operation),
keyed by article id:

* generate/paraphrase files: ``original_text`` (or ``text``), ``generated_text``,
  ``paraphrased_text``
* humanize files: ``humanized_text``
* round-trip translation files: ``original_text``, ``round_trip_text`` and the
  source language (``translated_langauge``/``translate_langauge``/``translated_language``,
  or a ``-<Language>.json`` filename suffix)

Example:
    python -m hero.data_process.convert_raw \\
        --dataset goodnews --split train --generator Meta-Llama-3-8B-Instruct \\
        --generate-paraphrase-files raw/goodnews_train-generate_paraphrase-art*-Meta-Llama-3-8B-Instruct.json \\
        --humanize-files raw/goodnews_train-humanize-art*-Meta-Llama-3-8B-Instruct.json \\
        --translate-files raw/goodnews_train-translate-art*-Meta-Llama-3-8B-Instruct-*.json \\
        --output data/goodnews/train/Meta-Llama-3-8B-Instruct.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from loguru import logger

from hero.data.labels import (
    GENERATED,
    HUMAN,
    HUMANIZED,
    LANGUAGE_TO_CLASS,
    PARAPHRASED,
)

# LLMs often prefix their answer with e.g. "Here is the paraphrased text:".
PREAMBLE = re.compile(r"^Here is [^:]+:\s+")
LANGUAGE_FIELDS = ("translated_langauge", "translate_langauge", "translated_language")
# Human texts are matched on this many leading characters.
MATCH_PREFIX = 200


def clean_machine_text(text):
    return PREAMBLE.sub("", (text or "").strip()).strip()


def load_raw(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def human_text(record):
    return (record.get("original_text") or record.get("text") or "").strip()


def translation_language(record, path):
    for field in LANGUAGE_FIELDS:
        if record.get(field):
            return record[field]
    match = re.search(r"-([A-Z][a-z]+)\.json$", Path(path).name)
    return match.group(1) if match else None


def make_sample(source_id, label, text, args):
    return {
        "id": f"{source_id}:{label}",
        "source_id": source_id,
        "text": text,
        "label": label,
        "generator": args.generator,
        "dataset": args.dataset,
        "split": args.split,
    }


def convert(args):
    samples = []
    stats = Counter()
    # Human-text prefix -> article id, used to link translations back to articles.
    prefix_to_id = {}
    human_ids = set()

    for path in args.generate_paraphrase_files:
        for article_id, record in load_raw(path).items():
            text = human_text(record)
            if article_id in human_ids:
                stats["duplicate_article"] += 1
                continue
            human_ids.add(article_id)
            prefix_to_id[text[:MATCH_PREFIX]] = article_id
            for label, value in (
                (HUMAN, text),
                (GENERATED, clean_machine_text(record.get("generated_text"))),
                (PARAPHRASED, clean_machine_text(record.get("paraphrased_text"))),
            ):
                if value:
                    samples.append(make_sample(article_id, label, value, args))
                    stats[label] += 1
                else:
                    stats[f"empty_{label}"] += 1

    for path in args.humanize_files:
        for article_id, record in load_raw(path).items():
            text = clean_machine_text(record.get("humanized_text"))
            if not text:
                stats[f"empty_{HUMANIZED}"] += 1
                continue
            if article_id not in human_ids:
                stats["humanized_unlinked"] += 1
            samples.append(make_sample(article_id, HUMANIZED, text, args))
            stats[HUMANIZED] += 1

    for path in args.translate_files:
        for key, record in load_raw(path).items():
            language = translation_language(record, path)
            label = LANGUAGE_TO_CLASS.get(language)
            if label is None:
                stats[f"skipped_language_{language}"] += 1
                continue
            text = clean_machine_text(record.get("round_trip_text"))
            if not text:
                stats[f"empty_{label}"] += 1
                continue
            # Some translation files are keyed by position instead of article id.
            source_id = prefix_to_id.get(human_text(record)[:MATCH_PREFIX])
            if source_id is None:
                source_id = key if key in human_ids else f"{Path(path).stem}:{key}"
                stats["translation_unlinked"] += 1
            samples.append(make_sample(source_id, label, text, args))
            stats[label] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(samples)} samples to {output}")
    logger.info(f"Stats: {dict(sorted(stats.items()))}")
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--generate-paraphrase-files", nargs="+", required=True)
    parser.add_argument("--humanize-files", nargs="*", default=[])
    parser.add_argument("--translate-files", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    convert(parse_args())
