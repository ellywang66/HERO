"""Fine-grained machine-influenced text categories used by HERO.

The default order matches the label indices used for the paper's experiments.
"""

HUMAN = "human"
GENERATED = "generated"
PARAPHRASED = "paraphrased"
HUMANIZED = "humanized"
TRANSLATED_ZH = "translated_zh"
TRANSLATED_FR = "translated_fr"
TRANSLATED_ES = "translated_es"
TRANSLATED_RU = "translated_ru"

DEFAULT_CLASSES = [
    HUMAN,
    GENERATED,
    PARAPHRASED,
    TRANSLATED_ZH,
    HUMANIZED,
    TRANSLATED_FR,
    TRANSLATED_ES,
    TRANSLATED_RU,
]

# Source language name -> translated category.
LANGUAGE_TO_CLASS = {
    "Chinese": TRANSLATED_ZH,
    "French": TRANSLATED_FR,
    "Spanish": TRANSLATED_ES,
    "Russian": TRANSLATED_RU,
}


def resolve_groups(groups, classes):
    """Map subcategory groups given as class names to label indices.

    Classes that are not part of ``classes`` are dropped, and groups with fewer
    than two remaining classes are skipped since there is nothing to separate.
    """
    class_to_idx = {name: i for i, name in enumerate(classes)}
    resolved = {}
    for group_name, members in groups.items():
        unknown = [m for m in members if m not in DEFAULT_CLASSES]
        if unknown:
            raise ValueError(f"Unknown classes {unknown} in subcategory group '{group_name}'")
        idx = [class_to_idx[m] for m in members if m in class_to_idx]
        if len(idx) >= 2:
            resolved[group_name] = idx
    return resolved
