#!/usr/bin/env python3
"""Train a HERO length specialist (or a baseline detector).

Examples:
    # One HERO specialist
    python hero/train.py +exp=hero/specialist_512

    # All three HERO specialists
    python hero/train.py -m +exp=hero/specialist_128,hero/specialist_256,hero/specialist_512

    # Fine-tuned DistilBERT baseline
    python hero/train.py +exp=baselines/distilbert
"""

import json
import os
import sys

# Make `import hero` work when this file is run as a script from the repo root.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import hydra  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from hero.data import (  # noqa: E402
    LengthCropCollator,
    TextClassificationDataset,
    cap_per_class,
    load_samples,
    resolve_groups,
    split_by_source,
)
from hero.losses import SubcategoryGuidanceLoss  # noqa: E402
from hero.modules import Detector, load_tokenizer  # noqa: E402
from hero.trainer import Trainer  # noqa: E402
from hero.utils import get_device, seeding  # noqa: E402


def build_datasets(cfg):
    classes = list(cfg.classes)
    samples = load_samples(cfg.data.train_files, classes, root=cfg.data.root)
    train, val = split_by_source(samples, cfg.data.val_fraction, cfg.seed)
    train = cap_per_class(train, cfg.data.max_train_per_class, cfg.seed)
    train_set = TextClassificationDataset(train, classes)
    val_set = TextClassificationDataset(val, classes)
    logger.info(f"Train samples per class: {train_set.class_counts()}")
    logger.info(f"Val samples per class: {val_set.class_counts()}")
    return train_set, val_set


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    output_dir = cfg.experiment_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.add(os.path.join(output_dir, "train.log"))
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    seeding(cfg.seed)
    device = get_device(cfg.trainer.device)
    classes = list(cfg.classes)

    tokenizer = load_tokenizer(cfg.model.backbone, cache_dir=cfg.model.cache_dir)
    train_set, val_set = build_datasets(cfg)

    train_collate = LengthCropCollator(
        tokenizer,
        max_length=cfg.max_length,
        crop_p=cfg.data.crop.p,
        crop_lengths=cfg.data.crop.lengths,
        crop_position=cfg.data.crop.position,
        seed=cfg.seed,
    )
    val_collate = LengthCropCollator(tokenizer, max_length=cfg.max_length)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.trainer.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=val_collate,
    )

    groups = resolve_groups(OmegaConf.to_container(cfg.loss.groups), classes)
    logger.info(f"Subcategory Guidance groups (label indices): {groups}")
    loss_fn = SubcategoryGuidanceLoss(groups, weight=cfg.loss.weight)
    model = Detector(cfg.model.backbone, classes, cache_dir=cfg.model.cache_dir)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        num_epochs=cfg.trainer.num_epochs,
        lr=cfg.trainer.lr,
        lr_decay_factor=cfg.trainer.lr_decay_factor,
        lr_patience=cfg.trainer.lr_patience,
        log_interval=cfg.trainer.log_interval,
        checkpoint_meta={
            "backbone": cfg.model.backbone,
            "classes": classes,
            "max_length": cfg.max_length,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )
    best_mAP = trainer.fit()
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({"best_val_mAP": best_mAP, "max_length": cfg.max_length}, f, indent=2)
    logger.info(f"Done. Best val mAP: {best_mAP:.4f}. Checkpoints in {output_dir}")
    return best_mAP


if __name__ == "__main__":
    main()
