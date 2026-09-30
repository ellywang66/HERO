import json
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from hero.utils.metrics import mean_average_precision


class Trainer:
    """Trains one detector (one length specialist) and keeps the best checkpoint.

    The learning rate starts at ``lr`` and is halved whenever validation mAP stops
    improving. The checkpoint with the highest validation mAP is saved as
    ``best.pt``.
    """

    def __init__(
        self,
        model,
        loss_fn,
        train_loader,
        val_loader,
        device,
        output_dir,
        num_epochs,
        lr,
        lr_decay_factor=0.5,
        lr_patience=0,
        log_interval=100,
        checkpoint_meta=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.checkpoint_meta = checkpoint_meta or {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=lr_decay_factor, patience=lr_patience
        )

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_epoch(self, epoch):
        self.model.train()
        running = {}
        progress = tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for step, batch in enumerate(progress, start=1):
            batch = self._to_device(batch)
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            loss, terms = self.loss_fn(logits, batch["labels"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for name, value in terms.items():
                running[name] = running.get(name, 0.0) + value.item()
            if step % self.log_interval == 0:
                progress.set_postfix({k: f"{v / step:.4f}" for k, v in running.items()})
        return {k: v / max(1, len(self.train_loader)) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_labels, all_probs, total_loss = [], [], 0.0
        for batch in tqdm(self.val_loader, desc="val", dynamic_ncols=True, leave=False):
            batch = self._to_device(batch)
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            total_loss += torch.nn.functional.cross_entropy(
                logits, batch["labels"], reduction="sum"
            ).item()
            all_probs.append(logits.softmax(dim=-1).cpu())
            all_labels.append(batch["labels"].cpu())
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        mAP, aps = mean_average_precision(labels, probs)
        return {"val_loss": total_loss / max(1, len(labels)), "val_mAP": mAP, "val_APs": aps}

    def save(self, name, epoch, metrics):
        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                **self.checkpoint_meta,
            },
            self.output_dir / name,
        )

    def fit(self):
        best_mAP = float("-inf")
        history_path = self.output_dir / "metrics.jsonl"
        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            self.scheduler.step(val_metrics["val_mAP"])

            record = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
            }
            with open(history_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(
                f"epoch {epoch}: train_loss={train_metrics['total']:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} val_mAP={val_metrics['val_mAP']:.4f}"
            )

            self.save("last.pt", epoch, val_metrics)
            if val_metrics["val_mAP"] > best_mAP:
                best_mAP = val_metrics["val_mAP"]
                self.save("best.pt", epoch, val_metrics)
                logger.info(f"New best checkpoint (val_mAP={best_mAP:.4f})")
        return best_mAP
