"""Subcategory Guidance (Sec. 3.1.1).

    L_total = L_CE + lambda * sum_g L_g

For each group ``g`` of easily confused categories (generated/humanized and the
four translation source languages), ``L_g`` is a cross entropy computed only on
samples from that group and only over the logits of that group's categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubcategoryGuidanceLoss(nn.Module):
    def __init__(self, groups, weight):
        """
        Args:
            groups: mapping from group name to the label indices in that group.
            weight: lambda. ``0`` disables guidance (plain cross entropy).
        """
        super().__init__()
        self.weight = weight
        self.group_names = list(groups)
        for name, idx in groups.items():
            self.register_buffer(f"idx_{name}", torch.tensor(idx, dtype=torch.long))

    def group_loss(self, logits, labels, idx):
        mask = torch.isin(labels, idx)
        if not mask.any():
            return logits.new_zeros(())
        sub_logits = logits[mask][:, idx]
        # Position of each label inside the group.
        sub_labels = (labels[mask].unsqueeze(1) == idx.unsqueeze(0)).long().argmax(dim=1)
        return F.cross_entropy(sub_logits, sub_labels)

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels)
        terms = {"ce": ce}
        total = ce
        if self.weight:
            for name in self.group_names:
                loss = self.group_loss(logits, labels, getattr(self, f"idx_{name}"))
                terms[name] = loss
                total = total + self.weight * loss
        terms["total"] = total
        return total, terms
