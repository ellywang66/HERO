import pytest
import torch
import torch.nn.functional as F

from hero.data.labels import DEFAULT_CLASSES, resolve_groups
from hero.losses import SubcategoryGuidanceLoss

GROUPS = {
    "generated_humanized": ["generated", "humanized"],
    "translated": ["translated_zh", "translated_fr", "translated_es", "translated_ru"],
}


def test_resolve_groups_default_order():
    assert resolve_groups(GROUPS, DEFAULT_CLASSES) == {
        "generated_humanized": [1, 4],
        "translated": [3, 5, 6, 7],
    }


def test_resolve_groups_drops_missing_classes():
    classes = ["human", "generated", "paraphrased", "translated_zh", "humanized"]
    # Only one translated class left, so the translated group is skipped.
    assert resolve_groups(GROUPS, classes) == {"generated_humanized": [1, 4]}


def test_resolve_groups_rejects_unknown_class():
    with pytest.raises(ValueError):
        resolve_groups({"bad": ["human", "typo"]}, DEFAULT_CLASSES)


def test_zero_weight_is_cross_entropy():
    logits = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    loss_fn = SubcategoryGuidanceLoss(resolve_groups(GROUPS, DEFAULT_CLASSES), weight=0.0)
    total, _ = loss_fn(logits, labels)
    assert torch.allclose(total, F.cross_entropy(logits, labels))


def test_guidance_uses_group_samples_and_logits_only():
    logits = torch.randn(5, 8)
    labels = torch.tensor([0, 1, 4, 6, 2])
    loss_fn = SubcategoryGuidanceLoss(resolve_groups(GROUPS, DEFAULT_CLASSES), weight=0.5)
    total, terms = loss_fn(logits, labels)

    expected_gh = F.cross_entropy(logits[[1, 2]][:, [1, 4]], torch.tensor([0, 1]))
    expected_tr = F.cross_entropy(logits[[3]][:, [3, 5, 6, 7]], torch.tensor([2]))
    assert torch.allclose(terms["generated_humanized"], expected_gh)
    assert torch.allclose(terms["translated"], expected_tr)
    expected = F.cross_entropy(logits, labels) + 0.5 * (expected_gh + expected_tr)
    assert torch.allclose(total, expected)


def test_group_without_samples_contributes_zero():
    logits = torch.randn(2, 8)
    labels = torch.tensor([0, 2])
    loss_fn = SubcategoryGuidanceLoss(resolve_groups(GROUPS, DEFAULT_CLASSES), weight=1.0)
    total, terms = loss_fn(logits, labels)
    assert terms["generated_humanized"].item() == 0.0
    assert torch.allclose(total, F.cross_entropy(logits, labels))


def test_guidance_gradient_reaches_shared_logits():
    logits = torch.randn(2, 8, requires_grad=True)
    labels = torch.tensor([1, 4])
    loss_fn = SubcategoryGuidanceLoss(resolve_groups(GROUPS, DEFAULT_CLASSES), weight=1.0)
    _, terms = loss_fn(logits, labels)
    terms["generated_humanized"].backward()
    grad = logits.grad.abs().sum(dim=0)
    assert grad[[1, 4]].sum() > 0
    assert grad[[0, 2, 3, 5, 6, 7]].sum() == 0
