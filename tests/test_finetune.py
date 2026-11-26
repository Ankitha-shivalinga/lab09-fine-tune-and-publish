import numpy as np
import torch

from finetune import (
    MAX_LENGTH,
    body_of,
    build_model,
    count_trainable,
    encode,
    fine_tune,
    freeze_body,
)
from prepare import LABEL2ID


def snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def same(before: dict, after: dict) -> bool:
    return all(torch.equal(before[name], after[name]) for name in before)


def body_and_head(model):
    body = snapshot(body_of(model))
    head = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not name.startswith(model.base_model_prefix + ".")
    }
    return body, head


def test_tokenization_truncates_and_keeps_labels(tokenizer):
    texts = ["short one", "word " * 500]
    rows = encode(tokenizer, texts, ["billing", "station"])
    assert len(rows[1]["input_ids"]) == MAX_LENGTH
    assert len(rows[0]["input_ids"]) < MAX_LENGTH  # no padding to the maximum
    assert [r["labels"] for r in rows] == [LABEL2ID["billing"], LABEL2ID["station"]]


def test_freezing_leaves_only_the_head_trainable():
    model = build_model(seed=0)
    total = count_trainable(model)
    trainable = freeze_body(model)
    body = sum(p.numel() for p in body_of(model).parameters())
    assert trainable == total - body
    assert 0 < trainable < 1_000_000  # the head is about 0.6 million parameters
    assert not any(p.requires_grad for p in body_of(model).parameters())


def test_head_only_training_leaves_the_body_unchanged(tiny, tokenizer):
    body_before, head_before = body_and_head(build_model(seed=0))
    model, _, _ = fine_tune(
        *tiny, variant="head", epochs=1, seed=0, tokenizer=tokenizer, log=None
    )
    body_after, head_after = body_and_head(model)
    assert same(body_before, body_after)  # fails if the body was not frozen
    assert not same(head_before, head_after)  # but the head did learn


def test_full_fine_tuning_changes_the_body(tiny, tokenizer):
    body_before, _ = body_and_head(build_model(seed=0))
    model, _, _ = fine_tune(
        *tiny, variant="full", epochs=1, seed=0, tokenizer=tokenizer, log=None
    )
    body_after, _ = body_and_head(model)
    assert not same(body_before, body_after)


def test_same_seed_repeats_a_run(tiny, tokenizer):
    runs = [
        fine_tune(
            *tiny, variant="full", epochs=1, seed=7, tokenizer=tokenizer, log=None
        )
        for _ in range(2)
    ]
    first, second = (snapshot(model) for model, _, _ in runs)
    assert runs[0][2][0]["train_loss"] == runs[1][2][0]["train_loss"]
    assert all(torch.allclose(first[n], second[n], atol=1e-6) for n in first)


def test_model_can_memorize_a_tiny_batch(tiny, tokenizer):
    texts, labels, _, _ = tiny
    _, _, history = fine_tune(
        texts,
        labels,
        texts,  # score on the training rows: this is a memorization check
        labels,
        variant="full",
        lr=1e-4,
        epochs=25,
        seed=0,
        batch_size=8,
        tokenizer=tokenizer,
        log=None,
    )
    assert history[-1]["val_accuracy"] == 1.0
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert np.isfinite(history[-1]["train_loss"])
