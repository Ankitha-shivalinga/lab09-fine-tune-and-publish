"""Fine-tune DistilBERT for the four queues with a loop you can read."""

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from prepare import ID2LABEL, LABEL2ID

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 16
VARIANTS = ("head", "full")

# Starting points from Module 09. run_experiments.py --stage tune compares
# other values on the validation rows and writes the winners to settings.json.
DEFAULT_SETTINGS = {
    "head": {"lr": 1e-3, "epochs": 5},
    "full": {"lr": 5e-5, "epochs": 4},
}


def load_tokenizer(name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(name)


def build_model(seed: int, name: str = MODEL_NAME):
    """Load the pretrained body with a new four-way head.

    The seed is set first, because the head's random weights are drawn here.
    """
    set_seed(seed)
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )


def body_of(model) -> torch.nn.Module:
    """The pretrained part of the model (everything except the new head)."""
    return getattr(model, model.base_model_prefix)


def freeze_body(model) -> int:
    """Stop the body from training. Returns how many parameters can still change."""
    for parameter in body_of(model).parameters():
        parameter.requires_grad = False
    return count_trainable(model)


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def encode(
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[str] | None = None,
    max_length: int = MAX_LENGTH,
) -> list[dict]:
    """Tokenize without padding (the collator pads each batch), and add label ids."""
    encoded = tokenizer(list(texts), truncation=True, max_length=max_length)
    rows = []
    for i in range(len(texts)):
        row = {key: values[i] for key, values in encoded.items()}
        if labels is not None:
            row["labels"] = LABEL2ID[labels[i]]
        rows.append(row)
    return rows


def make_loader(
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[str] | None = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        encode(tokenizer, texts, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


@torch.no_grad()
def predict_proba(model, tokenizer, texts: Sequence[str]) -> np.ndarray:
    """Probabilities for every queue, one row per text, columns in LABELS order."""
    model.eval()
    chunks = []
    for batch in make_loader(tokenizer, texts, batch_size=64):
        logits = model(**batch).logits
        chunks.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(chunks)


def predict(model, tokenizer, texts: Sequence[str]) -> list[str]:
    best = predict_proba(model, tokenizer, texts).argmax(axis=1)
    return [ID2LABEL[int(i)] for i in best]


def fine_tune(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    variant: str = "full",
    lr: float | None = None,
    epochs: int | None = None,
    seed: int = 0,
    batch_size: int = BATCH_SIZE,
    tokenizer=None,
    log: Callable[[str], None] | None = print,
):
    """Train head only or the whole model. Returns model, tokenizer and history.

    The history has one entry per epoch with the training loss and the
    validation accuracy, as the lab asks.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    lr = DEFAULT_SETTINGS[variant]["lr"] if lr is None else lr
    epochs = DEFAULT_SETTINGS[variant]["epochs"] if epochs is None else epochs
    tokenizer = tokenizer or load_tokenizer()

    model = build_model(seed)  # seeds the head, the dropout and the shuffling
    if variant == "head":
        freeze_body(model)

    train_loader = make_loader(
        tokenizer, train_texts, train_labels, batch_size, shuffle=True, seed=seed
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total = epochs * len(train_loader)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)

    history = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        losses = []
        for batch in train_loader:
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        predicted = predict(model, tokenizer, val_texts)
        accuracy = float(np.mean(np.array(predicted) == np.array(val_labels)))
        entry = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4),
            "val_accuracy": round(accuracy, 4),
            "seconds": round(time.perf_counter() - start, 1),
        }
        history.append(entry)
        if log:
            log(
                f"  {variant:4s} epoch {epoch}  train loss {entry['train_loss']:.4f}"
                f"  validation accuracy {entry['val_accuracy']:.4f}"
            )
    model.eval()
    return model, tokenizer, history
