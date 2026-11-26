"""Save the fine-tuned model, write its model card, and load it back for scoring."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import MAX_LENGTH, MODEL_NAME

MODEL_DIR = Path("model")
CARD_HEADINGS = ("## Intended use", "## Training data", "## Results", "## Limitations")
EMPTY_MESSAGE = "Please type a support message first."


def save_model(model, tokenizer, folder: Path = MODEL_DIR) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def load_classifier(source: str | Path = MODEL_DIR):
    """Load a tokenizer and model from a folder or a Hub model id."""
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    model = AutoModelForSequenceClassification.from_pretrained(str(source)).eval()
    return tokenizer, model


@torch.no_grad()
def label_scores(classifier, text: str) -> dict[str, float]:
    """A probability for every queue. Raises ValueError for empty text."""
    if text is None or not text.strip():
        raise ValueError(EMPTY_MESSAGE)
    tokenizer, model = classifier
    batch = tokenizer(
        [text.strip()], truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    probabilities = torch.softmax(model(**batch).logits, dim=-1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probabilities)}


def _row(name: str, part: dict) -> str:
    return f"| {name} | {part['accuracy']:.4f} | {part['macro_f1']:.4f} |"


def write_model_card(folder: Path, final: dict) -> Path:
    """Write README.md for the model repository from the numbers in final.json."""
    test = final["test"]
    own = final["own_messages"]
    ft = final["published_model"]
    names = {
        "most_common": "Most common label (baseline)",
        "tfidf_logreg": "TF-IDF + logistic regression (baseline)",
        ft["method"]: f"This model: DistilBERT, {ft['variant']} fine-tuning",
    }
    test_rows = "\n".join(_row(names[k], test[k]) for k in names)
    own_rows = "\n".join(_row(names[k], own[k]) for k in names)
    card = f"""---
license: apache-2.0
base_model: {MODEL_NAME}
pipeline_tag: text-classification
language: en
tags:
- text-classification
- distilbert
- aigc-5005
---

# Lakeshore Ticket Sorter (DistilBERT)

A small classifier that suggests a support queue (billing, bike_fault, station or
account) for a short customer message. It is
[{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) with a new four-way head,
fine-tuned for a course lab (Humber AIGC 5005, Lab 09).

## Intended use

To *suggest* a queue for a short English support message to a person who then
checks it. It was made for teaching, to compare fine-tuning with a cheap baseline.
It is not meant to route real customer messages without a person in the loop, and it
should not be used for any other kind of text.

## Training data

1,200 invented messages from `tickets_data.py` (`make_tickets()` with its default
arguments), split once into 720 training, 240 validation and 240 test rows
(stratified, seed {final["split_seed"]}). This model was trained on a stratified
sample of **{ft["train_examples"]} training examples** (seed {ft["seed"]}), with a
learning rate of {ft["lr"]} for {ft["epochs"]} epochs, batch size {ft["batch_size"]},
and messages truncated at {MAX_LENGTH} tokens. The settings were chosen on the
validation rows only.

## Results

Every row below was trained on the same {ft["train_examples"]} training examples
and scored on the same 240 test rows, which were used once.

| Method | Test accuracy | Test macro F1 |
|---|---|---|
{test_rows}

On 40 messages written by hand (10 per queue), never used for training or tuning:

| Method | Accuracy | Macro F1 |
|---|---|---|
{own_rows}

The method chosen on the validation rows was **{final["final_method"]}**. Means and
ranges over three seeds at every training size are in the project repository.

## Limitations

- The training text was invented by a script. It repeats a small set of sentence
  patterns, so scores on real customer messages will probably be lower.
- About one generated message in eight says nothing about its topic, so no model can
  know its label; about one in ten mixes two topics.
- English only, short messages only (longer text is cut at {MAX_LENGTH} tokens).
- The scores are probabilities from softmax, not a promise that the answer is right.
"""
    path = Path(folder) / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def load_final(path: Path = Path("reports") / "final.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
