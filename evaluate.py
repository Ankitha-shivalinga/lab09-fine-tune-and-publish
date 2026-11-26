"""Scores, the summary over seeds, the learning curve and the confusion matrix."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from tickets_data import _VAGUE, LABELS  # noqa: E402

PRETTY = {
    "most_common": "Most common label",
    "tfidf_logreg": "TF-IDF + logistic",
    "ft_head": "DistilBERT, head only",
    "ft_full": "DistilBERT, full fine-tuning",
}


def scores(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Accuracy and macro F1, rounded to four places."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=LABELS, average="macro")), 4
        ),
    }


def is_vague(text: str) -> bool:
    """True if the message is one of the generator's topic-free sentences."""
    return any(sentence in text for sentence in _VAGUE)


def accuracy_by_kind(
    texts: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, float]:
    """Accuracy on the vague messages and on the rest, to check the ceiling."""
    vague = [is_vague(t) for t in texts]
    out = {"n_vague": int(sum(vague)), "n_clear": int(len(vague) - sum(vague))}
    for kind, keep in (("vague", True), ("clear", False)):
        pairs = [
            (t, p) for v, t, p in zip(vague, y_true, y_pred, strict=True) if v == keep
        ]
        hits = sum(t == p for t, p in pairs)
        out[f"{kind}_accuracy"] = round(hits / len(pairs), 4) if pairs else float("nan")
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean, lowest and highest validation scores per method and size."""
    grouped = results.groupby(["method", "size"])
    table = grouped.agg(
        runs=("seed", "count"),
        acc_mean=("val_accuracy", "mean"),
        acc_min=("val_accuracy", "min"),
        acc_max=("val_accuracy", "max"),
        f1_mean=("val_macro_f1", "mean"),
    )
    return table.round(4).reset_index()


def plot_learning_curve(results: pd.DataFrame, path: Path) -> None:
    """Validation accuracy against training examples, mean with the seed range."""
    table = summarize(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in PRETTY:
        part = table[table["method"] == method].sort_values("size")
        if part.empty:
            continue
        ax.plot(part["size"], part["acc_mean"], marker="o", label=PRETTY[method])
        ax.fill_between(part["size"], part["acc_min"], part["acc_max"], alpha=0.2)
    ax.set_xscale("log")
    sizes = sorted(table["size"].unique())
    ax.minorticks_off()
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy by training size (mean, band = range of seeds)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(
    panels: dict[str, tuple[Sequence[str], Sequence[str]]], path: Path
) -> None:
    """One confusion matrix per method, on the test rows."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, (y_true, y_pred)) in zip(axes, panels.items(), strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=LABELS, ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
