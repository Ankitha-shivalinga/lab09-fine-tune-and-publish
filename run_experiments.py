"""Run the whole comparison and write everything in reports/.

    python run_experiments.py               # tune, grid, final (all stages)
    python run_experiments.py --stage tune  # settings chosen on validation only
    python run_experiments.py --stage grid  # every method x size x seed
    python run_experiments.py --stage final # choose on validation, test once
    python run_experiments.py --quick       # tiny smoke test in quick_reports/

The test rows and the 40 hand-written messages are only read in the final stage.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from baselines import fit_baseline
from evaluate import (
    PRETTY,
    accuracy_by_kind,
    plot_confusion,
    plot_learning_curve,
    scores,
    summarize,
)
from finetune import BATCH_SIZE, DEFAULT_SETTINGS, fine_tune, load_tokenizer, predict
from prepare import (
    SEEDS,
    SIZES,
    SPLIT_SEED,
    load_own_messages,
    load_tickets,
    small_sample,
    split_data,
)
from share import MODEL_DIR, save_model, write_model_card

TFIDF_C_GRID = (1.0, 10.0)
FT_GRID = {
    "head": {"lr": (1e-3, 3e-3), "epochs": (5, 10)},
    "full": {"lr": (3e-5, 5e-5), "epochs": (3, 5)},
}
METHODS = ("most_common", "tfidf_logreg", "ft_head", "ft_full")


def config(quick: bool) -> dict:
    if quick:
        return {
            "out": Path("quick_reports"),
            "sizes": (16, 24),
            "seeds": (0,),
            "tfidf_c": (1.0,),
            "ft_grid": {
                "head": {"lr": (1e-3,), "epochs": (1,)},
                "full": {"lr": (5e-5,), "epochs": (1,)},
            },
            "model_dir": Path("quick_reports") / "model",
        }
    return {
        "out": Path("reports"),
        "sizes": SIZES,
        "seeds": SEEDS,
        "tfidf_c": TFIDF_C_GRID,
        "ft_grid": FT_GRID,
        "model_dir": MODEL_DIR,
    }


def stage_tune(cfg: dict, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Try a few settings at the largest size, seed 0, scored on validation."""
    size = max(cfg["sizes"])
    sample = small_sample(train, size, seed=0)
    tokenizer = load_tokenizer()
    rows = []
    for c in cfg["tfidf_c"]:
        model = fit_baseline("tfidf_logreg", sample["text"], sample["label"], c=c)
        acc = scores(val["label"], model.predict(val["text"]))["accuracy"]
        rows.append({"method": "tfidf_logreg", "C": c, "val_accuracy": acc})
        print(f"tune tfidf C={c}: {acc:.4f}")
    for variant, grid in cfg["ft_grid"].items():
        for lr in grid["lr"]:
            for epochs in grid["epochs"]:
                print(f"tune {variant} lr={lr} epochs={epochs}")
                _, _, history = fine_tune(
                    sample["text"].tolist(),
                    sample["label"].tolist(),
                    val["text"].tolist(),
                    val["label"].tolist(),
                    variant=variant,
                    lr=lr,
                    epochs=epochs,
                    seed=0,
                    tokenizer=tokenizer,
                )
                rows.append(
                    {
                        "method": f"ft_{variant}",
                        "lr": lr,
                        "epochs": epochs,
                        "val_accuracy": history[-1]["val_accuracy"],
                    }
                )
    tuning = pd.DataFrame(rows)
    tuning.insert(0, "size", size)
    tuning.to_csv(cfg["out"] / "tuning.csv", index=False)

    def best(method: str) -> pd.Series:
        part = tuning[tuning["method"] == method]
        return part.loc[part["val_accuracy"].idxmax()]  # first wins a tie

    settings = {"tfidf_logreg": {"C": float(best("tfidf_logreg")["C"])}}
    for variant in cfg["ft_grid"]:
        row = best(f"ft_{variant}")
        settings[f"ft_{variant}"] = {
            "lr": float(row["lr"]),
            "epochs": int(row["epochs"]),
        }
    (cfg["out"] / "settings.json").write_text(json.dumps(settings, indent=2))
    print("chosen settings:", settings)
    return settings


def load_settings(cfg: dict) -> dict:
    path = cfg["out"] / "settings.json"
    if path.exists():
        return json.loads(path.read_text())
    print("no settings.json yet, using the Module 09 defaults")
    return {
        "tfidf_logreg": {"C": 1.0},
        **{f"ft_{v}": dict(DEFAULT_SETTINGS[v]) for v in DEFAULT_SETTINGS},
    }


def train_and_predict(method, settings, sample, texts, seed, tokenizer, val):
    """Fit one method on the sample. Returns (predict function, history, model)."""
    if method in ("most_common", "tfidf_logreg"):
        c = settings["tfidf_logreg"]["C"]
        model = fit_baseline(method, sample["text"], sample["label"], seed=seed, c=c)
        return model.predict(list(texts)).tolist(), [], model
    variant = method.removeprefix("ft_")
    model, tok, history = fine_tune(
        sample["text"].tolist(),
        sample["label"].tolist(),
        val["text"].tolist(),
        val["label"].tolist(),
        variant=variant,
        lr=settings[method]["lr"],
        epochs=settings[method]["epochs"],
        seed=seed,
        tokenizer=tokenizer,
    )
    return predict(model, tok, list(texts)), history, model


def stage_grid(cfg: dict, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Every method on the same sampled rows, scored on the same validation rows."""
    settings = load_settings(cfg)
    tokenizer = load_tokenizer()
    results, logs = [], []
    for size in cfg["sizes"]:
        for seed in cfg["seeds"]:
            sample = small_sample(train, size, seed=seed)
            for method in METHODS:
                print(f"size {size} seed {seed} {method}")
                start = time.perf_counter()
                predicted, history, _ = train_and_predict(
                    method, settings, sample, val["text"], seed, tokenizer, val
                )
                result = scores(val["label"], predicted)
                params = settings.get(method, {})
                results.append(
                    {
                        "method": method,
                        "size": size,
                        "seed": seed,
                        "val_accuracy": result["accuracy"],
                        "val_macro_f1": result["macro_f1"],
                        "lr": params.get("lr", ""),
                        "epochs": params.get("epochs", ""),
                        "C": params.get("C", ""),
                        "seconds": round(time.perf_counter() - start, 1),
                    }
                )
                for entry in history:
                    logs.append({"method": method, "size": size, "seed": seed, **entry})
    frame = pd.DataFrame(results)
    frame.to_csv(cfg["out"] / "results.csv", index=False)
    pd.DataFrame(logs).to_csv(cfg["out"] / "training_log.csv", index=False)
    summarize(frame).to_csv(cfg["out"] / "summary.csv", index=False)
    plot_learning_curve(frame, cfg["out"] / "learning_curve.png")
    print(summarize(frame).to_string(index=False))
    return frame


def choose(results: pd.DataFrame, methods) -> str:
    """Highest mean validation accuracy at the largest size (validation only)."""
    largest = results[results["size"] == results["size"].max()]
    means = largest.groupby("method")["val_accuracy"].mean()
    return str(means[list(methods)].idxmax())


def stage_final(cfg: dict, train, val, test, quick: bool) -> dict:
    results = pd.read_csv(cfg["out"] / "results.csv")
    settings = load_settings(cfg)
    final_method = choose(results, METHODS)
    best_ft = choose(results, ("ft_head", "ft_full"))
    size = int(results["size"].max())
    print(f"chosen on validation: {final_method}; best fine-tuned: {best_ft}")

    try:
        own = load_own_messages()
    except (FileNotFoundError, ValueError):
        if not quick:
            raise
        own = None
        print("quick run: no own_messages.csv, skipping that score")

    sample = small_sample(train, size, seed=0)
    tokenizer = load_tokenizer()
    test_scores, own_scores, test_predictions, by_kind = {}, {}, {}, {}
    published = None
    own_texts = [] if own is None else own["text"].tolist()
    n_val, n_test = len(val), len(test)
    for method in ("most_common", "tfidf_logreg", best_ft):
        texts = val["text"].tolist() + test["text"].tolist() + own_texts
        predicted, _, model = train_and_predict(
            method, settings, sample, texts, 0, tokenizer, val
        )
        on_val = predicted[:n_val]
        on_test = predicted[n_val : n_val + n_test]
        on_own = predicted[n_val + n_test :]
        by_kind[method] = accuracy_by_kind(val["text"], val["label"], on_val)
        test_predictions[method] = on_test
        test_scores[method] = scores(test["label"], on_test)  # the one look
        if own is not None:
            own_scores[method] = scores(own["label"], on_own)
        if method == best_ft:
            published = model

    save_model(published, tokenizer, cfg["model_dir"])
    final = {
        "split_seed": SPLIT_SEED,
        "rule": "highest mean validation accuracy at the largest size",
        "final_method": final_method,
        "train_examples": size,
        "validation_mean_at_largest": (
            results[results["size"] == size]
            .groupby("method")["val_accuracy"]
            .mean()
            .round(4)
            .to_dict()
        ),
        "validation_vague_vs_clear": by_kind,
        "test": test_scores,
        "own_messages": own_scores,
        "published_model": {
            "method": best_ft,
            "variant": "head-only" if best_ft == "ft_head" else "full",
            "train_examples": size,
            "seed": 0,
            "lr": settings[best_ft]["lr"],
            "epochs": settings[best_ft]["epochs"],
            "batch_size": BATCH_SIZE,
        },
    }
    (cfg["out"] / "final.json").write_text(json.dumps(final, indent=2))
    panels = {}
    for method in ("tfidf_logreg", best_ft):
        mark = "  [final choice]" if method == final_method else ""
        panels[PRETTY[method] + mark] = (test["label"], test_predictions[method])
    plot_confusion(panels, cfg["out"] / "confusion_matrix.png")
    if own is not None:
        write_model_card(cfg["model_dir"], final)
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "tune", "grid", "final"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    stage = args.stage or "all"

    cfg = config(args.quick)
    cfg["out"].mkdir(parents=True, exist_ok=True)
    train, val, test = split_data(load_tickets())
    if stage in ("all", "tune"):
        stage_tune(cfg, train, val)
    if stage in ("all", "grid"):
        stage_grid(cfg, train, val)
    if stage in ("all", "final"):
        stage_final(cfg, train, val, test, args.quick)


if __name__ == "__main__":
    main()
