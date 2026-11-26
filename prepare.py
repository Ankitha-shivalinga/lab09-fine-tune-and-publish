"""Load the tickets, split them once, and draw small stratified training samples."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tickets_data import LABELS, make_tickets

SPLIT_SEED = 42
SIZES = (50, 100, 200, 400)
SEEDS = (0, 1, 2)
OWN_MESSAGES = Path("data") / "own_messages.csv"

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


def load_tickets() -> pd.DataFrame:
    """Return the 1,200 generated tickets, always with the default arguments."""
    return make_tickets()


def split_data(
    frame: pd.DataFrame, seed: int = SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split once into stratified 60/20/20 train, validation and test parts."""
    train, rest = train_test_split(
        frame, train_size=0.6, stratify=frame["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, train_size=0.5, stratify=rest["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def small_sample(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a stratified sample of n rows from the training part only."""
    if n > len(train):
        raise ValueError(f"asked for {n} rows but the training part has {len(train)}")
    if n == len(train):
        return train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample, _ = train_test_split(
        train, train_size=n, stratify=train["label"], random_state=seed
    )
    return sample.reset_index(drop=True)


def load_own_messages(path: Path = OWN_MESSAGES) -> pd.DataFrame:
    """Read the 40 hand-written messages and check their labels."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} is missing. Write your 40 messages first (see the README)."
        )
    own = pd.read_csv(path)
    if list(own.columns) != ["text", "label"]:
        raise ValueError("own_messages.csv must have exactly the columns text,label")
    unknown = set(own["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels in own_messages.csv: {sorted(unknown)}")
    counts = own["label"].value_counts()
    if len(own) != 40 or not (counts == 10).all() or len(counts) != len(LABELS):
        raise ValueError(
            "own_messages.csv needs 40 messages, 10 for each queue; "
            f"found {counts.to_dict()}"
        )
    return own


if __name__ == "__main__":
    tickets = load_tickets()
    train, val, test = split_data(tickets)
    print("rows:", len(train), len(val), len(test))
    print(train["label"].value_counts().to_string())
    for n in SIZES:
        counts = small_sample(train, n, seed=0)["label"].value_counts().to_dict()
        print(n, counts)
