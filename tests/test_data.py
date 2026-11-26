import pytest

from baselines import fit_baseline
from evaluate import scores
from prepare import load_tickets, small_sample, split_data
from tickets_data import LABELS


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_split_sizes_and_no_overlap(seed):
    train, val, test = split_data(load_tickets(), seed=seed)
    assert (len(train), len(val), len(test)) == (720, 240, 240)
    ids = [set(part["ticket_id"]) for part in (train, val, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert len(ids[0] | ids[1] | ids[2]) == 1200


def test_split_is_stratified(parts):
    train, val, _ = parts
    share_train = train["label"].value_counts(normalize=True)
    share_val = val["label"].value_counts(normalize=True)
    assert (share_train - share_val).abs().max() < 0.02


@pytest.mark.parametrize("n", [20, 50, 300])
def test_small_sample_rows_labels_and_source(parts, n):
    train, val, test = parts
    sample = small_sample(train, n, seed=3)
    assert len(sample) == n
    assert set(sample["label"]) == set(LABELS)
    assert set(sample["ticket_id"]) <= set(train["ticket_id"])
    assert not set(sample["ticket_id"]) & set(val["ticket_id"])
    assert not set(sample["ticket_id"]) & set(test["ticket_id"])
    assert sample.equals(small_sample(train, n, seed=3))  # the seed repeats it


def test_small_sample_refuses_too_many(parts):
    with pytest.raises(ValueError):
        small_sample(parts[0], 10_000, seed=0)


def test_tfidf_beats_most_common_on_validation(parts):
    train, val, _ = parts
    sample = small_sample(train, 200, seed=0)
    results = {}
    for name in ("most_common", "tfidf_logreg"):
        model = fit_baseline(name, sample["text"], sample["label"])
        results[name] = scores(val["label"], model.predict(val["text"]))["accuracy"]
    assert results["tfidf_logreg"] > results["most_common"] + 0.2
