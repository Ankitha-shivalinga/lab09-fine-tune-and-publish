"""Two cheap opponents: the most common label, and TF-IDF with logistic regression."""

from collections.abc import Sequence

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

BASELINES = ("most_common", "tfidf_logreg")


def most_common_label(texts: Sequence[str], labels: Sequence[str]) -> Pipeline:
    """Always predict the label that is most common in the training rows."""
    model = make_pipeline(TfidfVectorizer(), DummyClassifier(strategy="most_frequent"))
    return model.fit(list(texts), list(labels))


def tfidf_logreg(
    texts: Sequence[str], labels: Sequence[str], seed: int = 0, c: float = 1.0
) -> Pipeline:
    """Word and word-pair counts, weighted by TF-IDF, into a logistic regression."""
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=c, random_state=seed),
    )
    return model.fit(list(texts), list(labels))


def fit_baseline(
    name: str,
    texts: Sequence[str],
    labels: Sequence[str],
    seed: int = 0,
    c: float = 1.0,
) -> Pipeline:
    """Fit a baseline by name on the given training rows only."""
    if name == "most_common":
        return most_common_label(texts, labels)
    if name == "tfidf_logreg":
        return tfidf_logreg(texts, labels, seed=seed, c=c)
    raise ValueError(f"unknown baseline: {name}")
