import pytest

from prepare import load_tickets, split_data


@pytest.fixture(scope="session")
def parts():
    return split_data(load_tickets())


@pytest.fixture(scope="session")
def tiny(parts):
    """Eight training rows (two per queue) and eight validation rows."""
    train, val, _ = parts
    pick = train.groupby("label").head(2)
    return (
        pick["text"].tolist(),
        pick["label"].tolist(),
        val["text"][:8].tolist(),
        val["label"][:8].tolist(),
    )


@pytest.fixture(scope="session")
def tokenizer():
    from finetune import load_tokenizer

    return load_tokenizer()
