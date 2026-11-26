import pytest

from finetune import build_model, predict_proba
from share import (
    CARD_HEADINGS,
    EMPTY_MESSAGE,
    label_scores,
    load_classifier,
    save_model,
    write_model_card,
)
from tickets_data import LABELS

FAKE_FINAL = {
    "split_seed": 42,
    "final_method": "tfidf_logreg",
    "test": {
        name: {"accuracy": 0.8, "macro_f1": 0.79}
        for name in ("most_common", "tfidf_logreg", "ft_full")
    },
    "own_messages": {
        name: {"accuracy": 0.7, "macro_f1": 0.69}
        for name in ("most_common", "tfidf_logreg", "ft_full")
    },
    "published_model": {
        "method": "ft_full",
        "variant": "full",
        "train_examples": 400,
        "seed": 0,
        "lr": 5e-5,
        "epochs": 3,
        "batch_size": 16,
    },
}


@pytest.fixture(scope="module")
def saved(tmp_path_factory, tokenizer):
    folder = tmp_path_factory.mktemp("model")
    model = build_model(seed=0).eval()
    save_model(model, tokenizer, folder)
    return folder, model


def test_saved_model_reloads_with_the_same_prediction(saved, tokenizer):
    folder, model = saved
    texts = ["The dock would not release my bike.", "I was charged twice."]
    reloaded_tokenizer, reloaded = load_classifier(folder)
    before = predict_proba(model, tokenizer, texts)
    after = predict_proba(reloaded, reloaded_tokenizer, texts)
    assert after == pytest.approx(before, abs=1e-6)
    assert reloaded.config.id2label[0] == LABELS[0]


def test_model_card_has_the_required_parts(tmp_path):
    text = write_model_card(tmp_path, FAKE_FINAL).read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "base_model: distilbert/distilbert-base-uncased" in text
    assert "license:" in text
    for heading in CARD_HEADINGS:
        assert heading in text
    assert "invented" in text.split("## Limitations")[1]
    assert "400 training examples" in text


def test_app_returns_a_score_for_every_queue(saved):
    from app import make_classify

    classify = make_classify(load_classifier(saved[0]))
    result = classify("The kiosk screen is frozen and I cannot unlock a bike.")
    assert set(result) == set(LABELS)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
def test_app_gives_a_clear_error_for_empty_text(saved, empty):
    import gradio as gr

    from app import make_classify

    classifier = load_classifier(saved[0])
    with pytest.raises(ValueError, match=EMPTY_MESSAGE):
        label_scores(classifier, empty)
    with pytest.raises(gr.Error):
        make_classify(classifier)(empty)
