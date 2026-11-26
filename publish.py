"""Upload the saved model (with its card) and the Space to the Hugging Face Hub.

Sign in once on your own computer with `hf auth login`. No token goes in this file.

    python publish.py
    python publish.py --model-name lakeshore-ticket-sorter --space-name ticket-sorter
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from share import CARD_HEADINGS, MODEL_DIR

SPACE_DIR = Path("space")


def check_ready(model_id: str) -> None:
    card = MODEL_DIR / "README.md"
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit("model/ is missing: run python run_experiments.py first")
    if not card.exists() or not all(h in card.read_text() for h in CARD_HEADINGS):
        raise SystemExit("model/README.md (the model card) is missing or incomplete")
    if model_id not in (SPACE_DIR / "app.py").read_text():
        raise SystemExit(f'Set MODEL_ID = "{model_id}" in space/app.py first')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="lakeshore-ticket-sorter")
    parser.add_argument("--space-name", default="lakeshore-ticket-sorter")
    args = parser.parse_args()

    api = HfApi()  # uses the token saved by `hf auth login`
    user = api.whoami()["name"]
    model_id = f"{user}/{args.model_name}"
    space_id = f"{user}/{args.space_name}"
    check_ready(model_id)

    api.create_repo(model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=model_id,
        repo_type="model",
        commit_message="Upload fine-tuned model and model card",
    )
    print(f"model: https://huggingface.co/{model_id}")

    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=SPACE_DIR,
        repo_id=space_id,
        repo_type="space",
        commit_message="Upload Gradio app",
    )
    print(f"space: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
