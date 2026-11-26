"""Pin every requirements file to the versions installed in this environment.

Run once after the environment works:  python pin_versions.py
"""

from importlib.metadata import version
from pathlib import Path

MAIN = [
    "torch",
    "transformers",
    "accelerate",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "gradio",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
]
DEV = ["pytest", "ruff"]
SPACE = ["torch", "transformers"]  # gradio comes with the Space's sdk


def pins(names: list[str]) -> str:
    # drop a local build tag such as +cpu, which plain pip cannot find
    return "".join(f"{name}=={version(name).split('+')[0]}\n" for name in names)


if __name__ == "__main__":
    Path("requirements.txt").write_text(pins(MAIN))
    Path("requirements-dev.txt").write_text("-r requirements.txt\n" + pins(DEV))
    Path("space/requirements.txt").write_text(pins(SPACE))
    print(Path("requirements.txt").read_text())
