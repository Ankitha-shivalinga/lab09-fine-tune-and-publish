"""Gradio app that suggests a queue for a support message.

python app.py                       # loads the local model/ folder
python app.py your-name/model-repo  # loads a model from the Hub
"""

import sys

import gradio as gr

from share import MODEL_DIR, label_scores, load_classifier

EXAMPLES = [
    "My card got charged two times for one ride yesterday.",
    "The back brake on my bike barely works.",
    "Every dock at the station was full so I could not return the bike.",
    "I can't sign in since I changed my phone.",
]


def make_classify(classifier):
    def classify(text: str) -> dict[str, float]:
        try:
            return label_scores(classifier, text)
        except ValueError as error:
            raise gr.Error(str(error)) from error

    return classify


def build_demo(source=MODEL_DIR) -> gr.Interface:
    classify = make_classify(load_classifier(source))
    return gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=3, label="Support message"),
        outputs=gr.Label(num_top_classes=4, label="Suggested queue"),
        title="Lakeshore Ticket Sorter",
        description=(
            "Suggests a queue (billing, bike_fault, station, account) for a short "
            "support message. Trained on invented messages for a course lab."
        ),
        examples=EXAMPLES,
        flagging_mode="never",
    )


if __name__ == "__main__":
    build_demo(sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR).launch()
