"""The Lakeshore Ticket Sorter as it runs on a Hugging Face Space."""

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Change this to your own model repository before you publish.
MODEL_ID = "YOUR-HF-USERNAME/lakeshore-ticket-sorter"
MAX_LENGTH = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()


@torch.no_grad()
def classify(text: str) -> dict[str, float]:
    if text is None or not text.strip():
        raise gr.Error("Please type a support message first.")
    batch = tokenizer(
        [text.strip()], truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    probabilities = torch.softmax(model(**batch).logits, dim=-1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probabilities)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Support message"),
    outputs=gr.Label(num_top_classes=4, label="Suggested queue"),
    title="Lakeshore Ticket Sorter",
    description=(
        "Suggests a queue (billing, bike_fault, station, account) for a short "
        f"support message. Model: {MODEL_ID}. Trained on invented messages for a "
        "course lab, so treat the scores as suggestions."
    ),
    examples=[
        "My card got charged two times for one ride yesterday.",
        "The back brake on my bike barely works.",
        "Every dock at the station was full so I could not return the bike.",
        "I can't sign in since I changed my phone.",
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
