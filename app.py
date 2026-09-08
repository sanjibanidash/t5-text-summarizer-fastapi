from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# --------------------------------------------------
# Load trained model and tokenizer
# --------------------------------------------------

MODEL_PATH = "./saved_summary_model"

model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)


# --------------------------------------------------
# Select device
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# HTML templates
# --------------------------------------------------

templates = Jinja2Templates(directory=".")


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class DialogueInput(BaseModel):
    dialogue: str


# --------------------------------------------------
# Data cleaning
# --------------------------------------------------

def clean_data(text: str) -> str:

    # Replace line breaks
    text = re.sub(r"\r\n|\r|\n", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip().lower()

    return text


# --------------------------------------------------
# Summarization function
# --------------------------------------------------

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    # T5 was trained with the summarization prefix
    dialogue = "summarize: " + dialogue

    # Tokenize input
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to the selected device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate summary
    with torch.no_grad():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Convert token IDs back into text
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# --------------------------------------------------
# API endpoint
# --------------------------------------------------

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )