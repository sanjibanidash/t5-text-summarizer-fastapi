# 📝 T5 Text Summarizer with FastAPI

An end-to-end NLP application that summarizes conversations using a **fine-tuned T5 Transformer model** from Hugging Face and serves predictions through a **FastAPI REST API** with a simple HTML frontend.

---

## 📌 Project Overview

Reading long conversations, customer interactions, meeting discussions, or support chats can be time-consuming.

This project builds an automated text summarization system that takes a dialogue as input and generates a concise summary containing the key information.

The project covers the complete workflow:

**Dataset → Data Preprocessing → T5 Fine-Tuning → Model Saving → Model Inference → FastAPI API → Web Interface**

---

## 🎯 Objective

Build an NLP-based text summarization system capable of:

- Understanding conversational text
- Generating concise summaries
- Serving predictions through a REST API
- Providing a simple web interface for users
- Demonstrating an end-to-end Transformer-based NLP workflow

---

## 🤖 Model

### T5 (Text-to-Text Transfer Transformer)

The project uses the **T5 Transformer architecture** from Hugging Face.

T5 treats NLP tasks as a text-to-text problem.

For summarization:

```text
Input Dialogue
      ↓
"Summarize: <dialogue>"
      ↓
Fine-tuned T5 Model
      ↓
Generated Summary

📊 Dataset

The project uses the SAMSum dataset, which contains human-written summaries of everyday conversations.

                  SAMSum Dataset
                        │
                        ▼
                Data Preprocessing
                        │
                        ▼
              T5 Tokenization
                        │
                        ▼
                T5 Fine-Tuning
                        │
                        ▼
              Trained T5 Model
                        │
                        ▼
              Saved Model & Tokenizer
                        │
                        ▼
              ┌───────────────────┐
              │    FastAPI API    │
              │   POST /summarize/│
              └─────────┬─────────┘
                        │
                        ▼
                Generated Summary
                        │
                        ▼
                HTML Web Interface
