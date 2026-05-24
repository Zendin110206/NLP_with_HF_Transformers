# Natural Language Processing with Hugging Face Transformers

This repository contains my Infinite Learning Task 14 assignment for practicing Natural Language Processing workflows with Hugging Face Transformers.

## Assignment Context

- Program: AI Development - Infinite Learning
- Task: Tugas 14 - Coding: NLP with HuggingFace Transformers
- Topic: Generative AI and Natural Language Processing
- Main notebook: `hugging_face_transformers.ipynb`

## Project Scope

The notebook introduces several NLP tasks using Hugging Face pipelines:

- sentiment analysis
- zero-shot topic classification
- text generation
- fill-mask prediction
- named entity recognition
- question answering
- text summarization
- translation

## Current Status

This repository is being completed in small checkpoints. The notebook has been added first so the assignment link is available in GitHub, and the analysis sections will be filled after each NLP task is executed and reviewed.

## Planned Deliverables

- Completed notebook with executed outputs.
- Written analysis for each TODO section.
- Final README summary based on the notebook results.

## Local Setup

This repository can be run locally with a project-specific Python virtual environment.

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name nlp-hf-transformers --display-name "NLP HF Transformers (.venv)"
```

After the environment is ready, open `hugging_face_transformers.ipynb` and select the `NLP HF Transformers (.venv)` kernel.
