# Natural Language Processing with Hugging Face Transformers

Generative AI guided project on Cognitive Class by IBM, completed as Infinite Learning Task 14.

## Name: Muhammad Zaenal Abidin Abdurrahman

## Assignment Context

- Program: AI Development - Infinite Learning
- Task: Tugas 14 - Coding: NLP with HuggingFace Transformers
- Topic: Generative AI and Natural Language Processing
- Repository name: `NLP_with_HF_Transformers`
- Main notebook: `hugging_face_transformers.ipynb`

## My Todo

### 1. Example 1 - Sentiment Analysis

```python
# TODO 1 - Sentiment Analysis
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

todo_1_text = (
    "I used to feel confused when learning NLP, "
    "but running the model step by step makes the topic much clearer."
)

todo_1_result = classifier(todo_1_text)
todo_1_result
```

Result:

```python
[{'label': 'POSITIVE', 'score': 0.9923784732818604}]
```

Analysis on example 1:

The sentiment analysis model predicts the sentence as positive because the overall meaning shows learning progress. Although the sentence starts with confusion, the final message explains that running the model step by step makes the topic clearer. The confidence score represents the model's confidence for this prediction, not an absolute guarantee that the interpretation is always correct.

### 2. Example 2 - Topic Classification

```python
# TODO 2 - Topic Classification
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

todo_2_text = (
    "The student built a local notebook environment, "
    "ran a Hugging Face model, and documented the prediction result."
)

todo_2_labels = ["machine learning", "cooking", "travel"]

todo_2_result = classifier(
    todo_2_text,
    candidate_labels=todo_2_labels,
)
todo_2_result
```

Result:

```python
{
    'sequence': 'The student built a local notebook environment, ran a Hugging Face model, and documented the prediction result.',
    'labels': ['machine learning', 'travel', 'cooking'],
    'scores': [0.9374801516532898, 0.039210930466651917, 0.0233088880777359],
}
```

Analysis on example 2:

The zero-shot classifier assigns the sentence to the machine learning topic because the text mentions a notebook environment, a Hugging Face model, and a prediction result. The other candidate labels, cooking and travel, are unrelated to the main activity described in the sentence. The score should be read as the model's confidence among the provided candidate labels, not as absolute proof that the label is always correct.

### 3. Example 3 and 3.5 - Text Generator and Masked Language Modeling

```python
# TODO 3 - Text Generation
generator = pipeline("text-generation", model="distilgpt2")

todo_3_prompt = (
    "After practicing NLP with Hugging Face pipelines, "
    "I started to understand how"
)

todo_3_result = generator(
    todo_3_prompt,
    max_length=45,
    num_return_sequences=2,
    do_sample=True,
    temperature=0.8,
    pad_token_id=generator.tokenizer.eos_token_id,
)
todo_3_result
```

Result:

```python
[
    {
        'generated_text': 'After practicing NLP with Hugging Face pipelines, I started to understand how it works. I am not a NLP supporter, but I understand how it works. In one of my projects I made a blog that showed how'
    },
    {
        'generated_text': 'After practicing NLP with Hugging Face pipelines, I started to understand how much of a culture, especially in the United States, is shaped by the way the world works.\n\n\nFor example, a recent study found'
    },
]
```

Analysis on example 3:

The text-generation model continues the unfinished prompt by producing two possible completions. The outputs may not be identical because generation can sample different next tokens from the model's probability distribution. The prompt gives the model context about practicing NLP with Hugging Face pipelines, so a useful completion should stay close to learning, models, or text-processing concepts. Because this is open-ended generation, the result should be reviewed for coherence rather than treated as a single correct answer.

```python
# TODO 3.5 - Masked Language Modeling
unmasker = pipeline("fill-mask", "distilroberta-base")

todo_35_sentence = "Hugging Face pipelines make NLP <mask> for beginners."

todo_35_result = unmasker(todo_35_sentence, top_k=4)
todo_35_result
```

Result:

```python
[
    {
        'score': 0.14113889634609222,
        'token': 3013,
        'token_str': ' easier',
        'sequence': 'Hugging Face pipelines make NLP easier for beginners.',
    },
    {
        'score': 0.11336222290992737,
        'token': 1365,
        'token_str': ' easy',
        'sequence': 'Hugging Face pipelines make NLP easy for beginners.',
    },
    {
        'score': 0.07402745634317398,
        'token': 5616,
        'token_str': ' useful',
        'sequence': 'Hugging Face pipelines make NLP useful for beginners.',
    },
    {
        'score': 0.044157929718494415,
        'token': 6500,
        'token_str': ' accessible',
        'sequence': 'Hugging Face pipelines make NLP accessible for beginners.',
    },
]
```

Analysis on example 3.5:

The fill-mask pipeline predicts candidate words for the `<mask>` token inside a fixed sentence. In this example, the sentence is about Hugging Face pipelines making NLP more approachable for beginners, so the strongest predictions describe ease, accessibility, or learning support. Compared with open-ended text generation, masked language modeling is more constrained because the sentence structure is already provided and the model only fills the missing token.

### 4. Example 4 - Named Entity Recognition (NER)

```python
# TODO 4 - Named Entity Recognition
ner = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
)

todo_4_sentence = (
    "Satya Nadella works at Microsoft in Seattle "
    "and visited Telkom University in Bandung."
)

todo_4_result = ner(todo_4_sentence)
todo_4_result
```

Result:

```python
[
    {'entity_group': 'PER', 'score': 0.9937351, 'word': 'Satya Nadella', 'start': 0, 'end': 13},
    {'entity_group': 'ORG', 'score': 0.99953735, 'word': 'Microsoft', 'start': 23, 'end': 32},
    {'entity_group': 'LOC', 'score': 0.9982046, 'word': 'Seattle', 'start': 36, 'end': 43},
    {'entity_group': 'ORG', 'score': 0.9960933, 'word': 'Telkom University', 'start': 56, 'end': 73},
    {'entity_group': 'LOC', 'score': 0.99456966, 'word': 'Bandung', 'start': 77, 'end': 84},
]
```

Analysis on example 4:

The NER pipeline identifies named entities in the sentence and groups them into categories such as person, organization, and location. In this example, the sentence includes a person name, company name, university name, and city names, so the model returns a mix of `PER`, `ORG`, and `LOC` entities. The `aggregation_strategy="simple"` parameter is used instead of the older `grouped_entities=True` argument so the notebook stays compatible with the installed Transformers version.

### 5. Example 5 - Question Answering

```python
# TODO 5 - Question Answering
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

todo_5_question = "What library provides pipelines for NLP tasks?"
todo_5_context = (
    "Hugging Face Transformers is a Python library that provides "
    "pipelines for common NLP tasks such as sentiment analysis, "
    "question answering, summarization, and translation."
)

todo_5_result = qa_model(question=todo_5_question, context=todo_5_context)
todo_5_result
```

Result:

```python
{
    'score': 0.8781073689460754,
    'start': 0,
    'end': 25,
    'answer': 'Hugging Face Transformers',
}
```

Analysis on example 5:

The question-answering pipeline extracts an answer span from the provided context instead of generating a free-form response. In this example, the question asks which library provides pipelines for NLP tasks, and the context explicitly mentions Hugging Face Transformers as that library. The `answer` field points to a phrase that appears directly in the context, while the `score` represents the model's confidence in that extracted span.

### 6. Example 6 - Text Summarization

```python
# TODO 6 - Text Summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

todo_6_text = """
Natural language processing helps students turn unstructured text into useful information. In this assignment, I practiced several Hugging Face pipelines, including sentiment analysis, topic classification, text generation, masked language modeling, named entity recognition, and question answering. Each pipeline solves a different problem, but they all show how pretrained transformer models can reduce the amount of code needed to build NLP applications. Instead of training a model from zero, I can load an existing model, prepare a clear input, run the pipeline, and interpret the output carefully. This workflow is helpful for beginners because it connects theory with visible results. However, the outputs still need human review because models can be uncertain, incomplete, or context-dependent. By testing every task step by step, I can understand both the usefulness and the limitations of transformer-based NLP. It also makes documentation easier for future review.
"""

todo_6_result = summarizer(
    todo_6_text,
    max_length=80,
    min_length=30,
    do_sample=False,
)
todo_6_result
```

Result:

```python
[
    {
        'summary_text': ' In this assignment, students learn how pretrained transformer models can reduce the amount of code needed to build NLP applications . This workflow is helpful for beginners because it connects theory with visible results . However, the outputs still need human review because models can be uncertain, incomplete, or context-dependent .'
    }
]
```

Analysis on example 6:

The summarization pipeline compresses a longer paragraph into a shorter version while trying to preserve the main ideas. In this example, the input paragraph explains how Hugging Face pipelines help learners practice multiple NLP tasks and why model outputs still need human review. The result is shorter than the original paragraph and keeps the main points about pretrained models, visible learning results, and careful interpretation.

### 7. Example 7 - Translation

```python
# TODO 7 - Indonesian to French Translation
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")

todo_7_sentence = "Saya sedang belajar pemrosesan bahasa alami di Bandung hari ini."

todo_7_result = translator_id(todo_7_sentence)
todo_7_result
```

Result:

```python
[{'translation_text': "J'étudie le langage naturel à Bandung aujourd'hui."}]
```

Analysis on example 7:

The translation pipeline converts an Indonesian sentence into French using a model trained for the Indonesian-to-French language pair. In this example, the source sentence says that I am learning natural language processing in Bandung today, so the translation preserves the learning activity, the NLP topic, and the location. The output is not only a word-by-word replacement; it tries to produce a natural sentence in the target language.

---

## Analysis on This Project

This project provides a practical introduction to Natural Language Processing with Hugging Face Transformers. The notebook covers several common NLP workflows, including sentiment analysis, zero-shot topic classification, text generation, masked language modeling, named entity recognition, question answering, summarization, and translation.

The main insight from this project is that pretrained Transformer models make NLP experimentation more accessible. With the `pipeline` API, each task can be tested with only a small amount of Python code. However, model outputs still require careful interpretation. Scores represent model confidence, not guaranteed truth, and generative outputs can be fluent while still being incomplete, unexpected, or contextually weak.

This assignment also shows the importance of a reproducible local environment. The repository uses `requirements.txt` and a project-specific virtual environment so the notebook can be rerun without relying only on Google Colab.

## Practice Exercise Summary

The notebook also includes additional practice exercises after the main TODO sections.

| Exercise   | Task                     | Result summary                                                                         |
| ---------- | ------------------------ | -------------------------------------------------------------------------------------- |
| Exercise 1 | Tweet sentiment analysis | Compared a tweet-specific sentiment model with the default sentiment pipeline.         |
| Exercise 2 | Zero-shot classification | Classified an NLP learning sentence under the `education` label.                       |
| Exercise 3 | Text generation          | Generated a deterministic continuation with GPT-2.                                     |
| Exercise 4 | Named entity recognition | Extracted `Anjela`, `IBM`, and `Seoul` as person, organization, and location entities. |
| Exercise 5 | Question answering       | Extracted `Hugging Face Transformers` from a custom context.                           |
| Exercise 6 | Text summarization       | Summarized a paragraph about transformer pipelines for beginners.                      |
| Exercise 7 | Translation              | Translated an English NLP sentence into German.                                        |

## Repository Contents

| File                              | Description                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------------------- |
| `hugging_face_transformers.ipynb` | Completed notebook with examples, TODO sections, practice exercises, outputs, and written analysis. |
| `requirements.txt`                | Python dependencies for running the notebook locally.                                               |
| `.gitignore`                      | Ignore rules for local environments, cache files, and notebook checkpoints.                         |
| `README.md`                       | Assignment summary, results, setup instructions, and project analysis.                              |

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

## Reproducibility Notes

- The repository uses `transformers[sentencepiece]==4.41.2` to keep the notebook compatible with the pipeline APIs used in the assignment.
- The NER examples use `aggregation_strategy="simple"` instead of the older `grouped_entities=True` parameter.
- Text-generation output can vary when sampling is enabled, so generated text should be interpreted as one possible continuation rather than a fixed answer.
- The original notebook came from a Colab-style guided project, but this repository is organized to run locally through `requirements.txt`.
