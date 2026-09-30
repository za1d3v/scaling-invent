# scaling-invent
# rag hallucination mini study

A small experiment investigating whether Retrieval-Augmented Generation (RAG) reduces hallucinations in large language models.

## research question

> Does providing an LLM with relevant retrieved information reduce unsupported or incorrect answers?

## experiment

This project compares two approaches:

### 1. llm 0nly

The question is sent directly to the language model.

### 2. rag

Relevant information is retrieved from a small knowledge base and provided to the language model before answering.

The same questions are tested against both approaches.

## dataset

The experiment contains:

* Answerable questions
* Unanswerable questions
* Ambiguous questions
* Adversarial questions

The knowledge base contains fictional information about a company called Acme Corporation.

Using fictional information makes the experiment reproducible and avoids relying on potentially changing external information.

## metrics

The experiment records:

* Whether the expected information appears in the answer
* Whether the answer contains an abstention
* Whether the RAG system retrieved relevant information
* Potential hallucination cases

These metrics are intended as experimental signals rather than definitive measures of factuality.

## example

Without RAG:

```text
Question:
Who is the CEO of Acme Corporation?

LLM:
John Smith is the CEO of Acme Corporation.
```

With RAG:

```text
Question:
Who is the CEO of Acme Corporation?

RAG:
The knowledge base does not contain information identifying
Acme Corporation's CEO.
```

## why this matters

RAG is often described as a way to reduce hallucinations by grounding an LLM in external information.

This experiment tests a more specific question:

> Does retrieval actually prevent hallucination, or can an LLM still produce unsupported information even when relevant context is provided?

## running the experiment

Install the OpenAI Python library:

```bash
pip install openai
```

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Run:

```bash
python experiment.py
```

## limitations

This is a small exploratory experiment.

The dataset is intentionally small, and the evaluation uses simple automated signals. A larger study should use more questions, multiple models, semantic retrieval, human evaluation, and more rigorous factuality metrics.

## author

Independent research into LLM reliability, RAG, and AI security.
