# scaling-invent
# RAG Hallucination Mini Study

A small experiment investigating whether Retrieval-Augmented Generation (RAG) reduces hallucinations in large language models.

## Research Question

> Does providing an LLM with relevant retrieved information reduce unsupported or incorrect answers?

## Experiment

This project compares two approaches:

### 1. LLM Only

The question is sent directly to the language model.

### 2. RAG

Relevant information is retrieved from a small knowledge base and provided to the language model before answering.

The same questions are tested against both approaches.

## Dataset

The experiment contains:

* Answerable questions
* Unanswerable questions
* Ambiguous questions
* Adversarial questions

The knowledge base contains fictional information about a company called Acme Corporation.

Using fictional information makes the experiment reproducible and avoids relying on potentially changing external information.

## Metrics

The experiment records:

* Whether the expected information appears in the answer
* Whether the answer contains an abstention
* Whether the RAG system retrieved relevant information
* Potential hallucination cases

These metrics are intended as experimental signals rather than definitive measures of factuality.

## Example

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

## Why This Matters

RAG is often described as a way to reduce hallucinations by grounding an LLM in external information.

This experiment tests a more specific question:

> Does retrieval actually prevent hallucination, or can an LLM still produce unsupported information even when relevant context is provided?

## Running the Experiment

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

## Limitations

This is a small exploratory experiment.

The dataset is intentionally small, and the evaluation uses simple automated signals. A larger study should use more questions, multiple models, semantic retrieval, human evaluation, and more rigorous factuality metrics.

## Author

Independent research into LLM reliability, RAG, and AI security.
