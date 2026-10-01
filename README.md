# RAG Hallucination Mini Study

A small experimental study investigating whether Retrieval-Augmented Generation (RAG) reduces hallucinations in large language models.

## Research Question

Does providing an LLM with relevant retrieved information reduce unsupported or incorrect answers?

This study explores whether grounding an LLM with retrieved context improves answer reliability and examines situations where RAG may still fail.

## Experiment

The experiment compares two approaches:

### 1. LLM Only

The question is sent directly to the language model without additional context.

### 2. RAG

Relevant information is retrieved from a small knowledge base and provided to the language model before answering.

The same questions are tested against both approaches.

## Dataset

The dataset contains four types of questions:

- Answerable questions
- Unanswerable questions
- Ambiguous questions
- Adversarial questions

The knowledge base contains fictional information about a company called Acme Corporation.

Using fictional information makes the experiment reproducible and avoids relying on potentially changing external information.

## Metrics

The experiment records several simple evaluation signals:

- Whether the expected information appears in the answer
- Whether the model abstains when information is unavailable
- Whether the RAG system retrieves relevant information
- Potential hallucination cases
- Similarity between the generated answer and expected information

These metrics are intended as experimental signals rather than definitive measures of factuality.

## Example

### Without RAG

Question:
Who is the CEO of Acme Corporation?

LLM:
John Smith is the CEO of Acme Corporation.

### With RAG

Question:
Who is the CEO of Acme Corporation?

RAG:
The knowledge base does not contain information identifying Acme Corporation's CEO.

The example illustrates an important RAG behavior: when the knowledge base does not contain the requested information, the model should ideally abstain rather than invent an answer.

## Why This Matters

RAG is often described as a way to reduce hallucinations by grounding an LLM in external information.

However, retrieval does not automatically guarantee that an answer is correct.

This experiment investigates a more specific question:

Does retrieval actually prevent hallucination, or can an LLM still produce unsupported information even when relevant context is provided?

Potential failure cases include:

- Incorrect retrieval
- Missing information
- Ambiguous information
- Conflicting information
- Adversarial content
- The model ignoring retrieved context

## Running the Experiment

Install the required Python libraries:

pip install openai scikit-learn

Set your OpenAI API key.

Windows PowerShell:

$env:OPENAI_API_KEY = "your-api-key"

macOS / Linux:

export OPENAI_API_KEY = "your-api-key"

Run the experiment:

python experiment.py

## Limitations

This is a small exploratory experiment.

The dataset is intentionally small, and the evaluation uses simple automated signals. The results should not be interpreted as a definitive measurement of hallucination or factuality.

A larger study could include:

- More questions
- Multiple LLM providers and models
- Embedding-based retrieval
- Vector databases
- Semantic evaluation
- Human evaluation
- RAG poisoning experiments
- Prompt injection testing
- More rigorous factuality metrics

## Future Work

A natural extension of this project is to test whether malicious or misleading documents can cause a RAG system to produce incorrect answers.

This would allow the experiment to move beyond hallucination measurement and into practical RAG security testing.

## Author

Za1d3v

Independent research into LLM reliability, RAG, and AI security.
