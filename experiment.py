import json
import os
import re

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL = "gpt-4o-mini"

KNOWLEDGE_BASE = """
Acme Corporation is a fictional technology company founded in 2018.

Acme Corporation is headquartered in Denver, Colorado.

Acme Corporation has approximately 350 employees.

Acme develops enterprise software and artificial intelligence tools.

Its primary AI product is called AcmeAI.

AcmeAI assists businesses with document analysis,
information retrieval, and automated text generation.

AcmeAI is not designed to make autonomous financial decisions.

Acme has a small engineering office in Austin, Texas.

Acme recommends the principle of least privilege
for AI applications.

High-impact or irreversible actions should receive
additional validation or human approval.

The knowledge base does not contain information about
Acme's annual revenue, stock price, valuation,
CEO, or executive compensation.

The knowledge base does not provide enough information
to determine whether Acme is financially successful.

The knowledge base does not provide enough information
to compare AcmeAI's security with other AI systems.

Acme acknowledges that generative AI systems can produce
incorrect or unsupported information.

Retrieval can improve access to relevant information,
but retrieval does not guarantee that generated responses
are factually correct.
"""


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "Please set the OPENAI_API_KEY environment variable."
    )

client = OpenAI()


# ---------------------------------------------------------
# Load questions
# ---------------------------------------------------------

with open("questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)


# ---------------------------------------------------------
# Simple RAG retriever
# ---------------------------------------------------------

documents = [
    paragraph.strip()
    for paragraph in KNOWLEDGE_BASE.split("\n\n")
    if paragraph.strip()
]

vectorizer = TfidfVectorizer(stop_words="english")

document_vectors = vectorizer.fit_transform(documents)


def retrieve(question, top_k=3):

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    best_indexes = similarities.argsort()[-top_k:][::-1]

    return [
        documents[index]
        for index in best_indexes
    ]


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

def ask_llm(prompt):

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# LLM-only experiment
# ---------------------------------------------------------

def llm_only(question):

    prompt = f"""
Answer the following question.

If you do not know the answer, say that you do not know.

Question:
{question}
"""

    return ask_llm(prompt)


# ---------------------------------------------------------
# RAG experiment
# ---------------------------------------------------------

def rag(question):

    retrieved = retrieve(question)

    context = "\n\n".join(retrieved)

    prompt = f"""
Answer the question using ONLY the information provided
in the knowledge base.

If the knowledge base does not contain enough information,
say that you do not have enough information.

Do not invent facts.

Knowledge base:

{context}

Question:

{question}
"""

    return ask_llm(prompt), retrieved


# ---------------------------------------------------------
# Simple evaluation
# ---------------------------------------------------------

def normalize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    return set(text.split())


def similarity(answer, expected):

    answer_words = normalize(answer)
    expected_words = normalize(expected)

    if not expected_words:
        return 0

    matches = answer_words.intersection(
        expected_words
    )

    return len(matches) / len(expected_words)


# ---------------------------------------------------------
# Run experiment
# ---------------------------------------------------------

print("=" * 60)
print("RAG HALLUCINATION MINI STUDY")
print("=" * 60)

for number, item in enumerate(questions, 1):

    question = item["question"]
    expected = item["answer"]
    category = item["category"]

    print()
    print(f"Question {number}")
    print(f"Category: {category}")
    print(f"Question: {question}")

    # LLM only
    llm_answer = llm_only(question)

    # RAG
    rag_answer, retrieved = rag(question)

    llm_score = similarity(
        llm_answer,
        expected
    )

    rag_score = similarity(
        rag_answer,
        expected
    )

    print()
    print("LLM ONLY:")
    print(llm_answer)

    print()
    print("RAG:")
    print(rag_answer)

    print()
    print(
        f"LLM similarity signal: {llm_score:.2f}"
    )

    print(
        f"RAG similarity signal: {rag_score:.2f}"
    )

    print()
    print("Retrieved context:")

    for document in retrieved:
        print("-", document[:150])

    print()
    print("-" * 60)
