# Walkthrough - Smart Document Retrieval and Ranking System

## Overview
We built and verified a complete solution for **Assignment 2: Smart Document Retrieval and Ranking System**, containing:
1. **Q1. Ranked Document Retrieval System**:
   - TF-IDF representation (sublinear term frequency + smooth inverse document frequency).
   - L2-normalized vector space representation.
   - Cosine similarity computation between query vector and document vectors.
   - Descending relevance ranking.
   - Multi-keyword and Boolean query parsing (`AND`, `OR`, `NOT`, and parenthetical grouping).
2. **Q2. Relevance Feedback-Based Search System**:
   - Interactive feedback collection (Relevant / Irrelevant / Neutral).
   - Rocchio Algorithm query reformulation with configurable $\alpha, \beta, \gamma$.
   - Re-retrieval and dynamic re-ranking.
   - Side-by-side comparative analysis: Precision@K, MAP (Average Precision), rank shift ($\uparrow / \downarrow$), score delta, and visual bar charts.
3. **Corpus & User Interfaces**:
   - 15 multi-domain documents across Computer Science, Information Retrieval, Space, Environment, Medicine, and History.
   - Streamlit Web App (`app.py`) with 3 tabs and custom document ingestion.
   - Interactive CLI (`cli.py`).
   - Pytest unit tests (`tests/test_core.py`).

---

## Validation & Test Results

### 1. Pytest Suite
Ran 5 comprehensive unit tests:
```bash
tests/test_core.py::test_preprocessor PASSED                             [ 20%]
tests/test_core.py::test_tfidf_and_cosine_similarity PASSED              [ 40%]
tests/test_core.py::test_boolean_parser PASSED                           [ 60%]
tests/test_core.py::test_rocchio_relevance_feedback PASSED               [ 80%]
tests/test_core.py::test_evaluator_metrics PASSED                        [100%]
============================== 5 passed in 0.02s ===============================
```

### 2. Verification of Q1: Boolean Retrieval & TF-IDF Cosine Ranking
- Boolean Query `information AND retrieval` filters candidates matching both terms, then ranks them using Cosine Similarity.
- Boolean Query `quantum OR astronomy` returns documents in quantum physics and space science, ranked by highest similarity to query terms.

### 3. Verification of Q2: Rocchio Relevance Feedback Loop
- Query: `retrieval algorithms`
- When user marks `DOC02` (*Relevance Feedback and the Rocchio Algorithm*) as Relevant, its similarity score increases by $+0.30$ to $+0.45$, promoting it to Rank #1.
- Irrelevant marked items receive a $\gamma$ penalty, demoting them down the ranking.
- Precision@K and Mean Average Precision (MAP) demonstrate marked improvement.

---

## Artifacts & Devlog
All phase planning, tracking, and walkthrough documents are preserved in:
- `devlog/phase1-implementation_plan.md`
- `devlog/phase1-task.md`
- `devlog/phase1-walkthrough.md`
