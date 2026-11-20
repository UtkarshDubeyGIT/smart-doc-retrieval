# Implementation Plan - Smart Document Retrieval and Ranking System

## Objective
Build a complete, modular, and interactive system addressing:
- **Q1. Ranked Document Retrieval System**:
  - TF-IDF representation for documents and queries.
  - Cosine similarity computation between query vector and document vectors.
  - Ranking and displaying documents in descending order of relevance scores.
  - Multi-keyword query support with Boolean operators (`AND`, `OR`).
- **Q2. Relevance Feedback-Based Search System**:
  - Interactive user feedback collection (relevant / irrelevant tagging).
  - Rocchio Algorithm / Query modification using the collected feedback ($\vec{q}_{new} = \alpha \vec{q}_0 + \beta \frac{1}{|D_r|} \sum_{d \in D_r} \vec{d} - \gamma \frac{1}{|D_{nr}|} \sum_{d \in D_{nr}} \vec{d}$).
  - Re-retrieval and re-ranking of documents.
  - Side-by-side or before/after comparison analysis demonstrating retrieval quality improvement (precision, rank shifts, score differences).

In addition, provide:
- Comprehensive sample document corpus (across tech, science, history, space, AI, etc.) + custom document upload/creation capability.
- Clean CLI interface and interactive Web UI (Streamlit or FastAPI/React/HTML).
- Unit tests validating TF-IDF, Boolean evaluation, Cosine Similarity, and Rocchio feedback.
- GitHub repository creation and push under `smart-doc-retrieval`.

## System Architecture

1. **`core/preprocessor.py`**: Tokenization, lowercasing, stopword removal, stemming/lemmatization.
2. **`core/tfidf.py`**: Vocabulary building, Term Frequency (TF), Inverse Document Frequency (IDF), document & query vector generation, Cosine similarity.
3. **`core/boolean_parser.py`**: Shunting-yard or AST parser for Boolean search (`AND`, `OR`, multi-keyword phrases), yielding matching document candidates and filtered ranked retrieval.
4. **`core/rocchio.py`**: Rocchio relevance feedback vector recalculation with configurable $\alpha, \beta, \gamma$ hyperparameters.
5. **`core/evaluator.py`**: Evaluation metrics and before/after comparison diagnostics (MAP, MRR, rank shift analysis, overlap).
6. **`app.py` / `web_app.py`**: Streamlit application with modern UI tabs:
   - Ranked Search & Boolean Search
   - Relevance Feedback & Before/After Comparison
   - Document Corpus Explorer & Custom Document Ingestion
7. **`tests/`**: Unit test suite covering Q1 & Q2 core logic.

## Verification Plan
1. Run pytest suite on TF-IDF, Cosine Similarity, Boolean parsing, and Rocchio feedback.
2. Verify Boolean operators (`AND`, `OR`) both syntactically and logically on sample corpus.
3. Test Relevance Feedback loop: Tag top 2 documents as relevant, tag 1 as irrelevant, inspect new top-k rankings and cosine scores.
4. Verify Web UI loads and operates cleanly.
5. Push to GitHub with README, screenshots/guides, and devlog documentation.
