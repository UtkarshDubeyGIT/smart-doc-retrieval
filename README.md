# Smart Document Retrieval and Ranking System

A complete implementation of **Assignment 2: Smart Document Retrieval and Ranking System (Q1 & Q2)** featuring Vector Space TF-IDF modeling, Cosine Similarity ranking, Boolean Query parsing, and Rocchio Relevance Feedback re-ranking with side-by-side comparative analysis.

---

## 🌟 Key Features

### Q1: Ranked Document Retrieval System
- **TF-IDF Vector Space Model**: Custom term frequency (sublinear $1 + \ln(tf)$) and inverse document frequency (smoothed $\ln((N+1)/(df+1)) + 1$) representation.
- **Cosine Similarity Ranking**: Computes angular similarity between the L2-normalized query vector and document vectors:
  $$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2} = \vec{q}_{norm} \cdot \vec{d}_{norm}$$
- **Multi-Keyword & Boolean Search**: Supports boolean logical operators (`AND`, `OR`, `NOT`) and parenthetical groupings (e.g. `(quantum OR neural) AND learning`) using an Inverted Index and Shunting-Yard parser.
- **Descending Relevance Score Ordering**: Retrieved documents are returned sorted by descending similarity score.

### Q2: Relevance Feedback-Based Search System
- **User Feedback Collection**: Tag retrieved documents as *Relevant* ($\mathcal{D}_r$) or *Irrelevant* ($\mathcal{D}_{nr}$).
- **Rocchio Algorithm Query Reformulation**:
  $$\vec{q}_{new} = \alpha \vec{q}_0 + \frac{\beta}{|\mathcal{D}_r|} \sum_{\vec{d} \in \mathcal{D}_r} \vec{d} - \frac{\gamma}{|\mathcal{D}_{nr}|} \sum_{\vec{d} \in \mathcal{D}_{nr}} \vec{d}$$
  - Adjustable hyperparameters: $\alpha$ (query persistence), $\beta$ (positive reinforcement), $\gamma$ (negative penalty).
  - Negative feature values are clipped to zero.
- **Re-Retrieval & Dynamic Re-Ranking**: Computes updated cosine scores and re-ranks the collection in real-time.
- **Comparative Quality Analysis**:
  - Precision@K before vs. after feedback.
  - Mean Average Precision (MAP) improvements.
  - Rank shift tracking ($\uparrow$ / $\downarrow$) and cosine score delta comparisons.
  - Interactive visual bar chart displaying score shifts.

---

## 📂 Project Structure

```text
smart-doc-retrieval/
├── core/
│   ├── preprocessor.py       # Tokenization, stop words, stemming
│   ├── tfidf.py              # TF-IDF calculation & Cosine Similarity
│   ├── boolean_parser.py     # Inverted index & Shunting-Yard Boolean parser
│   ├── rocchio.py            # Rocchio relevance feedback vector recalculation
│   └── evaluator.py          # Precision@K, AP, rank shift comparative metrics
├── data/
│   └── corpus.py             # Multi-domain document corpus (AI, Space, Medicine, etc.)
├── tests/
│   └── test_core.py          # Pytest unit tests for Q1 and Q2 requirements
├── devlog/                   # Phase documentation & devlog tracking
│   ├── phase1-implementation_plan.md
│   ├── phase1-task.md
│   └── phase1-walkthrough.md
├── app.py                    # Interactive Streamlit Web Application
├── cli.py                    # Interactive Command-Line Interface
├── requirements.txt          # Python dependencies
└── README.md                 # Full documentation and guide
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/UtkarshDubeyGIT/smart-doc-retrieval.git
cd smart-doc-retrieval

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Interactive Web UI (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Interactive CLI
```bash
python3 cli.py
```

### 5. Run Test Suite
```bash
pytest -v
```

---

## 🧪 Verification & Examples

### Example 1: Multi-Keyword & Boolean Search (Q1)
- **Query**: `information retrieval AND vector`
  - Output: Retrieves `DOC01` (Introduction to IR and Vector Space Models) with highest relevance.
- **Query**: `(quantum OR neural) AND learning`
  - Output: Matches documents containing both quantum/neural and learning contexts.

### Example 2: Relevance Feedback & Rocchio Shift (Q2)
1. **Initial Search**: `retrieval algorithms`
2. **Top Results**: Documents covering general retrieval and algorithms.
3. **User Action**: Mark `DOC02` (Relevance Feedback and the Rocchio Algorithm) as **Relevant** and unrelated topics as **Irrelevant**.
4. **Resulting Improvement**:
   - `DOC02` score increases by +0.35+ and moves up to Rank #1.
   - Precision@K and Mean Average Precision increase substantially.
   - Negative topics are demoted.

---

## 📚 Mathematical Formulations

### TF-IDF Weighting
- **Term Frequency**:
  $$\text{TF}(t, d) = 1 + \ln(\text{count}(t, d)) \quad \text{if } \text{count}(t, d) > 0 \text{ else } 0$$
- **Inverse Document Frequency**:
  $$\text{IDF}(t) = \ln\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1$$
- **Vector Normalization**:
  $$\vec{d}_{norm} = \frac{\vec{d}}{\|\vec{d}\|_2}$$

---

## 📄 License
MIT License. Created for Assignment 2 on Smart Document Retrieval and Ranking.
