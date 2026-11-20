"""
Comprehensive test suite for:
- Text Preprocessing
- TF-IDF Vectorizer & Cosine Similarity
- Boolean Query Parsing (AND, OR, NOT, multi-term, nested)
- Rocchio Relevance Feedback
- Evaluator Metrics
"""
import pytest
from core.preprocessor import preprocess_text, SimpleStemmer
from core.tfidf import TFIDFVectorizer
from core.boolean_parser import BooleanQueryParser
from core.rocchio import RocchioRelevanceFeedback
from core.evaluator import SearchEvaluator

def test_preprocessor():
    text = "Information Retrieval and Search Systems are Learning!"
    tokens = preprocess_text(text)
    # Stop words like 'and', 'are' should be removed
    assert "and" not in tokens
    assert "are" not in tokens
    # Stemmed variants
    assert any("retriev" in t for t in tokens)
    assert any("system" in t for t in tokens)

def test_tfidf_and_cosine_similarity():
    docs = [
        "information retrieval search engine vector space",
        "deep neural networks language models natural language",
        "space telescope astronomy stars galaxies cosmos"
    ]
    ids = ["doc1", "doc2", "doc3"]
    
    vectorizer = TFIDFVectorizer()
    vectorizer.fit_transform(docs, ids)
    
    # Query matching doc1
    q1 = "information retrieval vector"
    q1_vec = vectorizer.transform_query(q1)
    ranks1 = vectorizer.rank_documents(q1_vec)
    
    assert ranks1[0][0] == "doc1"
    assert ranks1[0][2] > 0.0
    
    # Query matching doc3
    q3 = "astronomy telescope cosmos"
    q3_vec = vectorizer.transform_query(q3)
    ranks3 = vectorizer.rank_documents(q3_vec)
    assert ranks3[0][0] == "doc3"

def test_boolean_parser():
    docs = [
        "quantum computing algorithms physics",       # 0
        "quantum machine learning artificial intelligence", # 1
        "deep learning neural networks image vision",  # 2
        "astronomy telescope galaxies stars"          # 3
    ]
    parser = BooleanQueryParser(docs)
    
    # AND operator
    res_and = parser.evaluate("quantum AND computing")
    assert res_and == {0}
    
    # OR operator
    res_or = parser.evaluate("physics OR astronomy")
    assert res_or == {0, 3}
    
    # AND with multiple
    res_and2 = parser.evaluate("quantum AND learning")
    assert res_and2 == {1}
    
    # Parentheses & combination
    res_complex = parser.evaluate("(quantum OR neural) AND learning")
    assert res_complex == {1, 2}

def test_rocchio_relevance_feedback():
    docs = [
        "information retrieval systems ranking algorithms tfidf", # 0
        "relevance feedback rocchio query expansion precision",   # 1
        "mars rovers planetary astronomy space exploration",      # 2
    ]
    ids = ["d1", "d2", "d3"]
    vectorizer = TFIDFVectorizer()
    vectorizer.fit_transform(docs, ids)

    # Initial query about retrieval
    query = "retrieval algorithms"
    q_vec = vectorizer.transform_query(query)
    
    initial_ranks = vectorizer.rank_documents(q_vec)
    assert initial_ranks[0][0] == "d1"

    # User marks d2 (relevance feedback) as relevant, and d3 as irrelevant
    rocchio = RocchioRelevanceFeedback(alpha=1.0, beta=0.8, gamma=0.3)
    d2_idx = ids.index("d2")
    d3_idx = ids.index("d3")
    
    updated_q_vec = rocchio.update_query_vector(
        query_vec=q_vec,
        relevant_vecs=[vectorizer.doc_vectors[d2_idx]],
        non_relevant_vecs=[vectorizer.doc_vectors[d3_idx]]
    )

    # Re-rank
    new_ranks = vectorizer.rank_documents(updated_q_vec)
    # d2 score should have significantly increased due to positive feedback
    d2_score_before = next(r[2] for r in initial_ranks if r[0] == "d2")
    d2_score_after = next(r[2] for r in new_ranks if r[0] == "d2")
    assert d2_score_after > d2_score_before

def test_evaluator_metrics():
    before = [("d1", 0, 0.9), ("d2", 1, 0.5), ("d3", 2, 0.1)]
    after = [("d2", 1, 0.95), ("d1", 0, 0.8), ("d3", 2, 0.05)]
    relevant = {"d2"}
    
    eval_result = SearchEvaluator.compare_rankings(before, after, relevant, top_k=2)
    assert eval_result["precision_after"] >= eval_result["precision_before"]
    assert len(eval_result["comparison_table"]) == 2
