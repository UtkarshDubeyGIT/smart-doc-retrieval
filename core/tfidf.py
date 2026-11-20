"""
TF-IDF Vectorizer and Cosine Similarity Engine.
Implements standard Term Frequency (TF), Inverse Document Frequency (IDF),
L2-normalized document & query vector representation, and Cosine Similarity calculation.
"""
import math
from typing import Dict, List, Tuple
from collections import Counter
from .preprocessor import preprocess_text

class TFIDFVectorizer:
    def __init__(self, use_sublinear_tf: bool = True, smooth_idf: bool = True):
        """
        :param use_sublinear_tf: If True, uses 1 + ln(tf) for tf > 0 to dampen effect of repeated terms.
        :param smooth_idf: If True, adds 1 to document frequencies to prevent division by zero.
        """
        self.use_sublinear_tf = use_sublinear_tf
        self.smooth_idf = smooth_idf
        self.vocabulary: Dict[str, int] = {}
        self.idf_values: Dict[str, float] = {}
        self.doc_count: int = 0
        self.doc_vectors: List[Dict[int, float]] = []
        self.doc_ids: List[str] = []

    def fit_transform(self, documents: List[str], doc_ids: List[str] = None) -> List[Dict[int, float]]:
        """
        Learns vocabulary and IDF from documents and transforms documents into TF-IDF sparse vectors.
        """
        self.doc_count = len(documents)
        self.doc_ids = doc_ids if doc_ids is not None else [f"doc_{i+1}" for i in range(self.doc_count)]
        
        # Tokenize all documents
        tokenized_docs = [preprocess_text(doc) for doc in documents]
        
        # Build vocabulary & document frequency (DF)
        df_counter = Counter()
        vocab_set = set()
        for tokens in tokenized_docs:
            unique_terms = set(tokens)
            df_counter.update(unique_terms)
            vocab_set.update(unique_terms)
            
        # Assign sorted vocabulary indexes for determinism
        sorted_vocab = sorted(list(vocab_set))
        self.vocabulary = {term: idx for idx, term in enumerate(sorted_vocab)}
        
        # Compute IDF values
        # Formula: idf(t) = ln((N + 1) / (df(t) + 1)) + 1 if smooth_idf else ln(N / df(t))
        self.idf_values = {}
        for term, df in df_counter.items():
            if self.smooth_idf:
                idf = math.log((self.doc_count + 1.0) / (df + 1.0)) + 1.0
            else:
                idf = math.log(float(self.doc_count) / max(1.0, float(df)))
            self.idf_values[term] = idf
            
        # Vectorize each document and L2-normalize
        self.doc_vectors = []
        for tokens in tokenized_docs:
            vec = self._vectorize_tokens(tokens)
            self.doc_vectors.append(vec)
            
        return self.doc_vectors

    def _vectorize_tokens(self, tokens: List[str]) -> Dict[int, float]:
        """
        Converts token list to sparse vector {feature_idx: tf_idf_weight} normalized to unit L2 norm.
        """
        term_counts = Counter(tokens)
        raw_vec: Dict[int, float] = {}
        
        for term, count in term_counts.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                # TF calculation
                if self.use_sublinear_tf:
                    tf = 1.0 + math.log(count)
                else:
                    tf = float(count)
                idf = self.idf_values.get(term, 0.0)
                raw_vec[idx] = tf * idf
                
        # L2-normalization
        norm = math.sqrt(sum(val * val for val in raw_vec.values()))
        if norm > 0.0:
            return {idx: val / norm for idx, val in raw_vec.items()}
        return raw_vec

    def transform_query(self, query: str) -> Dict[int, float]:
        """
        Transforms user query string into a TF-IDF vector matching document vector space.
        """
        tokens = preprocess_text(query)
        return self._vectorize_tokens(tokens)

    @staticmethod
    def cosine_similarity(vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
        """
        Calculates cosine similarity between two sparse L2-normalized vectors.
        Because vectors are already unit-length, cosine similarity is simply their dot product.
        """
        # Iterate over smaller dictionary for performance
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1
            
        dot_product = 0.0
        for idx, val in vec1.items():
            if idx in vec2:
                dot_product += val * vec2[idx]
        return max(0.0, min(1.0, dot_product))

    def rank_documents(self, query_vec: Dict[int, float], filter_doc_indices: List[int] = None) -> List[Tuple[str, int, float]]:
        """
        Calculates cosine similarity of query_vec with all documents (or subset specified by filter_doc_indices).
        Returns list of (doc_id, doc_index, score) sorted descending by relevance score.
        """
        scores = []
        indices_to_evaluate = filter_doc_indices if filter_doc_indices is not None else range(len(self.doc_vectors))
        
        for idx in indices_to_evaluate:
            doc_id = self.doc_ids[idx]
            doc_vec = self.doc_vectors[idx]
            score = self.cosine_similarity(query_vec, doc_vec)
            scores.append((doc_id, idx, score))
            
        # Sort descending by score; secondary sort by doc_id for determinism
        scores.sort(key=lambda x: (x[2], -x[1]), reverse=True)
        return scores
