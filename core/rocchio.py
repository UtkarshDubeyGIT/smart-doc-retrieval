"""
Rocchio Algorithm for Relevance Feedback.
Modifies query representation based on user-designated relevant and non-relevant documents.

Formula:
  q_new = alpha * q_0 + (beta / |D_r|) * sum(d_r for d_r in D_r) - (gamma / |D_nr|) * sum(d_nr for d_nr in D_nr)
Negative component values are clipped to 0 (as term frequencies cannot be negative in standard vector space).
"""
import math
from typing import Dict, List

class RocchioRelevanceFeedback:
    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.25):
        """
        :param alpha: Weight of original query vector.
        :param beta: Weight of relevant documents centroid.
        :param gamma: Weight of non-relevant documents centroid.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update_query_vector(
        self,
        query_vec: Dict[int, float],
        relevant_vecs: List[Dict[int, float]],
        non_relevant_vecs: List[Dict[int, float]]
    ) -> Dict[int, float]:
        """
        Computes updated query vector using the standard Rocchio formulation and L2-normalizes it.
        """
        new_vec: Dict[int, float] = {}
        all_features = set(query_vec.keys())
        for v in relevant_vecs:
            all_features.update(v.keys())
        for v in non_relevant_vecs:
            all_features.update(v.keys())

        num_rel = len(relevant_vecs)
        num_non_rel = len(non_relevant_vecs)

        for feat in all_features:
            q_val = query_vec.get(feat, 0.0)
            
            rel_sum = sum(v.get(feat, 0.0) for v in relevant_vecs) if num_rel > 0 else 0.0
            non_rel_sum = sum(v.get(feat, 0.0) for v in non_relevant_vecs) if num_non_rel > 0 else 0.0

            rel_contrib = (self.beta / num_rel) * rel_sum if num_rel > 0 else 0.0
            non_rel_contrib = (self.gamma / num_non_rel) * non_rel_sum if num_non_rel > 0 else 0.0

            updated_val = (self.alpha * q_val) + rel_contrib - non_rel_contrib
            # Clip negative values
            if updated_val > 0.0:
                new_vec[feat] = updated_val

        # L2-normalize updated vector
        norm = math.sqrt(sum(v * v for v in new_vec.values()))
        if norm > 0.0:
            return {feat: v / norm for feat, v in new_vec.items()}
        return new_vec
