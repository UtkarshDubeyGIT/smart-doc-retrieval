"""
Evaluation and Comparison Engine.
Compares ranked document lists before and after relevance feedback.
Calculates Rank Shift, Precision@K, Average Precision, and Overlap metrics.
"""
from typing import List, Tuple, Dict, Set

class SearchEvaluator:
    @staticmethod
    def compare_rankings(
        before_ranks: List[Tuple[str, int, float]],
        after_ranks: List[Tuple[str, int, float]],
        relevant_doc_ids: Set[str],
        top_k: int = 10
    ) -> Dict:
        """
        Calculates side-by-side metrics comparing top results before and after relevance feedback.
        """
        top_before = before_ranks[:top_k]
        top_after = after_ranks[:top_k]

        before_ids = [d[0] for d in top_before]
        after_ids = [d[0] for d in top_after]

        # Precision@K
        before_rel_count = sum(1 for doc_id in before_ids if doc_id in relevant_doc_ids)
        after_rel_count = sum(1 for doc_id in after_ids if doc_id in relevant_doc_ids)
        
        p_at_k_before = before_rel_count / max(1, len(before_ids))
        p_at_k_after = after_rel_count / max(1, len(after_ids))

        # Average Precision (AP@K)
        def calc_ap(ranked_ids: List[str]) -> float:
            hits = 0
            sum_prec = 0.0
            for i, doc_id in enumerate(ranked_ids):
                if doc_id in relevant_doc_ids:
                    hits += 1
                    sum_prec += hits / (i + 1)
            return sum_prec / max(1, len(relevant_doc_ids))

        ap_before = calc_ap(before_ids)
        ap_after = calc_ap(after_ids)

        # Rank position comparison for all documents
        rank_positions_before = {d[0]: i + 1 for i, d in enumerate(before_ranks)}
        rank_positions_after = {d[0]: i + 1 for i, d in enumerate(after_ranks)}
        
        scores_before = {d[0]: d[2] for d in before_ranks}
        scores_after = {d[0]: d[2] for d in after_ranks}

        comparison_table = []
        all_unique_top = list(dict.fromkeys(before_ids + after_ids))
        for doc_id in all_unique_top:
            pos_b = rank_positions_before.get(doc_id, None)
            pos_a = rank_positions_after.get(doc_id, None)
            sc_b = scores_before.get(doc_id, 0.0)
            sc_a = scores_after.get(doc_id, 0.0)
            
            # Rank shift: positive means moved up in ranking (e.g. from 5th to 2nd is +3)
            rank_shift = (pos_b - pos_a) if (pos_b is not None and pos_a is not None) else None
            score_diff = sc_a - sc_b
            
            comparison_table.append({
                "doc_id": doc_id,
                "is_relevant": doc_id in relevant_doc_ids,
                "rank_before": pos_b,
                "rank_after": pos_a,
                "rank_shift": rank_shift,
                "score_before": round(sc_b, 4),
                "score_after": round(sc_a, 4),
                "score_diff": round(score_diff, 4)
            })

        return {
            "top_k": top_k,
            "precision_before": round(p_at_k_before, 4),
            "precision_after": round(p_at_k_after, 4),
            "ap_before": round(ap_before, 4),
            "ap_after": round(ap_after, 4),
            "comparison_table": comparison_table
        }
