"""
Smart Document Retrieval & Ranking System - Web UI
Built with Streamlit for Question 1 & Question 2.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.tfidf import TFIDFVectorizer
from core.boolean_parser import BooleanQueryParser
from core.rocchio import RocchioRelevanceFeedback
from core.evaluator import SearchEvaluator
from data.corpus import SAMPLE_DOCUMENTS

st.set_page_config(
    page_title="Smart Document Retrieval & Relevance Feedback",
    page_icon="🔍",
    layout="wide"
)

# Initialize Session State
if "corpus" not in st.session_state:
    st.session_state.corpus = [dict(d) for d in SAMPLE_DOCUMENTS]

if "rebuild_index" not in st.session_state:
    st.session_state.rebuild_index = True

def get_engine():
    corpus = st.session_state.corpus
    doc_ids = [doc["id"] for doc in corpus]
    doc_texts = [f"{doc['title']}. {doc['content']}" for doc in corpus]
    
    vectorizer = TFIDFVectorizer()
    vectorizer.fit_transform(doc_texts, doc_ids)
    bool_parser = BooleanQueryParser(doc_texts)
    return vectorizer, bool_parser

vectorizer, bool_parser = get_engine()

# Sidebar: Hyperparameters & Management
with st.sidebar:
    st.title("⚙️ Engine Configuration")
    
    st.markdown("### Rocchio Hyperparameters (Q2)")
    alpha = st.slider("Alpha (Original Query α)", min_value=0.0, max_value=2.0, value=1.0, step=0.05,
                      help="Weight assigned to original query vector.")
    beta = st.slider("Beta (Relevant Documents β)", min_value=0.0, max_value=2.0, value=0.75, step=0.05,
                     help="Weight assigned to centroid of marked relevant documents.")
    gamma = st.slider("Gamma (Irrelevant Documents γ)", min_value=0.0, max_value=2.0, value=0.25, step=0.05,
                      help="Weight penalty assigned to centroid of marked irrelevant documents.")
    
    st.markdown("---")
    st.markdown("### Document Corpus")
    st.info(f"Total Documents: **{len(st.session_state.corpus)}**")
    
    # Custom Document Ingestion
    with st.expander("➕ Add Custom Document"):
        new_id = st.text_input("Document ID", value=f"DOC{len(st.session_state.corpus)+1:02d}")
        new_title = st.text_input("Title", placeholder="e.g. Advancements in Fusion Energy")
        new_category = st.text_input("Category", value="General")
        new_content = st.text_area("Content", placeholder="Enter document body text...")
        if st.button("Add to Corpus"):
            if new_title and new_content:
                st.session_state.corpus.append({
                    "id": new_id,
                    "title": new_title,
                    "category": new_category,
                    "content": new_content
                })
                st.success(f"Added {new_id} to corpus!")
                st.rerun()

st.title("🔍 Smart Document Retrieval and Ranking System")
st.caption("Implementation for Assignment 2: TF-IDF, Cosine Similarity, Boolean Retrieval, and Rocchio Relevance Feedback")

# Tabs for Main Application
tab_search, tab_comparison, tab_corpus = st.tabs([
    "🎯 Q1 & Q2: Search & Relevance Feedback",
    "📊 Q2: Comparative Analysis & Metrics",
    "📚 Document Corpus Explorer"
])

# ----------------- Tab 1: Search & Relevance Feedback -----------------
with tab_search:
    st.markdown("### 1. Enter Search Query")
    query_col1, query_col2 = st.columns([3, 1])
    
    with query_col1:
        query = st.text_input(
            "Search Query",
            value="information retrieval search",
            placeholder="e.g., 'machine learning', 'astronomy OR space', 'climate AND carbon NOT politics'",
            help="Supports multi-keyword search as well as Boolean operators: AND, OR, NOT, and parentheses."
        )
    with query_col2:
        top_k = st.selectbox("Max Results", options=[5, 10, 15], index=0)

    is_bool = bool_parser.is_boolean_query(query)
    matching_indices = None

    if is_bool:
        matching_indices = list(bool_parser.evaluate(query))
        st.info(f"**Boolean Query Detected**: Filter matched **{len(matching_indices)}** documents before TF-IDF ranking.")
        if not matching_indices:
            st.warning("⚠️ No documents satisfy the Boolean condition.")

    if query.strip() and (not is_bool or matching_indices):
        q_vec = vectorizer.transform_query(query)
        initial_ranks = vectorizer.rank_documents(q_vec, filter_doc_indices=matching_indices)
        
        st.session_state["initial_ranks"] = initial_ranks
        st.session_state["q_vec"] = q_vec
        st.session_state["matching_indices"] = matching_indices

        st.markdown(f"### Q1: Ranked Retrieval Results (Top {min(top_k, len(initial_ranks))})")
        st.caption("Documents ranked in descending order of Cosine Similarity calculated using TF-IDF vectors.")

        # Interactive Feedback Form (Q2)
        with st.form("feedback_form"):
            st.markdown("#### Q2: Mark Relevance Feedback")
            st.caption("Select whether each document is Relevant, Irrelevant, or Neutral to refine the query.")

            feedback_selections = {}
            top_display = initial_ranks[:top_k]

            for rank, (doc_id, doc_idx, score) in enumerate(top_display, 1):
                doc = st.session_state.corpus[doc_idx]
                card_col, radio_col = st.columns([4, 1.5])
                with card_col:
                    st.markdown(f"**#{rank} [{doc['id']}] {doc['title']}** &nbsp; `Relevance Score: {score:.4f}`")
                    st.caption(f"*Category: {doc['category']}* — {doc['content'][:220]}...")
                with radio_col:
                    choice = st.radio(
                        f"Feedback for {doc_id}",
                        options=["Neutral", "Relevant 👍", "Irrelevant 👎"],
                        horizontal=True,
                        key=f"feedback_{doc_id}",
                        label_visibility="collapsed"
                    )
                    feedback_selections[doc_id] = choice
                st.divider()

            submit_feedback = st.form_submit_button("🚀 Submit Relevance Feedback & Re-Rank (Q2)")

        if submit_feedback:
            rel_ids = [did for did, c in feedback_selections.items() if c == "Relevant 👍"]
            non_rel_ids = [did for did, c in feedback_selections.items() if c == "Irrelevant 👎"]

            st.session_state["rel_ids"] = rel_ids
            st.session_state["non_rel_ids"] = non_rel_ids

            all_doc_ids = [d["id"] for d in st.session_state.corpus]
            rel_vecs = [vectorizer.doc_vectors[all_doc_ids.index(did)] for did in rel_ids if did in all_doc_ids]
            non_rel_vecs = [vectorizer.doc_vectors[all_doc_ids.index(did)] for did in non_rel_ids if did in all_doc_ids]

            rocchio = RocchioRelevanceFeedback(alpha=alpha, beta=beta, gamma=gamma)
            new_q_vec = rocchio.update_query_vector(q_vec, rel_vecs, non_rel_vecs)
            feedback_ranks = vectorizer.rank_documents(new_q_vec, filter_doc_indices=matching_indices)

            st.session_state["feedback_ranks"] = feedback_ranks
            st.session_state["new_q_vec"] = new_q_vec
            st.success("Relevance feedback applied successfully! Check the re-ranked results below and the Comparative Analysis tab.")

        # Show Re-ranked results if available
        if "feedback_ranks" in st.session_state:
            st.markdown("### 🔄 Re-Ranked Documents after Relevance Feedback")
            re_top = st.session_state["feedback_ranks"][:top_k]
            
            re_data = []
            for rank, (doc_id, doc_idx, score) in enumerate(re_top, 1):
                doc = st.session_state.corpus[doc_idx]
                tag = "Relevant" if doc_id in st.session_state.get("rel_ids", []) else ("Irrelevant" if doc_id in st.session_state.get("non_rel_ids", []) else "Unlabeled")
                re_data.append({
                    "Rank": rank,
                    "Doc ID": doc_id,
                    "Title": doc["title"],
                    "Category": doc["category"],
                    "Cosine Similarity": f"{score:.4f}",
                    "Feedback Tag": tag
                })
            st.dataframe(pd.DataFrame(re_data), use_container_width=True)

# ----------------- Tab 2: Comparative Analysis -----------------
with tab_comparison:
    st.markdown("### 📊 Before vs. After Feedback Comparative Analysis")
    st.caption("Demonstrating the retrieval quality improvement achieved by Rocchio Query Vector Adjustment.")

    if "initial_ranks" in st.session_state and "feedback_ranks" in st.session_state:
        rel_ids = set(st.session_state.get("rel_ids", []))
        non_rel_ids = set(st.session_state.get("non_rel_ids", []))

        eval_res = SearchEvaluator.compare_rankings(
            st.session_state["initial_ranks"],
            st.session_state["feedback_ranks"],
            rel_ids,
            top_k=top_k
        )

        # Metric KPI cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Precision Before", f"{eval_res['precision_before']:.2f}")
        delta_p = eval_res['precision_after'] - eval_res['precision_before']
        kpi2.metric("Precision After", f"{eval_res['precision_after']:.2f}", delta=f"{delta_p:+.2f}")
        
        kpi3.metric("MAP (AP) Before", f"{eval_res['ap_before']:.2f}")
        delta_ap = eval_res['ap_after'] - eval_res['ap_before']
        kpi4.metric("MAP (AP) After", f"{eval_res['ap_after']:.2f}", delta=f"{delta_ap:+.2f}")

        st.markdown("#### Side-by-Side Rank & Score Comparison")
        comp_df = pd.DataFrame(eval_res["comparison_table"])
        
        # Add titles for clarity
        id_to_title = {d["id"]: d["title"] for d in st.session_state.corpus}
        comp_df["title"] = comp_df["doc_id"].map(id_to_title)
        
        # Format tags
        def get_tag_label(row):
            if row["doc_id"] in rel_ids:
                return "Relevant 👍"
            elif row["doc_id"] in non_rel_ids:
                return "Irrelevant 👎"
            return "Neutral"
        
        comp_df["user_tag"] = comp_df.apply(get_tag_label, axis=1)

        display_cols = ["doc_id", "title", "user_tag", "rank_before", "rank_after", "rank_shift", "score_before", "score_after", "score_diff"]
        st.dataframe(comp_df[display_cols], use_container_width=True)

        # Matplotlib visualization of Score Shifts
        st.markdown("#### Relevance Score Shift Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        docs_plot = comp_df["doc_id"].tolist()
        x = np.arange(len(docs_plot))
        width = 0.35

        ax.bar(x - width/2, comp_df["score_before"], width, label='Score Before Feedback', color='#90caf9')
        ax.bar(x + width/2, comp_df["score_after"], width, label='Score After Feedback', color='#4caf50')

        ax.set_ylabel('Cosine Similarity Score')
        ax.set_title('Cosine Similarity Changes per Document After Rocchio Feedback')
        ax.set_xticks(x)
        ax.set_xticklabels(docs_plot, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.info("ℹ️ Perform a search and submit relevance feedback in the first tab to view the comparative analysis.")

# ----------------- Tab 3: Corpus Explorer -----------------
with tab_corpus:
    st.markdown("### 📚 Corpus Documents")
    st.caption("Explore the document collection indexed by the TF-IDF engine.")
    
    corpus_df = pd.DataFrame(st.session_state.corpus)
    st.dataframe(corpus_df[["id", "category", "title", "content"]], use_container_width=True)
