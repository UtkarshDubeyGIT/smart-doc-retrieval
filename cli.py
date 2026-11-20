"""
Smart Document Retrieval & Relevance Feedback Engine
Interactive CLI for Question 1 & Question 2.
"""
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from core.tfidf import TFIDFVectorizer
from core.boolean_parser import BooleanQueryParser
from core.rocchio import RocchioRelevanceFeedback
from core.evaluator import SearchEvaluator
from data.corpus import SAMPLE_DOCUMENTS

console = Console()

def run_cli():
    console.print(Panel.fit(
        "[bold cyan]Smart Document Retrieval and Ranking System[/bold cyan]\n"
        "[dim]TF-IDF Vector Space Model • Boolean Query Parsing • Rocchio Relevance Feedback[/dim]",
        border_style="cyan"
    ))

    # Initialize Engine
    corpus = SAMPLE_DOCUMENTS
    doc_ids = [doc["id"] for doc in corpus]
    doc_texts = [f"{doc['title']}. {doc['content']}" for doc in corpus]

    vectorizer = TFIDFVectorizer()
    vectorizer.fit_transform(doc_texts, doc_ids)
    bool_parser = BooleanQueryParser(doc_texts)
    rocchio = RocchioRelevanceFeedback(alpha=1.0, beta=0.75, gamma=0.25)

    console.print(f"[green]✓[/green] Loaded corpus with [bold]{len(corpus)}[/bold] documents across {len(set(d['category'] for d in corpus))} domains.\n")

    while True:
        console.print("[bold yellow]Enter your search query[/bold yellow] (or type [red]'exit'[/red] to quit):")
        console.print("[dim]Supports: Plain text ('deep learning') or Boolean operators ('machine AND learning', 'mars OR space NOT climate')[/dim]")
        query = Prompt.ask("Query")
        if query.strip().lower() in ("exit", "quit", "q"):
            break

        if not query.strip():
            continue

        # Evaluate Boolean filter if present
        is_bool = bool_parser.is_boolean_query(query)
        matching_indices = None
        if is_bool:
            matching_indices = list(bool_parser.evaluate(query))
            console.print(f"[blue]ℹ Boolean query detected.[/blue] Matching candidates: {len(matching_indices)} documents.")
            if not matching_indices:
                console.print("[red]No documents satisfied the Boolean condition.[/red]\n")
                continue

        # Transform query and rank
        q_vec = vectorizer.transform_query(query)
        initial_ranks = vectorizer.rank_documents(q_vec, filter_doc_indices=matching_indices)

        # Display Top Results (Q1)
        table = Table(title=f"Initial Retrieval Results (Q1) - Top {min(5, len(initial_ranks))}")
        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("Doc ID", style="magenta")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Category", style="yellow")
        table.add_column("Title", style="white")

        top_5 = initial_ranks[:5]
        for idx, (doc_id, doc_idx, score) in enumerate(top_5, 1):
            doc = corpus[doc_idx]
            table.add_row(str(idx), doc_id, f"{score:.4f}", doc["category"], doc["title"])
        console.print(table)

        # Prompt for Relevance Feedback (Q2)
        if Confirm.ask("\n[bold]Would you like to provide Relevance Feedback (Q2) to improve rankings?[/bold]"):
            console.print("\n[dim]Select document IDs from top results. Multiple IDs separated by space (e.g. DOC01 DOC02)[/dim]")
            rel_input = Prompt.ask("Enter Relevant document IDs (leave blank for none)", default="")
            non_rel_input = Prompt.ask("Enter Irrelevant document IDs (leave blank for none)", default="")

            rel_ids = set(rel_input.strip().split())
            non_rel_ids = set(non_rel_input.strip().split())

            rel_vecs = [vectorizer.doc_vectors[doc_ids.index(did)] for did in rel_ids if did in doc_ids]
            non_rel_vecs = [vectorizer.doc_vectors[doc_ids.index(did)] for did in non_rel_ids if did in doc_ids]

            if not rel_vecs and not non_rel_vecs:
                console.print("[yellow]No valid feedback provided. Keeping current ranking.[/yellow]\n")
                continue

            # Update Query using Rocchio
            new_q_vec = rocchio.update_query_vector(q_vec, rel_vecs, non_rel_vecs)
            feedback_ranks = vectorizer.rank_documents(new_q_vec, filter_doc_indices=matching_indices)

            # Comparative Analysis
            eval_res = SearchEvaluator.compare_rankings(initial_ranks, feedback_ranks, rel_ids, top_k=5)

            comp_table = Table(title="Comparative Ranking Analysis (Before vs After Feedback)")
            comp_table.add_column("Doc ID", style="magenta")
            comp_table.add_column("User Tag", justify="center")
            comp_table.add_column("Rank (Before)", justify="center")
            comp_table.add_column("Score (Before)", justify="right")
            comp_table.add_column("Rank (After)", justify="center")
            comp_table.add_column("Score (After)", justify="right")
            comp_table.add_column("Rank Shift", justify="center")

            for row in eval_res["comparison_table"]:
                tag = "[green]Relevant[/green]" if row["is_relevant"] else ("[red]Irrelevant[/red]" if row["doc_id"] in non_rel_ids else "[dim]Unlabeled[/dim]")
                shift_str = "-"
                if row["rank_shift"] is not None:
                    if row["rank_shift"] > 0:
                        shift_str = f"[bold green]↑ +{row['rank_shift']}[/bold green]"
                    elif row["rank_shift"] < 0:
                        shift_str = f"[bold red]↓ {row['rank_shift']}[/bold red]"
                    else:
                        shift_str = "[dim]0[/dim]"

                comp_table.add_row(
                    row["doc_id"],
                    tag,
                    str(row["rank_before"]) if row["rank_before"] else "N/A",
                    f"{row['score_before']:.4f}",
                    str(row["rank_after"]) if row["rank_after"] else "N/A",
                    f"{row['score_after']:.4f}",
                    shift_str
                )
            console.print(comp_table)

            console.print(Panel(
                f"[bold cyan]Performance Summary:[/bold cyan]\n"
                f"• Precision@5: [yellow]{eval_res['precision_before']:.2f}[/yellow] → [bold green]{eval_res['precision_after']:.2f}[/bold green]\n"
                f"• Average Precision (AP@5): [yellow]{eval_res['ap_before']:.2f}[/yellow] → [bold green]{eval_res['ap_after']:.2f}[/bold green]\n"
                f"[dim]The Rocchio feedback adjusted term weights towards relevant documents while penalizing irrelevant features.[/dim]",
                title="Metrics",
                border_style="green"
            ))

        console.print("\n" + "—" * 60 + "\n")

if __name__ == "__main__":
    run_cli()
