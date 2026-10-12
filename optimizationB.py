import os
import json
import time
from google import genai
from baseline_retriever import ds, build_ids, BaselineRetriever, process_papers

# Initialize the Gemini client (using gemini-3.6 as requested)
client = genai.Client()


def rewrite_query(query: str) -> tuple[str, int]:
    """Uses gemini-3.6 to rewrite and expand a query for better scientific retrieval."""
    prompt = (
        "You are an expert academic research assistant. Rewrite the following user question "
        "into a concise, keyword-rich search query optimized for dense text retrieval over computer science research papers. "
        "Include relevant technical synonyms or terms if helpful. Output ONLY the rewritten query text.\n\n"
        f"User Question: {query}"
    )
    try:
        response = client.models.generate_content(
            # mapping to current gemini flash model/api as specified for generation
            model='gemini-3.6-flash-lite',
            contents=prompt,
            config={"temperature": 0}
        )
        rewritten = response.text.strip()
        return rewritten, 1
    except Exception as e:
        print(f"Query rewriting failed ({e}), falling back to original query.")
        return query, 0


def generate_opt_b_jsonl():
    os.makedirs("results", exist_ok=True)

    # Use standard baseline chunks for this optimization
    corpus_chunks = process_papers(ds, build_ids)
    retriever = BaselineRetriever(corpus_chunks)
    retriever.build_embeddings(cache_path="embeddings_cache_baseline.pkl")

    print("Generating dev_B.jsonl with Query Rewriting (gemini-3.6)...")
    dev_papers = [p for p in ds if p["id"] in build_ids]

    with open("results/dev_B.jsonl", "w", encoding="utf-8") as f:
        for paper in dev_papers:
            paper_id = paper["id"]
            qas = paper["qas"]

            questions_list = []
            if isinstance(qas, dict):
                q_ids = qas.get("question_id", [])
                q_texts = qas.get("question", [])
                for qid, qtext in zip(q_ids, q_texts):
                    questions_list.append(
                        {"question_id": qid, "question": qtext})
            elif isinstance(qas, list):
                for q in qas:
                    questions_list.append(
                        {"question_id": q["question_id"], "question": q["question"]})

            for q_item in questions_list:
                qid = q_item["question_id"]
                query = q_item["question"]

                t0 = time.time()
                # Step 1: Rewrite query using Gemini
                optimized_query, llm_calls = rewrite_query(query)

                # Step 2: Retrieve using rewritten query
                results, retrieval_latency = retriever.retrieve(
                    optimized_query, top_k=10)
                total_latency_ms = (time.time() - t0) * 1000

                record = {
                    "question_id": qid,
                    "target_paper_id": paper_id,
                    "retrieved": results,
                    "answer": None,
                    "abstained": None,
                    "latency_ms": round(total_latency_ms, 2),
                    "llm_calls": llm_calls
                }
                f.write(json.dumps(record) + "\n")

    print("dev_B.jsonl successfully generated!")


if __name__ == "__main__":
    generate_opt_b_jsonl()
