import os
import json
import time
from baseline_retriever import ds, build_ids, held_out_ids, BaselineRetriever, process_papers


def generate_baseline_jsonls():
    os.makedirs("results", exist_ok=True)

    # 1. Initialize and build baseline retriever on dev papers
    corpus_chunks = process_papers(ds, build_ids)
    retriever = BaselineRetriever(corpus_chunks)
    retriever.build_embeddings()

    # 2. Process Development Split (dev_baseline.jsonl)
    print("Generating dev_baseline.jsonl...")
    dev_papers = [p for p in ds if p["id"] in build_ids]

    with open("results/dev_baseline.jsonl", "w", encoding="utf-8") as f:
        for paper in dev_papers:
            paper_id = paper["id"]
            qas = paper["qas"]

            # Extract questions cleanly
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
                results, latency_ms = retriever.retrieve(query, top_k=10)

                record = {
                    "question_id": qid,
                    "target_paper_id": paper_id,
                    "retrieved": results,
                    "answer": None,
                    "abstained": None,
                    "latency_ms": round(latency_ms, 2),
                    "llm_calls": 0
                }
                f.write(json.dumps(record) + "\n")

    print("dev_baseline.jsonl successfully generated!")


if __name__ == "__main__":
    generate_baseline_jsonls()
