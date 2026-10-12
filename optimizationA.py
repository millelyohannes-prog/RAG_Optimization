import os
import json
import time
from baseline_retriever import ds, build_ids, BaselineRetriever


def process_papers_structure_aware(dataset, target_paper_ids):
    """Extracts chunks using natural paragraph structures and exact character offsets."""
    corpus_chunks = []
    subset = [paper for paper in dataset if paper["id"] in target_paper_ids]

    print(
        f"Processing structure-aware chunks for {len(subset)} development papers...")
    for paper in subset:
        paper_id = paper["id"]

        # 1. Handle Abstract if present
        if "abstract" in paper and paper["abstract"]:
            abs_text = paper["abstract"]
            corpus_chunks.append({
                "chunk_id": f"{paper_id}:abstract",
                "paper_id": paper_id,
                "text": abs_text,
                "source_spans": [{"paragraph_id": "abstract", "start": 0, "end": len(abs_text)}]
            })

        # 2. Handle Full Text sections and paragraphs
        if "full_text" in paper and paper["full_text"]:
            sections = paper["full_text"]
            for s_idx, section in enumerate(sections):
                if isinstance(section, dict):
                    paragraphs = section.get("paragraphs", [])
                    for p_idx, para in enumerate(paragraphs):
                        if isinstance(para, str) and len(para.strip()) > 0:
                            para_id = f"s{s_idx}:p{p_idx}"
                            corpus_chunks.append({
                                "chunk_id": f"{paper_id}:{para_id}",
                                "paper_id": paper_id,
                                "text": para,
                                "source_spans": [{"paragraph_id": para_id, "start": 0, "end": len(para)}]
                            })
                elif isinstance(section, str) and len(section.strip()) > 0:
                    para_id = f"s{s_idx}"
                    corpus_chunks.append({
                        "chunk_id": f"{paper_id}:{para_id}",
                        "paper_id": paper_id,
                        "text": section,
                        "source_spans": [{"paragraph_id": para_id, "start": 0, "end": len(section)}]
                    })

    print(f"Generated {len(corpus_chunks)} structure-aware paragraph chunks.")
    return corpus_chunks


def generate_opt_a_jsonl():
    os.makedirs("results", exist_ok=True)

    # Use structure-aware chunks instead of baseline chunks
    corpus_chunks = process_papers_structure_aware(ds, build_ids)

    # Initialize retriever with a distinct cache file for embeddings
    retriever = BaselineRetriever(corpus_chunks)
    retriever.build_embeddings(cache_path="embeddings_cache_opt_a.pkl")

    print("Generating dev_A.jsonl...")
    dev_papers = [p for p in ds if p["id"] in build_ids]

    with open("results/dev_A.jsonl", "w", encoding="utf-8") as f:
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

    print("dev_A.jsonl successfully generated!")


if __name__ == "__main__":
    generate_opt_a_jsonl()
