import random
from baseline_retriever import ds, build_ids, BaselineRetriever, process_papers

# 1. Initialize retriever for dev set
corpus_chunks = process_papers(ds, build_ids)
retriever = BaselineRetriever(corpus_chunks)
retriever.build_embeddings()

# 2. Extract development questions
dev_questions = []
dev_papers = [paper for paper in ds if paper["id"] in build_ids]

for paper in dev_papers:
    paper_id = paper["id"]
    qas = paper["qas"]
    # Handle QASPER qas format variation (dict vs list)
    questions_list = qas if isinstance(qas, list) else [
        {
            "question_id": qas["question_id"][i],
            "question": qas["question"][i],
            "answers": qas["answers"][i]
        }
        for i in range(len(qas.get("question_id", [])))
    ] if "question_id" in qas else []

    # Alternatively parse standard QASPER structure
    if isinstance(qas, dict) and "question" in qas:
        # standard dict lists
        q_ids = qas.get("question_id", [])
        q_texts = qas.get("question", [])
        q_answers = qas.get("answers", [])
        for qid, qtext, qans in zip(q_ids, q_texts, q_answers):
            dev_questions.append({
                "question_id": qid,
                "target_paper_id": paper_id,
                "question": qtext,
                "answers": qans
            })
    elif isinstance(qas, list):
        for q in qas:
            dev_questions.append({
                "question_id": q["question_id"],
                "target_paper_id": paper_id,
                "question": q["question"],
                "answers": q.get("answers", [])
            })

print(
    f"Extracted {len(dev_questions)} development questions to analyze for failures.")

# 3. Find retrieval failures (where gold paragraphs don't align well or rank low)
failure_candidates = []
for item in dev_questions[:50]:  # Sample first 50 to find misses
    query = item["question"]
    target_id = item["target_paper_id"]

    results, _ = retriever.retrieve(query, top_k=10)

    # Check if target paper chunks are in top 10
    top_papers = [r["paper_id"] for r in results]
    found_target = target_id in top_papers

    if not found_target or top_papers[0] != target_id:
        failure_candidates.append({
            "question_id": item["question_id"],
            "target_paper_id": target_id,
            "question": query,
            "top_retrieved_papers": top_papers[:3]
        })
        if len(failure_candidates) >= 10:
            break

print(
    f"\nIdentified {len(failure_candidates)} failure cases for inspection table:")
for i, f in enumerate(failure_candidates, 1):
    print(
        f"{i}. QID: {f['question_id']} | Target Paper: {f['target_paper_id']}")
    print(f"   Query: {f['question']}")
    print(f"   Top Retrieved Papers: {f['top_retrieved_papers']}\n")
