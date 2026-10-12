import os
import json
import time
import math
import numpy as np
from collections import Counter
from baseline_retriever import ds, build_ids, BaselineRetriever, process_papers


class PurePythonBM25:
    """Custom pure-Python BM25 implementation from scratch as required by the brief."""

    def __init__(self, corpus_texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        self.doc_lens = [len(self._tokenize(text)) for text in corpus_texts]
        self.avg_doc_len = sum(self.doc_lens) / \
            self.corpus_size if self.corpus_size > 0 else 0

        # Calculate document frequencies and inverted index
        self.df = Counter()
        self.doc_freqs = []
        for text in corpus_texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1
            self.doc_freqs.append(Counter(tokens))

        # Calculate IDFs
        self.idf = {}
        for token, freq in self.df.items():
            # Standard BM25 IDF formula
            self.idf[token] = math.log(
                1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def _tokenize(self, text):
        return text.lower().split()

    def score(self, query, doc_idx):
        query_tokens = self._tokenize(query)
        doc_len = self.doc_lens[doc_idx]
        doc_freq = self.doc_freqs[doc_idx]
        score = 0.0

        for token in query_tokens:
            if token not in self.idf:
                continue
            freq = doc_freq[token]
            numerator = self.idf[token] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * \
                (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += numerator / denominator
        return score

    def get_scores(self, query):
        return [self.score(query, idx) for idx in range(self.corpus_size)]


class HybridRetriever(BaselineRetriever):
    def __init__(self, corpus_chunks):
        super().__init__(corpus_chunks)
        corpus_texts = [c["text"] for c in self.corpus_chunks]
        print("Initializing custom pure-Python BM25 index...")
        self.bm25 = PurePythonBM25(corpus_texts)

    def retrieve_hybrid(self, query, top_k=10, alpha=0.6):
        t0 = time.time()
        # 1. Dense scores
        if query in self.query_cache:
            q_emb = self.query_cache[query]
        else:
            q_emb = self.model.encode([query], normalize_embeddings=True)[0]
            self.query_cache[query] = q_emb
        dense_scores = np.dot(self.chunk_embeddings, q_emb)

        # Normalize dense scores to [0, 1]
        d_min, d_max = np.min(dense_scores), np.max(dense_scores)
        if d_max > d_min:
            dense_norm = (dense_scores - d_min) / (d_max - d_min)
        else:
            dense_norm = np.zeros_like(dense_scores)

        # 2. Sparse BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query))
        b_min, b_max = np.min(bm25_scores), np.max(bm25_scores)
        if b_max > b_min:
            bm25_norm = (bm25_scores - b_min) / (b_max - b_min)
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        # 3. Hybrid Fusion score (Alpha blending)
        hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        retrieved_results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.corpus_chunks[idx]
            retrieved_results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "score": float(hybrid_scores[idx]),
                "text": chunk["text"],
                "source_spans": chunk["source_spans"]
            })

        latency_ms = (time.time() - t0) * 1000
        return retrieved_results, latency_ms


def generate_opt_c_jsonl():
    os.makedirs("results", exist_ok=True)
    corpus_chunks = process_papers(ds, build_ids)

    retriever = HybridRetriever(corpus_chunks)
    retriever.build_embeddings(cache_path="embeddings_cache_baseline.pkl")

    print("Generating dev_C.jsonl with Hybrid Dense + BM25 Fusion...")
    dev_papers = [p for p in ds if p["id"] in build_ids]

    with open("results/dev_C.jsonl", "w", encoding="utf-8") as f:
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

                results, latency_ms = retriever.retrieve_hybrid(
                    query, top_k=10)

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

    print("dev_C.jsonl successfully generated!")


if __name__ == "__main__":
    generate_opt_c_jsonl()
