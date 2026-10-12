import os
import pickle
import random
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 1. Load Dataset & Create Reproducible Splits
url = ("hf://datasets/allenai/qasper@0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46/"
       "qasper/validation-00000-of-00001.parquet")
ds = load_dataset("parquet", data_files={
                  "validation": url}, split="validation")

paper_ids = sorted(set(ds["id"]))
random.Random(42).shuffle(paper_ids)
build_ids, held_out_ids = paper_ids[:80], paper_ids[80:130]

print(
    f"Loaded {len(build_ids)} dev papers and {len(held_out_ids)} held-out papers.")

# 2. Baseline Chunking: Overlapping fixed-length character chunks


def chunk_text_baseline(text, chunk_size=500, overlap=100):
    """Chunks text into fixed-length character windows with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"text": text[start:end], "start": start, "end": end})
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def process_papers(dataset, target_paper_ids):
    """Extracts and chunks papers belonging to the specified split safely."""
    corpus_chunks = []
    subset = [paper for paper in dataset if paper["id"] in target_paper_ids]

    print(f"Processing and chunking {len(subset)} development papers...")
    for paper in subset:
        paper_id = paper["id"]
        full_text_parts = []
        if "abstract" in paper and paper["abstract"]:
            full_text_parts.append(("abstract", paper["abstract"]))

        if "full_text" in paper and paper["full_text"]:
            sections = paper["full_text"]
            for s_idx, section in enumerate(sections):
                if isinstance(section, dict):
                    paragraphs = section.get("paragraphs", [])
                    for p_idx, para in enumerate(paragraphs):
                        if isinstance(para, str):
                            full_text_parts.append(
                                (f"s{s_idx}:p{p_idx}", para))
                elif isinstance(section, str):
                    full_text_parts.append((f"s{s_idx}", section))

        combined_text = "\n".join([text for _, text in full_text_parts])

        raw_chunks = chunk_text_baseline(combined_text)
        for i, c in enumerate(raw_chunks):
            corpus_chunks.append({
                "chunk_id": f"{paper_id}:c{i}",
                "paper_id": paper_id,
                "text": c["text"],
                "source_spans": [{"paragraph_id": "full_doc", "start": c["start"], "end": c["end"]}]
            })

    print(
        f"Generated {len(corpus_chunks)} total chunks for the development set.")
    return corpus_chunks

# 3. Embedding and Retrieval Setup


class BaselineRetriever:
    def __init__(self, corpus_chunks, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.corpus_chunks = corpus_chunks
        self.chunk_embeddings = None
        self.query_cache = {}

    def build_embeddings(self, cache_path="embeddings_cache.pkl"):
        if os.path.exists(cache_path):
            print("Loading cached chunk embeddings...")
            with open(cache_path, "rb") as f:
                self.chunk_embeddings = pickle.load(f)
        else:
            print("Computing chunk embeddings...")
            texts = [chunk["text"] for chunk in self.corpus_chunks]
            # BGE models benefit from instruction prefix for retrieval, but standard embedding works for baseline
            self.chunk_embeddings = self.model.encode(
                texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.chunk_embeddings, f)

    def retrieve(self, query, top_k=10):
        t0 = time.time()
        if query in self.query_cache:
            q_emb = self.query_cache[query]
        else:
            # For bge models, query encoding
            q_emb = self.model.encode([query], normalize_embeddings=True)[0]
            self.query_cache[query] = q_emb

        # Compute cosine similarity (since embeddings are normalized, dot product = cosine similarity)
        scores = np.dot(self.chunk_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.corpus_chunks[idx]
            retrieved_results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "score": float(scores[idx]),
                "text": chunk["text"],
                "source_spans": chunk["source_spans"]
            })

        latency_ms = (time.time() - t0) * 1000
        return retrieved_results, latency_ms
