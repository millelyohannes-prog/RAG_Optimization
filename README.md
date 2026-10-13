# QASPER RAG & Retrieval Pipeline

This repository contains a complete end-to-end Retrieval-Augmented Generation (RAG) and retrieval optimization pipeline built for the **QASPER** dataset task. The project implements a robust baseline, three distinct optimization strategies (Structure-Aware Chunking, Query Processing, and Hybrid BM25 + Dense Fusion), and a final combined system evaluated across development and held-out splits.

---

## 📁 Repository Structure

```text
├── results/
│   ├── dev_baseline.jsonl       # Baseline retrieval on dev split
│   ├── dev_A.jsonl              # Optimization A (Structure-aware chunking)
│   ├── dev_B.jsonl              # Optimization B (Query processing / rewriting)
│   ├── dev_C.jsonl              # Optimization C (Custom BM25 + Dense Hybrid Fusion)
│   ├── dev_final.jsonl          # Final combined system on dev split
│   ├── heldout_baseline.jsonl   # Baseline retrieval on held-out split
│   └── heldout_final.jsonl      # Final combined system on held-out split (+ LLM generation)
├── baseline_retriever.py        # Core dense baseline pipeline and embeddings
├── optimizationA.py             # Structure-aware chunking strategy
├── optimizationB.py             # Query expansion/processing optimization
├── optimizationC.py             # Pure-Python BM25 and Hybrid Fusion implementation
├── heldout_baseline.py          # Held-out baseline execution script
├── run_final_and_heldout.py     # Final system pipeline execution script
└── README.md

Project Reflection
A. Key Improvements
  1. Structure-Aware Chunking: Replacing fixed character boundaries with section-aware document parsing prevented sentences and scientific tables from being severed mid-context.
  2. Hybrid Search Integration: Combining BM25 sparse keyword matching with dense neural embeddings resolved exact-entity retrieval failures (e.g., parameter numbers, model names, and specialized acronyms).
  3. Score Normalization & Calibration: Min-Max scaling prevented high-magnitude BM25 scores from completely dominating dense vector similarity during rank fusion.
B. Failed Experiments & Iterations
  1. Pure Character Splitting: Initial trials with naive 500-character chunking led to split contexts and fragmented answers on multi-sentence questions.
  2. Unweighted Score Addition: Directly adding raw BM25 scores to Cosine Similarity vectors resulted in poor precision because BM25 values ($0\text{--}30+$ range) overrode dense embedding scores ($0\text{--}1$ range).
  3. Excessive Chunk Overlap: Setting overlapping windows above $30\%$ significantly increased index size and latency without providing measurable recall gains.
C. Quality vs. Cost & Latency Trade-offs
  1. Embedding Compute: Generating dense embeddings for every scientific paper chunk requires upfront GPU processing and disk space (cached vector pickles), but reduces query latency to sub-second lookup times.
  2. Hybrid Retrieval Overhead: Running two retrieval paths (BM25 index lookup + Vector similarity) adds roughly $15\text{--}25\text{ ms}$ per query compared to pure vector search, but yields a substantial gain in answer precision.
  3. Memory Footprint vs. Accuracy: Storing full document metadata alongside chunks increases memory usage, but provides the LLM with critical context headers required for accurate answer synthesis.

<img width="1488" height="842" alt="image" src="https://github.com/user-attachments/assets/8c30d130-3adf-4b6f-a6c4-ba07976120ff" />
