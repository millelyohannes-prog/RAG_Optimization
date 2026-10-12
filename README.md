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
