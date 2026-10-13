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
```
## Project Reflection
### A. Key Improvements
  1. Structure-Aware Chunking: Replacing fixed character boundaries with section-aware document parsing prevented sentences and scientific tables from being severed mid-context.
  2. Hybrid Search Integration: Combining BM25 sparse keyword matching with dense neural embeddings resolved exact-entity retrieval failures (e.g., parameter numbers, model names, and specialized acronyms).
  3. Score Normalization & Calibration: Min-Max scaling prevented high-magnitude BM25 scores from completely dominating dense vector similarity during rank fusion.
### B. Failed Experiments & Iterations
  1. Pure Character Splitting: Initial trials with naive 500-character chunking led to split contexts and fragmented answers on multi-sentence questions.
  2. Unweighted Score Addition: Directly adding raw BM25 scores to Cosine Similarity vectors resulted in poor precision because BM25 values ($0\text{--}30+$ range) overrode dense embedding scores ($0\text{--}1$ range).
  3. Excessive Chunk Overlap: Setting overlapping windows above $30\%$ significantly increased index size and latency without providing measurable recall gains.
### C. Quality vs. Cost & Latency Trade-offs
  1. Embedding Compute: Generating dense embeddings for every scientific paper chunk requires upfront GPU processing and disk space (cached vector pickles), but reduces query latency to sub-second lookup times.
  2. Hybrid Retrieval Overhead: Running two retrieval paths (BM25 index lookup + Vector similarity) adds roughly $15\text{--}25\text{ ms}$ per query compared to pure vector search, but yields a substantial gain in answer precision.
  3. Memory Footprint vs. Accuracy: Storing full document metadata alongside chunks increases memory usage, but provides the LLM with critical context headers required for accurate answer synthesis.

## 10 failure cases for inspection table:
1. QID: 7e62a53823aba08bc26b2812db016f5ce6159565 | Target Paper: 1811.00383
   Query: Which dataset(s) do they experiment with?
   Top Retrieved Papers: ['2001.09215', '1912.03627', '1910.12795']

2. QID: 455d4ef8611f62b1361be4f6387b222858bb5e56 | Target Paper: 1603.09631
   Query: How was this data collected?
   Top Retrieved Papers: ['1704.04452', '2001.09215', '1809.06537']

3. QID: 3d7d865e905295d11f1e85af5fa89b210e3e9fdf | Target Paper: 1911.06964
   Query: How many participants were trying this communication game?
   Top Retrieved Papers: ['1911.01371', '2001.08868', '2001.08868']

4. QID: 2ad4d3d222f5237ed97923640bc8e199409cbe52 | Target Paper: 1911.06964
   Query: What user variations have been tested?
   Top Retrieved Papers: ['1909.10012', '1911.00841', '1912.03627']

5. QID: 3fad42be0fb2052bb404b989cc7d58b440cd23a0 | Target Paper: 1911.06964
   Query: What are the baselines used?
   Top Retrieved Papers: ['1804.11346', '2004.04721', '1804.11346']

6. QID: ee417fea65f9b1029455797671da0840c8c1abbe | Target Paper: 2001.02284
   Query: Do they use off-the-shelf NLP systems to build their assitant?
   Top Retrieved Papers: ['1706.00139', '1911.01371', '2001.02284']

7. QID: 6424e442b34a576f904d9649d63acf1e4fdefdfc | Target Paper: 1808.09111
   Query: What datasets do they evaluate on?
   Top Retrieved Papers: ['2001.09215', '1912.08960', '1912.03627']

8. QID: 5eabfc6cc8aa8a99e6e42514ef9584569cb75dec | Target Paper: 1808.09111
   Query: Do they evaluate only on English datasets?
   Top Retrieved Papers: ['1912.03627', '2001.09215', '1703.07090']

9. QID: b14217978ad9c3c9b6b1ce393b1b5c6e7f49ecab | Target Paper: 1906.08593
   Query: On which tasks do they test their conflict method?
   Top Retrieved Papers: ['2001.07820', '1911.02855', '1906.08593']

10. QID: a99fdd34422f4231442c220c97eafc26c76508dd | Target Paper: 1809.00540
   Query: Do they use graphical models?
   Top Retrieved Papers: ['1806.04330', '1912.08960', '1809.06537']
