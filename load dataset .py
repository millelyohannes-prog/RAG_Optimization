import random
from datasets import load_dataset
url = ("hf://datasets/allenai/qasper@0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46/"
       "qasper/validation-00000-of-00001.parquet")
ds = load_dataset("parquet", data_files={
                  "validation": url}, split="validation")
paper_ids = sorted(set(ds["id"]))
random.Random(42).shuffle(paper_ids)
build_ids, held_out_ids = paper_ids[:80], paper_ids[80:130]

held = set(held_out_ids)
qids = sorted({str(qid) for paper in ds if paper["id"] in held
               for qid in (paper["qas"]["question_id"] if isinstance(paper["qas"], dict)
                           else [q["question_id"] for q in paper["qas"]])})


random.Random(42).shuffle(qids)
ANSWER_QIDS = set(qids[:20])
