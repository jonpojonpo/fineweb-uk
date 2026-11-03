#!/usr/bin/env python
"""Label a parquet of documents with a trained student and write a joined parquet with provenance."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow as pa, pyarrow.parquet as pq, torch
from transformers import AutoTokenizer
from train_student import Student, predict, HEADS
src, sdir, out = sys.argv[1:4]
ck = torch.load(f"{sdir}/student.pt", map_location="cpu"); m = Student(ck["base"]); m.load_state_dict(ck["state"]); m.cuda()
tok = AutoTokenizer.from_pretrained(sdir); rows = pq.read_table(src).to_pylist()
P = predict(m, tok, [r["text"] for r in rows], "cuda")
for i, r in enumerate(rows):
    for k, opts in HEADS.items():
        p = P[k][i].tolist(); r[f"s_{k}"] = opts[max(range(len(p)), key=p.__getitem__)]; r[f"s_{k}_probs"] = json.dumps(dict(zip(opts, [round(v, 4) for v in p])))
    r["labeller"] = "ModernBERT student v0, distilled from " + os.environ.get("TEACHER_NAME", "unknown")
pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd"); print("wrote", len(rows), out)
