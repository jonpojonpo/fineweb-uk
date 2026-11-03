#!/usr/bin/env python
"""Score a parquet of documents with Croydon. Writes id, croydon_score (0-5, probability-weighted) and croydon_probs."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow as pa, pyarrow.parquet as pq, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from train_croydon import Croydon, predict, score
src, mdir, out = sys.argv[1:4]; cfg = json.load(open(f"{mdir}/croydon.json"))
m = Croydon(cfg["base"]); m.load_state_dict(load_file(f"{mdir}/model.safetensors")); dev = "cuda" if torch.cuda.is_available() else "cpu"; m.to(dev)
tok = AutoTokenizer.from_pretrained(mdir); t = pq.read_table(src, columns=["id", "text"]).to_pylist()
for bs in (64, 16, 4, 0):  # the GPU may be shared; fall back to smaller batches, then the CPU
    try:
        if bs == 0: dev = "cpu"; m.to("cpu"); bs = 32
        P = predict(m, tok, [r["text"] for r in t], dev, bs=bs); break
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        torch.cuda.empty_cache()
S = score(P).tolist()
pq.write_table(pa.table({"id": [r["id"] for r in t], "croydon_score": [round(s, 4) for s in S], "croydon_probs": [json.dumps([round(v, 4) for v in p]) for p in P.tolist()]}), out + ".part", compression="zstd")
os.replace(out + ".part", out); print("wrote", len(t), out, flush=True)
