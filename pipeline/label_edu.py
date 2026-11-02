"""Score documents with the FineWeb-Edu classifier at a pinned revision."""
import json, sys, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForSequenceClassification as M
src, out, rev = sys.argv[1:4]; N = "HuggingFaceFW/fineweb-edu-classifier"
tb = pq.read_table(src, columns=["id", "text"]).to_pylist()
tok = AutoTokenizer.from_pretrained(N, revision=rev); m = M.from_pretrained(N, revision=rev, dtype=torch.bfloat16).cuda().eval()
tb.sort(key=lambda d: len(d["text"]))
with open(out, "w") as f:
    for b in range(0, len(tb), 64):
        c = tb[b:b+64]; e = tok([d["text"] for d in c], return_tensors="pt", padding="longest", truncation=True).to(0)
        with torch.no_grad(): s = m(**e).logits.squeeze(-1).float().cpu().tolist()
        for d, v in zip(c, s): f.write(json.dumps({"id": d["id"], "edu": v}) + "\n")
print("edu done", len(tb))
