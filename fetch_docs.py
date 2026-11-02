#!/usr/bin/env python
"""Download the pinned FineWeb shard (about 2 GB) and extract the task documents into data/. Text stays on your machine."""
import argparse, glob, hashlib, json, os, shlex
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
pins = {k: v for k, v in (shlex.split(l)[0].split("=", 1) for l in open("pins.env") if "=" in l and not l.startswith("#"))}
a = argparse.ArgumentParser(); a.add_argument("--tasks", default="tasks/shared_300.jsonl"); x = a.parse_args()
tasks = [json.loads(l) for l in open(x.tasks)]; want = {t["id"]: t for t in tasks}
p = hf_hub_download(pins["FINEWEB_REPO"], pins["SHARD"], repo_type="dataset", revision=pins["FINEWEB_REV"])
os.makedirs("data", exist_ok=True); got = {}
for b in pq.ParquetFile(p).iter_batches(batch_size=4096, columns=["id", "text", "url", "token_count"]):
    for r in b.to_pylist():
        if r["id"] in want:
            if hashlib.sha256(r["text"].encode()).hexdigest() != want[r["id"]]["sha256"]:
                raise SystemExit(f"hash mismatch for {r['id']}")
            got[r["id"]] = r
if len(got) != len(want): raise SystemExit(f"found {len(got)} of {len(want)} documents")
out = "data/docs_" + os.path.basename(x.tasks)
with open(out, "w") as f:
    for t in tasks: f.write(json.dumps(got[t["id"]]) + "\n")
print(f"{len(got)} documents verified, saved to {out}")
