#!/usr/bin/env python
"""Queue the documents the student rates most likely to be spam or PII, for the teacher to label next."""
import glob, json, sys
import pyarrow as pa, pyarrow.parquet as pq
d, n = sys.argv[1], int(sys.argv[2])
cols = ["id", "text", "s_spam_or_boilerplate_probs", "s_pii_risk_probs"]; R = []
for p in sorted(glob.glob(f"{d}/shards/*.labelled.parquet")): R += pq.read_table(p, columns=cols).to_pylist()
yes = lambda r, k: json.loads(r[k])["yes"]; pick = {}
for k in ("s_spam_or_boilerplate_probs", "s_pii_risk_probs"):
    for r in sorted(R, key=lambda r: -yes(r, k))[:n]: pick[r["id"]] = r
print("pool", len(R), "queued", len(pick))
pq.write_table(pa.Table.from_pylist([{"id": r["id"], "text": r["text"]} for r in pick.values()]), f"{d}/priority.parquet", compression="zstd")
json.dump(sorted(pick), open(f"{d}/priority_ids.json", "w"))
