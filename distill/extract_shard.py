#!/usr/bin/env python
"""Fetch one FineWeb-Edu shard (a path, or a 10BT index), keep its .uk documents with provenance, and delete the raw download."""
import hashlib, json, os, shlex, shutil, sys, tempfile
from urllib.parse import urlparse
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
pins = dict(shlex.split(l)[0].split("=", 1) for l in open("pins.env") if "=" in l and not l.startswith("#"))
out = sys.argv[2]; shard = sys.argv[1] if "/" in sys.argv[1] else f"sample/10BT/{int(sys.argv[1]):03d}_00000.parquet"
tmp = tempfile.mkdtemp(dir=os.path.dirname(out))
try:
    p = hf_hub_download(pins["EDU_REPO"], shard, repo_type="dataset", revision=pins["EDU_DATA_REV"], local_dir=tmp)
    w = None; n = k = t = 0
    for b in pq.ParquetFile(p).iter_batches(batch_size=2000):
        rows = []
        for r in b.to_pylist():
            n += 1
            try: h = (urlparse(r["url"]).hostname or "").lower()
            except ValueError: continue
            if h.endswith(".uk"):
                r.update(host=h, source=f"{pins['EDU_REPO']}:{shard}", sha256=hashlib.sha256(r["text"].encode()).hexdigest()); rows.append(r); k += 1; t += r["token_count"]
        if rows:
            tb = pa.Table.from_pylist(rows); w = w or pq.ParquetWriter(out + ".part", tb.schema, compression="zstd"); w.write_table(tb.cast(w.schema))
    w.close(); os.replace(out + ".part", out); print(json.dumps({"shard": shard, "scanned": n, "uk": k, "tokens": t}), flush=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
