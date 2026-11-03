"""Filter one FineWeb-Edu shard to .uk hosts, keeping provenance. FineWeb-Edu's own score is the quality label."""
import hashlib, json, shlex, sys
from urllib.parse import urlparse
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
pins = dict(shlex.split(l)[0].split("=", 1) for l in open("pins.env") if "=" in l and not l.startswith("#"))
out = sys.argv[1]; p = hf_hub_download(pins["EDU_REPO"], pins["EDU_SHARD"], repo_type="dataset", revision=pins["EDU_DATA_REV"])
w = None; n = k = t = 0
for b in pq.ParquetFile(p).iter_batches(batch_size=2000):
    rows = []
    for r in b.to_pylist():
        n += 1
        try: h = (urlparse(r["url"]).hostname or "").lower()
        except ValueError: continue
        if h.endswith(".uk"):
            r.update(host=h, source=f"{pins['EDU_REPO']}:{pins['EDU_SHARD']}", sha256=hashlib.sha256(r["text"].encode()).hexdigest())
            rows.append(r); k += 1; t += r["token_count"]
    if rows:
        tb = pa.Table.from_pylist(rows); w = w or pq.ParquetWriter(out, tb.schema, compression="zstd"); w.write_table(tb.cast(w.schema))
w.close(); print(json.dumps({"scanned": n, "uk": k, "rate": round(k / n, 4), "tokens": t}))
