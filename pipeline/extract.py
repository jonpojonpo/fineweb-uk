"""Filter one FineWeb parquet shard to .uk hosts and add a provenance record to each document."""
import hashlib, json, sys
from urllib.parse import urlparse
import pyarrow as pa, pyarrow.parquet as pq
src, out, shard = sys.argv[1:4]
pf = pq.ParquetFile(src); w = None; n = k = 0
for b in pf.iter_batches(batch_size=2000):
    rows = []
    for r in b.to_pylist():
        n += 1
        try: h = (urlparse(r["url"]).hostname or "").lower()
        except ValueError: continue
        if not h.endswith(".uk"): continue
        r.update(host=h, source=f"HuggingFaceFW/fineweb:{shard}", licence="ODC-By 1.0 (FineWeb compilation); Common Crawl ToU; underlying page rights not cleared",
                 sha256=hashlib.sha256(r["text"].encode()).hexdigest())
        rows.append(r); k += 1
    if rows:
        t = pa.Table.from_pylist(rows)
        w = w or pq.ParquetWriter(out, t.schema, compression="zstd")
        w.write_table(t.cast(w.schema))
w.close(); print(json.dumps({"scanned": n, "uk": k, "rate": k / n}))
