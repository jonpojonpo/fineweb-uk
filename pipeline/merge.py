"""Join documents and labels into one parquet file and print a summary."""
import collections, json, statistics as st, sys
import pyarrow as pa, pyarrow.parquet as pq
d = sys.argv[1]; docs = pq.read_table(f"{d}/docs.parquet").to_pylist()
L = {}
for l in open(f"{d}/d1.jsonl"):
    r = json.loads(l); L[r["id"]] = r["answers"]
E_ = {json.loads(l)["id"]: json.loads(l)["edu"] for l in open(f"{d}/edu.jsonl")}
rows = []; c = collections.defaultdict(collections.Counter); seen = collections.Counter()
for r in docs:
    a = L.get(r["id"])
    if a is None: continue
    seen[r["sha256"]] += 1
    r["edu_score"] = E_.get(r["id"]); r["d1_quality"] = a["quality"]["score"]
    for n in ("uk_relevance", "genre"):
        r[f"d1_{n}"] = a[n]["choice"]; r[f"d1_{n}_probs"] = json.dumps(a[n]["probabilities"]); c[n][a[n]["choice"]] += 1
    for n in ("spam_or_boilerplate", "pii_risk"):
        r[f"d1_{n}_p"] = a[n]["noul"]; c[n][a[n]["noul"] >= .5] += 1
    r["labeller"] = "LiquidAI/d1-3B-GGUF:Q8_0 + fineweb-edu-classifier; first 6000 chars for d1"
    rows.append(r)
pq.write_table(pa.Table.from_pylist(rows), f"{d}/fineweb_uk_sample_v0.parquet", compression="zstd")
T = sum(r["token_count"] for r in rows); hs = collections.Counter(r["host"] for r in rows)
print("docs", len(rows), "tokens", T, "exact-duplicate docs", sum(v - 1 for v in seen.values()), "hosts", len(hs))
for k, v in c.items(): print(k, v.most_common())
for th in (1, 2, 3): print(f"edu>={th}", round(sum(r['token_count'] for r in rows if (r['edu_score'] or 0) >= th) / T, 3), "of tokens")
keep = [r for r in rows if not r["d1_spam_or_boilerplate_p"] >= .5 and r["d1_pii_risk_p"] < .5 and r["d1_uk_relevance"] != "non_uk"]
print("after dropping spam/pii/non_uk:", len(keep), "docs", sum(r["token_count"] for r in keep), "tokens")
print("top hosts", hs.most_common(12)); print("gov.uk docs", sum(r["host"].endswith(".gov.uk") for r in rows))
