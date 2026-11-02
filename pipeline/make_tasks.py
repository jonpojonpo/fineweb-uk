#!/usr/bin/env python
"""Draw the labelling sets from a labelled sample. Output holds ids and hashes only, never page text.
shared: 300 documents for three annotators, over-sampling rare labels so every answer gets tested.
gold: 1000 disjoint documents, 700 random (for calibration) plus 300 rare-label.
"""
import json, random, sys
import pyarrow.parquet as pq
SEED = 20261102
src, shard = sys.argv[1:3]
cols = ["id", "sha256", "host", "edu_score", "d1_uk_relevance", "d1_genre", "d1_spam_or_boilerplate_p", "d1_pii_risk_p"]
R = sorted(pq.read_table(src, columns=cols).to_pylist(), key=lambda r: r["id"])
S = {"gov_uk": lambda r: r["host"].endswith(".gov.uk"), "legal": lambda r: r["d1_genre"] == "legal", "non_uk": lambda r: r["d1_uk_relevance"] == "non_uk",
     "spam": lambda r: r["d1_spam_or_boilerplate_p"] >= .5, "pii": lambda r: r["d1_pii_risk_p"] >= .5, "edu_high": lambda r: r["edu_score"] >= 3}
rng = random.Random(SEED); used = set()
def draw(n, test=lambda r: True):
    pool = [r for r in R if r["id"] not in used and test(r)]; got = rng.sample(pool, min(n, len(pool)))
    used.update(r["id"] for r in got); return got
def build(quota, rand):
    out = [(k, r) for k, n in quota for r in draw(n, S[k])] + [("random", r) for r in draw(rand)]
    rng.shuffle(out); return [{"id": r["id"], "sha256": r["sha256"], "shard": shard, "stratum": k} for k, r in out]
for name, quota, rand in (("shared_300", [(k, 35) for k in S], 90), ("gold_1000", [(k, 50) for k in S], 700)):
    rows = build(quota, rand)
    with open(f"tasks/{name}.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print(name, len(rows))
