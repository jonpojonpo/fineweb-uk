#!/usr/bin/env python
"""Documents and tokens in each (FineWeb-Edu score, Croydon score) square, both rounded to whole points."""
import collections, glob, json, os, re, sys
import pyarrow.parquet as pq
def grid(pairs, title):
    D, T = collections.Counter(), collections.Counter(); n = t = 0
    for e, c, tok in pairs:
        k = (min(5, max(0, round(e))), min(5, max(0, round(c)))); D[k] += 1; T[k] += tok; n += 1; t += tok
    print(f"\n{title}: {n:,} docs, {t:,} tokens. Rows: FineWeb-Edu score. Columns: Croydon score. Cells: % of tokens")
    print("edu\\uk " + "".join(f"{c:>7}" for c in range(6)) + "    docs")
    for e in range(5, -1, -1):
        print(f"{e:>6} " + "".join(f"{100 * T[(e, c)] / max(1, t):>7.1f}" for c in range(6)) + f"{sum(D[(e, c)] for c in range(6)):>9,}")
    return {f"{e},{c}": [D[(e, c)], T[(e, c)]] for e in range(6) for c in range(6)}
if __name__ == "__main__":
    out = {}
    def load(docs_glob, edu_col, score_suffix):
        pairs = []
        for p in sorted(glob.glob(docs_glob)):
            if not re.fullmatch(r"(\d{3}(_\d{5})?|scored)\.parquet", os.path.basename(p)): continue  # source shards only, not derived files
            s = p[:-len(".parquet")] + score_suffix
            try: S = {r["id"]: r["croydon_score"] for r in pq.read_table(s, columns=["id", "croydon_score"]).to_pylist()}
            except FileNotFoundError: continue
            for r in pq.read_table(p, columns=["id", edu_col, "token_count"]).to_pylist():
                if r["id"] in S: pairs.append((r[edu_col], S[r["id"]], r["token_count"]))
        return pairs
    out["fineweb_plain"] = grid(load("data/fineweb_v0/scored.parquet", "edu_score", ".croydon.parquet"), "Plain FineWeb .uk, one shard")
    out["edu_10bt"] = grid(load("data/edu_v0/shards/[0-9][0-9][0-9].parquet", "score", ".croydon.parquet"), "FineWeb-Edu .uk, 10BT sample")
    out["edu_100bt"] = grid(load("data/edu_100bt/shards/[0-9]*[0-9].parquet", "score", ".croydon.parquet"), "FineWeb-Edu .uk, 100BT sample")
    json.dump(out, open("data/croydon/grid.json", "w"))
