#!/usr/bin/env python
"""Agreement between annotators (Cohen's kappa), and optionally a model's accuracy and calibration against them.
  python score.py [--set shared_300] [--model data/sample_v0/d1.jsonl]
A decision with mean kappa below 0.6 needs its definition rewritten. Skipped documents aren't scored.
"""
import argparse, collections, glob, itertools, json
from questions import Q

def kappa(a, b, weights=None):
    n = len(a)
    if not n: return None
    cats = sorted(set(a) | set(b))
    d = weights or (lambda x, y: float(x != y))
    po = sum(d(x, y) for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[x] * cb[y] * d(x, y) for x in cats for y in cats) / n ** 2
    return None if pe == 0 else 1 - po / pe

def load(setname):
    who = collections.defaultdict(dict)
    for p in glob.glob("labels/*.jsonl"):
        for r in map(json.loads, open(p)):
            if r.get("set") == setname and r.get("labels"): who[r["annotator"]][r["id"]] = r["labels"]
    return who

def agreement(who):
    out = {}
    for n, q in Q.items():
        w = (lambda x, y: abs(int(x) - int(y)) / 5) if q.get("ordinal") else None
        ks = []
        for a, b in itertools.combinations(sorted(who), 2):
            ids = sorted(set(who[a]) & set(who[b]))
            k = kappa([who[a][i][n] for i in ids], [who[b][i][n] for i in ids], w)
            if k is not None: ks.append(k)
        out[n] = sum(ks) / len(ks) if ks else None
    return out

def model_probs(ans, n):
    a = ans[n]
    return {"yes": a["noul"], "no": 1 - a["noul"]} if "noul" in a else a["probabilities"]

def ece(pairs, bins=10):
    b = collections.defaultdict(list)
    for conf, hit in pairs: b[min(int(conf * bins), bins - 1)].append((conf, hit))
    return sum(abs(sum(c for c, _ in v) / len(v) - sum(h for _, h in v) / len(v)) * len(v) for v in b.values()) / len(pairs) if pairs else None

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--set", default="shared_300"); a.add_argument("--model"); x = a.parse_args()
    who = load(x.set); print("annotators:", {k: len(v) for k, v in sorted(who.items())})
    if len(who) >= 2:
        for n, k in agreement(who).items(): print(f"{n:22} kappa={'n/a' if k is None else f'{k:.3f}'}{' REDEFINE' if k is not None and k < .6 else ''}")
    else: print("Agreement needs at least two annotators.")
    if x.model:
        M = {r["id"]: r["answers"] for r in map(json.loads, open(x.model))}
        for n in Q:
            votes = collections.defaultdict(collections.Counter)
            for v in who.values():
                for i, l in v.items(): votes[i][l[n]] += 1
            pairs = []
            for i, c in votes.items():
                top = c.most_common(2)
                if i not in M or (len(top) == 2 and top[0][1] == top[1][1]): continue
                p = model_probs(M[i], n); pick = max(p, key=p.get); pairs.append((p[pick], float(pick == top[0][0])))
            if pairs: print(f"{n:22} model accuracy={sum(h for _, h in pairs)/len(pairs):.3f} ECE={ece(pairs):.3f} n={len(pairs)}")
        print("The shared set over-samples rare labels, so use gold_1000's random stratum for real-world accuracy.")
