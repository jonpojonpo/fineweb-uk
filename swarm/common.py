"""Shared by coordinator and worker. A work unit (WU) is a row-group range of one pinned FineWeb-Edu shard."""
import re
SHARD_RE = re.compile(r"sample/100BT/\d{3}_\d{5}\.parquet\Z")
LEVELS = [str(i) for i in range(6)]
# Both kinds return the same thing: a probability for each Croydon score, 0-5. "croydon" runs the ModernBERT
# classifier (any GPU or CPU); "croydon-teacher" runs Qwen through a local llama-server (about 20 GB of GPU memory).
KINDS = {"croydon": {"croydon": LEVELS}, "croydon-teacher": {"croydon": LEVELS}}
HEADS = KINDS["croydon"]
ID_RE = re.compile(r"<urn:uuid:[0-9a-f-]{36}>\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_RG_SPAN = 200; MAX_DOCS = 20000

def validate_results(rows, kind="croydon"):
    """Reject anything that isn't an id, a text hash and well-formed probabilities. Uploads never hold page text."""
    if not isinstance(rows, list) or len(rows) > MAX_DOCS: raise ValueError("bad result list")
    seen = set()
    for r in rows:
        if not isinstance(r, dict) or set(r) != {"id", "sha256", "labels"}: raise ValueError("bad row fields")
        if not (isinstance(r["id"], str) and ID_RE.match(r["id"])) or r["id"] in seen: raise ValueError("bad or repeated id")
        if not (isinstance(r["sha256"], str) and SHA_RE.match(r["sha256"])): raise ValueError("bad hash")
        seen.add(r["id"]); L = r["labels"]
        heads = KINDS[kind]
        if not isinstance(L, dict) or set(L) != set(heads): raise ValueError("bad heads")
        for h, opts in heads.items():
            p = L[h]
            if not isinstance(p, dict) or set(p) != set(opts): raise ValueError("bad options")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 for v in p.values()): raise ValueError("bad probability")
            if abs(sum(p.values()) - 1) > .01: raise ValueError("probabilities must sum to 1")
    return rows

def compare(a, b):
    """Agreement between two volunteers' results: same documents and text hashes, then label agreement."""
    A = {r["id"]: r for r in a}; B = {r["id"]: r for r in b}
    if set(A) != set(B) or any(A[i]["sha256"] != B[i]["sha256"] for i in A): return 0.0, 1.0
    if not A: return 1.0, 0.0
    hit = tv = n = 0
    for i in A:
        for h in A[i]["labels"]:
            p, q = A[i]["labels"][h], B[i]["labels"][h]; n += 1
            hit += max(p, key=p.get) == max(q, key=q.get); tv += .5 * sum(abs(p[k] - q[k]) for k in p)
    return hit / n, tv / n
