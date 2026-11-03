#!/usr/bin/env python
"""Volunteer client. Fetches pinned FineWeb-Edu row groups from Hugging Face, labels the .uk documents here, and uploads only
document ids, text hashes and label probabilities. It treats the coordinator as untrusted: the coordinator only names a
shard and a range, which must match the allowed pattern, and the repo and revision come from pins.env in this checkout.
  python swarm/worker.py --server https://... --name your-pseudonym --croydon jonpojonpo/croydon-v0
  python swarm/worker.py --server https://... --name your-pseudonym --teacher http://127.0.0.1:8092
"""
import argparse, gzip, hashlib, json, os, shlex, sys, time, urllib.error, urllib.request
from urllib.parse import urlparse
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from common import SHARD_RE, MAX_RG_SPAN, KINDS, validate_results

def pins():
    return dict(shlex.split(l)[0].split("=", 1) for l in open(os.path.join(ROOT, "pins.env")) if "=" in l and not l.startswith("#"))

def check_unit(u):
    """Never fetch anything the coordinator names unless it's a shard of the pinned sample, in a sane range."""
    if not (isinstance(u.get("shard"), str) and SHARD_RE.match(u["shard"])): raise ValueError("unit names a file outside the allowed sample")
    a, b = u.get("rg_start"), u.get("rg_end")
    if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b and b - a <= MAX_RG_SPAN): raise ValueError("bad row-group range")
    if u.get("kind") not in KINDS: raise ValueError("bad kind")

def fetch_uk(u, P, token=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token); f = pq.ParquetFile(fs.open(f"datasets/{P['EDU_REPO']}@{P['EDU_DATA_REV']}/{u['shard']}", block_size=4 * 1024 * 1024))
    docs = []
    for i in range(u["rg_start"], min(u["rg_end"], f.num_row_groups)):
        for r in f.read_row_group(i, columns=["id", "text", "url"]).to_pylist():
            try: h = (urlparse(r["url"]).hostname or "").lower()
            except ValueError: continue
            if h.endswith(".uk"): docs.append({"id": r["id"], "text": r["text"], "sha256": hashlib.sha256(r["text"].encode()).hexdigest()})
    return docs

class CroydonBackend:
    kind = "croydon"
    def __init__(s, path):
        sys.path.insert(0, os.path.join(ROOT, "distill"))
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from train_croydon import Croydon, predict
        if not os.path.isdir(path):  # a Hugging Face repo, at the revision pinned in pins.env
            from huggingface_hub import snapshot_download
            P = pins()
            if path != P.get("CROYDON_REPO") or not P.get("CROYDON_REV"): raise SystemExit("--croydon must be a local folder or the CROYDON_REPO pinned in pins.env")
            path = snapshot_download(path, revision=P["CROYDON_REV"], allow_patterns=["model.safetensors", "croydon.json", "tokenizer*", "special_tokens_map.json"])
        cfg = json.load(open(os.path.join(path, "croydon.json")))
        s.predict, s.dev, s.torch = predict, "cuda" if torch.cuda.is_available() else "cpu", torch
        s.m = Croydon(cfg["base"]); s.m.load_state_dict(load_file(os.path.join(path, "model.safetensors"))); s.tok = AutoTokenizer.from_pretrained(path)
        try: s.m.to(s.dev)
        except (torch.OutOfMemoryError, RuntimeError) as e:  # the GPU is full of something else: CPU is slower but works
            if s.dev != "cuda": raise
            print(f"GPU unavailable ({str(e)[:60]}...), using CPU", flush=True); s.dev = "cpu"; s.m.to("cpu"); torch.cuda.empty_cache()
    def label(s, docs):
        if not docs: return []
        for bs in (64, 16, 4):  # a volunteer's GPU may be busy with something else
            try: P = s.predict(s.m, s.tok, [d["text"] for d in docs], s.dev, bs=bs); break
            except (s.torch.OutOfMemoryError, s.torch.AcceleratorError):
                s.torch.cuda.empty_cache()
                if bs == 4:
                    print("GPU out of memory, switching to CPU", flush=True); s.dev = "cpu"; s.m.to("cpu")
                    P = s.predict(s.m, s.tok, [d["text"] for d in docs], s.dev, bs=16)
        return [{"croydon": {str(k): float(v) for k, v in enumerate(p)}} for p in P.tolist()]

class CroydonTeacherBackend:
    kind = "croydon-teacher"
    def __init__(s, url):
        os.environ["TEACHER_URL"] = url; sys.path.insert(0, os.path.join(ROOT, "distill"))
        import teacher_croydon as T
        s.T = T; s.pre, s.suf = T.wrapper()
    def label(s, docs):
        return [{"croydon": s.T.ask(d["text"][:s.T.MAX_CHARS], s.pre, s.suf)[0]} for d in docs]

def call(server, path, body, token=None, gz=False):
    raw = json.dumps(body).encode(); head = {"Content-Type": "application/json"}
    if gz: raw = gzip.compress(raw); head["Content-Encoding"] = "gzip"
    if token: head["Authorization"] = "Bearer " + token
    try: return json.load(urllib.request.urlopen(urllib.request.Request(server + path, raw, head), timeout=300))
    except urllib.error.HTTPError as e: raise RuntimeError(f"{path}: HTTP {e.code} {e.read()[:200]!r}")

def run_unit(u, backend, P, fetch=fetch_uk):
    check_unit(u)
    if u["kind"] != backend.kind: raise ValueError("unit kind doesn't match this worker")
    docs = fetch(u, P); labels = backend.label(docs)
    rows = [{"id": d["id"], "sha256": d["sha256"], "labels": L} for d, L in zip(docs, labels)]
    return validate_results(rows, u["kind"])

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--server", required=True); a.add_argument("--name", required=True); a.add_argument("--invite"); a.add_argument("--croydon", help="Croydon model folder, or the pinned CROYDON_REPO"); a.add_argument("--teacher", help="URL of a local llama-server running the pinned teacher"); a.add_argument("--max-units", type=int, default=0); a.add_argument("--status", action="store_true", help="show your contributions and overall progress, then exit")
    x = a.parse_args()
    if not x.status and bool(x.croydon) == bool(x.teacher): raise SystemExit("choose one of --croydon MODEL or --teacher URL")
    tf = os.path.expanduser(f"~/.config/fineweb-uk-swarm/{x.name}.token")
    if os.path.exists(tf): token = open(tf).read().strip()
    else:
        token = call(x.server, "/register", {"name": x.name, "invite": x.invite})["token"]; os.makedirs(os.path.dirname(tf), exist_ok=True)
        fd = os.open(tf, os.O_WRONLY | os.O_CREAT, 0o600); os.write(fd, token.encode()); os.close(fd)
    if x.status:
        me = call(x.server, "/me", {}, token); st = json.load(urllib.request.urlopen(x.server + "/stats", timeout=60))
        done = st["units"].get("agreed", 0)
        print(f"{me['name']}: {me['accepted']} units confirmed ({me['docs']:,} docs), {me['pending']} awaiting a second volunteer, "
              f"{me['rejected']} rejected, rank {me['rank']} of {me['volunteers']}\nOverall: {done:,} of {st['total_units']:,} units "
              f"({100 * done / max(1, st['total_units']):.1f}%), {st['agreed_docs']:,} documents, {st['active_last_hour']} volunteers active in the last hour. "
              f"Progress page: {x.server}/"); raise SystemExit
    backend = CroydonBackend(x.croydon) if x.croydon else CroydonTeacherBackend(x.teacher); P = pins(); n = fails = 0
    while not x.max_units or n < x.max_units:
        u = call(x.server, "/claim", {"kinds": [backend.kind]}, token)["unit"]
        if not u: print("no work available; waiting", flush=True); time.sleep(300); continue
        t = time.time()
        try:
            rows = run_unit(u, backend, P)
            st = call(x.server, "/submit", {"wu": u["wu"], "results": rows}, token, gz=True)["state"]
        except Exception as e:  # the lease expires and another volunteer gets the unit
            fails += 1; print(f"unit {u['wu']} failed: {str(e)[:200]}", flush=True)
            if fails >= 5: raise SystemExit("stopping after 5 failures in a row")
            time.sleep(min(600, 30 * fails)); continue
        fails = 0; n += 1
        me = call(x.server, "/me", {}, token)
        print(f"unit {u['wu']}: {len(rows)} docs in {time.time()-t:.0f}s, {st} | you: {me['accepted']} units confirmed "
              f"({me['docs']:,} docs), {me['pending']} awaiting a second volunteer, rank {me['rank']} of {me['volunteers']}", flush=True)
