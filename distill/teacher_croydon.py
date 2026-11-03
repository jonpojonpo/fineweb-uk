#!/usr/bin/env python
"""Croydon teacher: Qwen's probability for each UK-relevance score 0-5, read from the answer token. Appends, so it resumes.
Input: one or more parquet files with id and text. Takes an even share from each, shuffled with a fixed seed.
"""
import argparse, json, math, os, random, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from questions import CROYDON
import pyarrow.parquet as pq
U = os.environ.get("TEACHER_URL", "http://127.0.0.1:8092"); MAX_CHARS = 2400

def post(path, body):
    r = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))

def wrapper():
    p = post("/apply-template", {"messages": [{"role": "user", "content": "\x00"}], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    return p.split("\x00")

def ask(doc, pre, suf):
    prompt = f"{pre}<document>\n{doc}\n</document>\n\n{CROYDON['prompt']}\nAnswer with a single digit, 0 to 5.{suf}"
    r = post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 50, "temperature": 0, "cache_prompt": False})
    lp = {}
    for t in r["completion_probabilities"][0]["top_logprobs"]:
        s = t["token"].strip(); lp[s] = lp.get(s, 0) + math.exp(t["logprob"])
    raw = {k: lp.get(k, 0.0) for k in CROYDON["options"]}; mass = sum(raw.values()); tot = mass + 6e-6
    p = {k: (v + 1e-6) / tot for k, v in raw.items()}
    return p, mass, sum(int(k) * v for k, v in p.items())

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("out"); a.add_argument("src", nargs="+"); a.add_argument("--n", type=int, default=27000); a.add_argument("--seed", type=int, default=20261103)
    x = a.parse_args(); rng = random.Random(x.seed); per = x.n // len(x.src); docs = []
    for s in x.src:
        d = sorted(pq.read_table(s, columns=["id", "text"]).to_pylist(), key=lambda r: r["id"]); rng.shuffle(d)
        docs += [dict(r, src=os.path.basename(os.path.dirname(s)) or s) for r in d[:per]]
    rng.shuffle(docs)
    done = set()
    if os.path.exists(x.out):
        for l in open(x.out):
            try: done.add(json.loads(l)["id"])
            except ValueError: pass
    todo = [d for d in docs if d["id"] not in done]; pre, suf = wrapper(); print(len(docs), "selected,", len(todo), "to do", flush=True); t = time.time()
    with open(x.out, "a") as f:
        for i, d in enumerate(todo, 1):
            p, mass, score = ask(d["text"][:MAX_CHARS], pre, suf)
            f.write(json.dumps({"id": d["id"], "src": d["src"], "croydon": p, "croydon__mass": mass, "croydon_score": score}) + "\n"); f.flush()
            if i % 500 == 0: print(i, round(i / (time.time() - t), 2), "docs/s", flush=True)
    print("done", len(todo), round(time.time() - t), "s", flush=True)
