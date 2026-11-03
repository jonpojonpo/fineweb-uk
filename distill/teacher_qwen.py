#!/usr/bin/env python
"""Teacher: Qwen answer-token probabilities for each decision, with no output text parsed.
Each document gets its own random option order per question, so order bias averages out across the
training set, which the student can't see. Asks one question per prompt: speed-qwen's draft model returns
no probabilities for accepted tokens, so only an answer's first token is usable. Appends, so it resumes.
"""
import argparse, json, math, os, random, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from questions import Q
import pyarrow.parquet as pq
LETTERS = "ABCDEFGH"; MAX_CHARS = 2400; U = os.environ.get("TEACHER_URL", "http://127.0.0.1:8080")
QS = {k: v for k, v in Q.items() if k != "quality"}  # FineWeb-Edu's score already covers quality

def post(path, body):
    r = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))

def wrapper():
    """The chat wrapper around a user message, from the model's own template, with thinking off."""
    p = post("/apply-template", {"messages": [{"role": "user", "content": "\x00"}], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    return p.split("\x00")

def ask(doc, q, rng, pre, suf):
    keys = list(q["options"]); rng.shuffle(keys)
    lines = "\n".join(f"{LETTERS[i]}. {q['options'][k].replace('_', ' ')}" for i, k in enumerate(keys))
    prompt = f"{pre}<document>\n{doc}\n</document>\n\n{q['prompt']}\n{lines}\nAnswer with a single letter.{suf}"
    r = post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 50, "temperature": 0, "cache_prompt": True})
    lp = {}
    for t in r["completion_probabilities"][0]["top_logprobs"]:
        s = t["token"].strip()
        lp[s] = lp.get(s, 0) + math.exp(t["logprob"])
    raw = {k: lp.get(LETTERS[i], 0.0) for i, k in enumerate(keys)}; mass = sum(raw.values())
    eps = 1e-6; tot = mass + eps * len(raw)
    return {k: (v + eps) / tot for k, v in raw.items()}, mass, r.get("timings", {}).get("prompt_n", 0)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("src"); a.add_argument("out"); a.add_argument("--n", type=int, default=12000); a.add_argument("--seed", type=int, default=20261102)
    x = a.parse_args()
    docs = sorted(pq.read_table(x.src, columns=["id", "text"]).to_pylist(), key=lambda d: d["id"]); random.Random(x.seed).shuffle(docs); docs = docs[:x.n]
    done = set()
    if os.path.exists(x.out):
        for l in open(x.out):
            try: done.add(json.loads(l)["id"])
            except ValueError: pass  # a line cut short when a run was stopped
    todo = [d for d in docs if d["id"] not in done]; pre, suf = wrapper(); print(len(docs), "selected,", len(todo), "to do", flush=True)
    t = time.time()
    with open(x.out, "a") as f:
        for i, d in enumerate(todo, 1):
            text = d["text"][:MAX_CHARS]; rec = {"id": d["id"]}; rng = random.Random(d["id"])
            for n, q in QS.items():
                rec[n], rec[n + "__mass"], _ = ask(text, q, rng, pre, suf)
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 200 == 0: print(i, round(i / (time.time() - t), 2), "docs/s", flush=True)
    print("done", len(todo), round(time.time() - t), "s", flush=True)
