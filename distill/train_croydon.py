#!/usr/bin/env python
"""Croydon: a ModernBERT with one 0-5 head, trained on the teacher's score distribution. The score it reports is the
probability-weighted average, like FineWeb-Edu's. Holds out a random test set and reports agreement with the teacher.
"""
import argparse, json, math, os, random, statistics as st, sys, time
import pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
BASE = "answerdotai/ModernBERT-base"; MAX_LEN = 512; LEVELS = [str(i) for i in range(6)]

class Croydon(nn.Module):
    def __init__(s, base=BASE):
        super().__init__(); s.enc = AutoModel.from_pretrained(base, dtype=torch.float32); s.head = nn.Linear(s.enc.config.hidden_size, 6)
    def forward(s, **e):
        h = s.enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        return s.head((h * m).sum(1) / m.sum(1))

def predict(m, tok, texts, dev, bs=64):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i])); out = [None] * len(texts); m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for b in range(0, len(order), bs):
            c = order[b:b + bs]; e = tok([texts[i] for i in c], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
            p = m(**e).float().softmax(-1).cpu()
            for j, i in enumerate(c): out[i] = p[j]
    return torch.stack(out)

def score(p): return (p * torch.arange(6, dtype=p.dtype)).sum(-1)

def rank(x):
    o = sorted(range(len(x)), key=x.__getitem__); r = [0] * len(x)
    for a, i in enumerate(o): r[i] = a
    return r

def report(P, rows, name):
    q = torch.tensor([[r["croydon"][k] for k in LEVELS] for r in rows]); sp, sq = score(P).tolist(), score(q).tolist()
    mae = st.mean(abs(a - b) for a, b in zip(sp, sq)); rho = st.correlation(rank(sp), rank(sq))
    exact = (P.argmax(1) == q.argmax(1)).float().mean().item(); within1 = st.mean(abs(round(a) - round(b)) <= 1 for a, b in zip(sp, sq))
    hi = [i for i, v in enumerate(sq) if v >= 3.5]; hp = [i for i, v in enumerate(sp) if v >= 3.5]
    rec = len(set(hi) & set(hp)) / max(1, len(hi)); prec = len(set(hi) & set(hp)) / max(1, len(hp))
    print(f"--- {name} ({len(rows)} docs): spearman={rho:.3f} MAE={mae:.3f} exact={exact:.3f} within1={within1:.3f} | score>=3.5: recall={rec:.3f} precision={prec:.3f} | mean teacher={st.mean(sq):.2f} student={st.mean(sp):.2f}", flush=True)
    return dict(spearman=rho, mae=mae, exact=exact, within1=within1, recall_35=rec, precision_35=prec)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--teacher", default="data/croydon/teacher.jsonl"); a.add_argument("--docs", nargs="+", default=["data/edu_v0/docs.parquet", "data/fineweb_v0/docs.parquet"])
    a.add_argument("--out", default="data/croydon/croydon_v0"); a.add_argument("--test", type=int, default=1500); a.add_argument("--epochs", type=int, default=3); a.add_argument("--bs", type=int, default=16)
    a.add_argument("--lr", type=float, default=3e-5); a.add_argument("--limit", type=int, default=0); a.add_argument("--seed", type=int, default=7)
    x = a.parse_args(); torch.manual_seed(x.seed); dev = "cuda"; rng = random.Random(x.seed)
    T = {}
    for l in open(x.teacher):
        try: r = json.loads(l)
        except ValueError: continue
        if r.get("croydon__mass", 0) >= .5: T[r["id"]] = r
    text = {}
    for p in x.docs: text.update((d["id"], d["text"]) for d in pq.read_table(p, columns=["id", "text"]).to_pylist() if d["id"] in T)
    rows = sorted((r for r in T.values() if r["id"] in text), key=lambda r: r["id"]); rng.shuffle(rows)
    test, train = rows[:x.test], rows[x.test:]
    if x.limit: train = train[:x.limit]
    print(f"train {len(train)}, test {len(test)}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE); m = Croydon().to(dev)
    steps = x.epochs * math.ceil(len(train) / x.bs); opt = torch.optim.AdamW(m.parameters(), lr=x.lr, weight_decay=.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min((i + 1) / max(1, .1 * steps), max(0.0, (steps - i) / max(1, .9 * steps))))
    Y = torch.tensor([[r["croydon"][k] for k in LEVELS] for r in train]); idx = list(range(len(train))); t0 = time.time(); step = 0
    for ep in range(x.epochs):
        m.train(); rng.shuffle(idx)
        for b in range(0, len(idx), x.bs):
            c = idx[b:b + x.bs]; e = tok([text[train[i]["id"]] for i in c], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): o = m(**e)
            loss = F.cross_entropy(o.float(), Y[c].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); step += 1
            if step % 200 == 0: print(f"epoch {ep} step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        res = report(predict(m, tok, [text[r["id"]] for r in test], dev), test, f"held-out after epoch {ep}")
    by_src = {}
    for s in sorted({r["src"] for r in test}):
        sub = [r for r in test if r["src"] == s]; by_src[s] = report(predict(m, tok, [text[r["id"]] for r in sub], dev), sub, f"held-out, {s} only")
    t = time.time(); predict(m, tok, [text[r["id"]] for r in test], dev); rate = len(test) / (time.time() - t)
    os.makedirs(x.out, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().clone() for k, v in m.state_dict().items()}, f"{x.out}/model.safetensors"); tok.save_pretrained(x.out)
    json.dump({"name": "Croydon", "base": BASE, "levels": LEVELS, "max_length": MAX_LEN, "pooling": "mean", "score": "probability-weighted mean of 0-5",
               "teacher": os.environ.get("TEACHER_NAME", "Qwen3.8-27B official unsloth IQ4_XS"), "train": len(train), "test": len(test), "metrics": res, "metrics_by_source": by_src, "docs_per_s": rate},
              open(f"{x.out}/croydon.json", "w"), indent=1)
    print(f"speed {rate:.0f} docs/s; train {time.time()-t0:.0f}s; saved {x.out}", flush=True)
