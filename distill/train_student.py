#!/usr/bin/env python
"""Student: a ModernBERT with one softmax head per decision, trained on the teacher's soft labels.
Holds out a fixed set of documents and reports how closely the student matches the teacher on them,
including calibration against the teacher's own probabilities. Answers the teacher didn't really give
(under half its probability on the allowed options) are left out of that decision's loss.
"""
import argparse, json, math, os, random, statistics as st, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from questions import Q
HEADS = {k: list(v["options"]) for k, v in Q.items() if k != "quality"}
BASE = "answerdotai/ModernBERT-base"; MAX_LEN = 512; MIN_MASS = .5

class Student(nn.Module):
    def __init__(s, base=BASE):
        super().__init__()
        s.enc = AutoModel.from_pretrained(base, dtype=torch.float32)
        s.heads = nn.ModuleDict({k: nn.Linear(s.enc.config.hidden_size, len(v)) for k, v in HEADS.items()})
    def forward(s, **e):
        h = s.enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)  # mean of the document's tokens
        return {k: head(pooled) for k, head in s.heads.items()}

def load(docs_paths, teacher_path, check_path):
    T = {}
    for l in open(teacher_path):
        try: r = json.loads(l)
        except ValueError: continue  # a line cut short when a run was stopped
        if all(k in r for k in HEADS): T[r["id"]] = r
    ids = set(T); text = {}
    for p in docs_paths:
        text.update((d["id"], d["text"]) for d in pq.read_table(p, columns=["id", "text"]).to_pylist() if d["id"] in ids)
    T = {i: r for i, r in T.items() if i in text}
    rows = sorted(T.values(), key=lambda r: r["id"])
    held = {json.loads(l)["id"] for l in open(check_path)} if check_path and os.path.exists(check_path) else set()
    return rows, text, held

def targets(rows):
    out = {}
    for k, opts in HEADS.items():
        out[k] = (torch.tensor([[r[k][o] for o in opts] for r in rows], dtype=torch.float32),
                  torch.tensor([r[k + "__mass"] >= MIN_MASS for r in rows]))
    return out

def predict(m, tok, texts, dev, bs=64):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i])); out = {k: [None] * len(texts) for k in HEADS}
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(order), bs):
            c = order[b:b + bs]; e = tok([texts[i] for i in c], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
            o = m(**e)
            for k in HEADS:
                p = o[k].float().softmax(-1).cpu()
                for j, i in enumerate(c): out[k][i] = p[j]
    return {k: torch.stack(v) for k, v in out.items()}

def report(P, rows, name):
    print(f"--- {name} ({len(rows)} docs)")
    res = {}
    for k, opts in HEADS.items():
        keep = [i for i, r in enumerate(rows) if r[k + "__mass"] >= MIN_MASS]
        q = torch.tensor([[rows[i][k][o] for o in opts] for i in keep]); p = P[k][keep]
        agree = (p.argmax(1) == q.argmax(1)).float().mean().item()
        base = max(torch.bincount(q.argmax(1), minlength=len(opts)).tolist()) / len(keep)
        tv = .5 * (p - q).abs().sum(1).mean().item()
        kl = (q * (q.clamp_min(1e-9).log() - p.clamp_min(1e-9).log())).sum(1).mean().item()
        conf = p.max(1).values; hit = (p.argmax(1) == q.argmax(1)).float(); ece = 0.0
        for lo in torch.arange(0, 1, .1):
            s = (conf > lo) & (conf <= lo + .1)
            if s.any(): ece += s.float().mean().item() * abs(conf[s].mean().item() - hit[s].mean().item())
        extra = {}
        if opts == ["yes", "no"]:  # the rare answer is what matters, so report it directly
            qy, py = q[:, 0] >= .5, p[:, 0] >= .5; tp = (qy & py).sum().item()
            extra = dict(teacher_yes=qy.sum().item(), student_yes=py.sum().item(), recall=tp / max(1, qy.sum().item()), precision=tp / max(1, py.sum().item()))
        print(f"{k:22} agree={agree:.3f} (always-majority={base:.3f}) TV={tv:.3f} KL={kl:.3f} ECE={ece:.3f} n={len(keep)} {extra or ''}")
        res[k] = dict(agree=agree, majority=base, tv=tv, kl=kl, ece=ece, n=len(keep), **extra)
    return res

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--docs", nargs="+", default=["data/edu_v0/docs.parquet"]); a.add_argument("--priority", help="ids chosen because they look rare; never in the random test set, 20%% form their own test"); a.add_argument("--teacher", default="data/edu_v0/teacher.jsonl")
    a.add_argument("--check", default="data/edu_v0/teacher_two_order_check.jsonl"); a.add_argument("--out", default="data/edu_v0/student"); a.add_argument("--test", type=int, default=500)
    a.add_argument("--epochs", type=int, default=3); a.add_argument("--bs", type=int, default=16); a.add_argument("--lr", type=float, default=2e-5); a.add_argument("--limit", type=int, default=0); a.add_argument("--seed", type=int, default=7)
    x = a.parse_args(); torch.manual_seed(x.seed); dev = "cuda"
    rows, text, held = load(x.docs, x.teacher, x.check)
    rng = random.Random(x.seed); free = [r for r in rows if r["id"] not in held]; rng.shuffle(free)
    pri = set(json.load(open(x.priority))) if x.priority else set()
    rand = [r for r in free if r["id"] not in pri]; prior = [r for r in free if r["id"] in pri]
    hard_n = len(prior) // 5; hard, prior = prior[:hard_n], prior[hard_n:]
    test, train = rand[:x.test], rand[x.test:] + prior; rng.shuffle(train)
    if x.limit: train = train[:x.limit]
    check = [json.loads(l) for l in open(x.check)] if x.check and os.path.exists(x.check) else []
    check = [r for r in check if r["id"] in text]
    print(f"train {len(train)}, test {len(test)}, two-order check {len(check)}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE); m = Student().to(dev)
    steps = x.epochs * math.ceil(len(train) / x.bs); opt = torch.optim.AdamW(m.parameters(), lr=x.lr, weight_decay=.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min((i + 1) / max(1, .1 * steps), max(0.0, (steps - i) / max(1, .9 * steps))))
    Y = targets(train); idx = list(range(len(train))); t0 = time.time(); step = 0
    for ep in range(x.epochs):
        m.train(); rng.shuffle(idx)
        for b in range(0, len(idx), x.bs):
            c = idx[b:b + x.bs]; e = tok([text[train[i]["id"]] for i in c], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): o = m(**e)
            loss = 0.0
            for k in HEADS:
                y, ok = Y[k][0][c].to(dev), Y[k][1][c].to(dev)
                if ok.any(): loss = loss + F.cross_entropy(o[k].float()[ok], y[ok])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); step += 1
            if step % 100 == 0: print(f"epoch {ep} step {step}/{steps} loss {float(loss):.3f} {time.time()-t0:.0f}s", flush=True)
        report(predict(m, tok, [text[r["id"]] for r in test], dev), test, f"held-out after epoch {ep}")
    final = report(predict(m, tok, [text[r["id"]] for r in test], dev), test, "held-out, final")
    if hard: final_hard = report(predict(m, tok, [text[r["id"]] for r in hard], dev), hard, "held-out, likely spam or PII")
    if check: final_check = report(predict(m, tok, [text[r["id"]] for r in check], dev), check, "two-order teacher check")
    t = time.time(); predict(m, tok, [text[r["id"]] for r in test] * 4, dev); rate = 4 * len(test) / (time.time() - t)
    os.makedirs(x.out, exist_ok=True); torch.save({"state": m.state_dict(), "heads": HEADS, "base": BASE}, f"{x.out}/student.pt"); tok.save_pretrained(x.out)
    json.dump({"train": len(train), "test": len(test), "metrics": final, "hard_metrics": final_hard if hard else None, "docs_per_s": rate}, open(f"{x.out}/metrics.json", "w"), indent=1)
    print(f"student speed {rate:.0f} docs/s; train time {time.time()-t0:.0f}s; saved {x.out}", flush=True)
