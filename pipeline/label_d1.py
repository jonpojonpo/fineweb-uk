"""Label documents with d1-3B via /v1/systemone. Appends to the output, so it resumes after a stop."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor as T
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from questions import Q
import pyarrow.parquet as pq
src, out = sys.argv[1:3]
LV = ["meets no criterion", "1 point", "2 points", "3 points", "4 points", "5 points"]
qs = {}
for n, q in Q.items():
    if q.get("ordinal"): qs[n] = {"type": "score", "instructions": q["prompt"], "criteria": LV}
    elif set(q["options"]) == {"yes", "no"}: qs[n] = {"type": "noul", "instructions": q["prompt"]}
    else: qs[n] = {"type": "choice", "instructions": q["prompt"], "criteria": {k: v.replace("_", " ") for k, v in q["options"].items()}}
done = set(json.loads(l)["id"] for l in open(out)) if os.path.exists(out) else set()
tb = pq.read_table(src, columns=["id", "text"]).to_pylist()
todo = [d for d in tb if d["id"] not in done]; print(len(tb), "docs,", len(todo), "to do", flush=True)
def call(d):
    r = urllib.request.Request("http://127.0.0.1:8091/v1/systemone", json.dumps({"state": d["text"][:6000], "questions": qs}).encode(), {"Content-Type": "application/json"})
    for a in range(3):
        try: return d["id"], json.load(urllib.request.urlopen(r, timeout=300))["answers"], None
        except Exception as e: err = str(e)[:200]; time.sleep(2)
    return d["id"], None, err
t = time.time(); fails = 0
with open(out, "a") as f, T(8) as ex:
    for i, (id_, ans, err) in enumerate(ex.map(call, todo), 1):
        if ans is None: fails += 1; print("FAIL", id_, err, flush=True); continue
        f.write(json.dumps({"id": id_, "answers": ans}) + "\n")
        if i % 2000 == 0: f.flush(); print(i, round(i / (time.time() - t), 1), "docs/s", flush=True)
print("done", len(todo), "fails", fails, round(time.time() - t), "s", flush=True)
