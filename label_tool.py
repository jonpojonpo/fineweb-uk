#!/usr/bin/env python
"""Local labelling tool. Python standard library only; runs at http://127.0.0.1:8765 and saves to labels/<you>.jsonl.
  python label_tool.py --annotator your-pseudonym [--tasks tasks/shared_300.jsonl]
"""
import argparse, datetime, hashlib, http.server, json, os, re, webbrowser
from questions import Q, RUBRIC
GUIDE_VERSION = "0.1"; MAX_CHARS = 6000
HELP = {
 "quality": "Add one point per criterion met. See LABELLING.md.",
 "uk_relevance": "Judge the text only. Would it be out of place for a British reader?",
 "genre": "What kind of text is this, whoever published it?",
 "spam_or_boilerplate": "Mostly SEO filler, link lists, menus or templates?",
 "pii_risk": "Could a private person be identified and contacted or harmed? Public officials acting in their role don't count.",
}
SCHEMA = [{"name": n, "prompt": {"quality": "Educational quality (0 to 5 points)"}.get(n, q["prompt"]), "help": HELP[n],
           "options": [{"value": k, "label": k.replace("_", " ")} for k in q["options"]]} for n, q in Q.items()]
VALID = {n: set(q["options"]) for n, q in Q.items()}

def order(tasks, who):
    return sorted(tasks, key=lambda t: hashlib.sha256((who + t["id"]).encode()).hexdigest())

class Store:
    def __init__(s, tasks_path, who, base="."):
        name = os.path.basename(tasks_path)
        docs_path = os.path.join(base, "data", "docs_" + name)
        if not os.path.exists(docs_path): raise SystemExit(f"{docs_path} not found. Run: python fetch_docs.py --tasks {tasks_path}")
        s.docs = {d["id"]: d for d in map(json.loads, open(docs_path))}
        s.tasks = order([json.loads(l) for l in open(tasks_path)], who)
        s.who, s.set = who, name.replace(".jsonl", "")
        os.makedirs(os.path.join(base, "labels"), exist_ok=True)
        s.path = os.path.join(base, "labels", f"{who}.jsonl"); s.done = {}
        if os.path.exists(s.path):
            for r in map(json.loads, open(s.path)):
                if r.get("set") == s.set: s.done[r["id"]] = r  # the latest answer wins
    def doc(s, i):
        t = s.tasks[i]; text = s.docs[t["id"]]["text"]; r = s.done.get(t["id"])
        return {"i": i, "total": len(s.tasks), "text": text[:MAX_CHARS], "cut": len(text) > MAX_CHARS, "labels": r and r.get("labels"), "skipped": r and r.get("skip")}
    def first_open(s):
        return next((i for i, t in enumerate(s.tasks) if t["id"] not in s.done), len(s.tasks) - 1)
    def save(s, i, labels, skip):
        if not 0 <= i < len(s.tasks): raise ValueError("bad index")
        if skip is not None:
            if skip not in ("unclear", "upsetting", "not_english"): raise ValueError("bad skip reason")
            labels = None
        elif set(labels) != set(VALID) or any(labels[k] not in VALID[k] for k in VALID):
            raise ValueError("every question needs a valid answer")
        rec = {"id": s.tasks[i]["id"], "annotator": s.who, "set": s.set, "guide": GUIDE_VERSION, "labels": labels, "skip": skip,
               "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        with open(s.path, "a") as f: f.write(json.dumps(rec) + "\n")
        s.done[rec["id"]] = rec

PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_tool.html")).read()

def handler(store):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(s, *a): pass
        def send(s, code, body, ctype="application/json"):
            b = body.encode(); s.send_response(code); s.send_header("Content-Type", ctype + "; charset=utf-8")
            s.send_header("Content-Length", str(len(b))); s.send_header("Cache-Control", "no-store"); s.end_headers(); s.wfile.write(b)
        def do_GET(s):
            if s.path == "/": return s.send(200, PAGE, "text/html")
            if s.path == "/api/state": return s.send(200, json.dumps({"schema": SCHEMA, "rubric": RUBRIC, "annotator": store.who, "set": store.set, "start": store.first_open(), "done": len(store.done), "total": len(store.tasks)}))
            m = re.fullmatch(r"/api/doc/(\d+)", s.path)
            if m and int(m[1]) < len(store.tasks): return s.send(200, json.dumps(store.doc(int(m[1]))))
            s.send(404, '{"error":"not found"}')
        def do_POST(s):
            # Reject requests from other sites: only this tool's own page may write labels.
            if s.path != "/api/label" or s.headers.get("Content-Type") != "application/json": return s.send(400, '{"error":"bad request"}')
            try:
                d = json.loads(s.rfile.read(int(s.headers.get("Content-Length", 0))))
                store.save(int(d["i"]), d.get("labels"), d.get("skip"))
            except (ValueError, KeyError, TypeError) as e:
                return s.send(400, json.dumps({"error": str(e)}))
            s.send(200, json.dumps({"done": len(store.done)}))
    return H

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--annotator", required=True); a.add_argument("--tasks", default="tasks/shared_300.jsonl"); a.add_argument("--port", type=int, default=8765); a.add_argument("--no-browser", action="store_true")
    x = a.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,30}", x.annotator): raise SystemExit("--annotator must be a pseudonym: lowercase letters, digits and hyphens")
    st = Store(x.tasks, x.annotator)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", x.port), handler(st)); url = f"http://127.0.0.1:{x.port}"
    print(f"{len(st.done)}/{len(st.tasks)} done. Open {url}  (Ctrl+C to stop; progress is saved after every document)")
    if not x.no_browser: webbrowser.open(url)
    srv.serve_forever()
