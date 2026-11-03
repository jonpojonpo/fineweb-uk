#!/usr/bin/env python
"""Hands out work units, checks each against a second volunteer, and stores agreed labels. Standard library only.
  python swarm/coordinator.py init --db swarm.db --kind croydon --shards 0-139 --span 50
  python swarm/coordinator.py serve --db swarm.db --results results/
Listens on 127.0.0.1. For public use, put it behind an HTTPS reverse proxy, and set SWARM_INVITE to limit sign-ups.
"""
import argparse, gzip, hashlib, hmac, http.server, ipaddress, json, os, re, secrets, sqlite3, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import compare, validate_results, SHARD_RE, KINDS
MAX_BODY = 64 * 1024 * 1024; MAX_RAW = 256 * 1024 * 1024; LEASE = 3600
THRESH = {"croydon": (.97, .05), "croydon-teacher": (.85, .15)}  # top-score agreement, mean probability distance
SCHEMA = """
create table if not exists wu(id integer primary key, kind text, shard text, rg_start int, rg_end int, state text default 'open', rounds int default 0);
create table if not exists volunteer(id integer primary key, name text unique, token_hash text, accepted int default 0, rejected int default 0, joined real);
create table if not exists assign(wu int, volunteer int, lease real, submitted real, path text, primary key(wu, volunteer));
"""
MIGRATIONS = ["alter table assign add column docs int", "alter table wu add column docs int", "alter table wu add column agreed_at real", "alter table assign add column net text"]
H = lambda t: hashlib.sha256(t.encode()).hexdigest()

class Coord:
    def __init__(s, db, results=".", quorum=2, max_rounds=4):
        s.db = sqlite3.connect(db, check_same_thread=False); s.db.executescript(SCHEMA); s.lock = threading.Lock()
        for m in MIGRATIONS:
            try: s.db.execute(m)
            except sqlite3.OperationalError: pass  # column already there
        s.db.commit(); s._cache = (0, None)
        s.dir, s.q, s.max_rounds = results, quorum, max_rounds; os.makedirs(os.path.join(results, "raw"), exist_ok=True); os.makedirs(os.path.join(results, "agreed"), exist_ok=True)
        salt = os.path.join(os.path.dirname(os.path.abspath(db)), "net.salt")  # keyed hash, so stored values can't be reversed to addresses
        if not os.path.exists(salt):
            fd = os.open(salt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, secrets.token_bytes(32)); os.close(fd)
        s.salt = open(salt, "rb").read()
    def net(s, ip):
        """A keyed hash of the volunteer's network: the full IPv4 address, or the /64 for IPv6 (one household or device)."""
        try:
            a = ipaddress.ip_address(ip)
            key = str(ipaddress.ip_network(f"{a}/64", strict=False)) if a.version == 6 else str(a)
        except ValueError: key = "unknown"
        return hmac.new(s.salt, key.encode(), hashlib.sha256).hexdigest()[:32]
    def register(s, name, invite=None):
        need = os.environ.get("SWARM_INVITE")
        if need and invite != need: raise PermissionError("invite code required")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,30}", name or ""): raise ValueError("name must be a pseudonym: lowercase letters, digits and hyphens")
        tok = secrets.token_urlsafe(32)
        with s.lock:
            try: s.db.execute("insert into volunteer(name, token_hash, joined) values(?,?,?)", (name, H(tok), time.time())); s.db.commit()
            except sqlite3.IntegrityError: raise ValueError("name already taken")
        return tok
    def who(s, tok):
        r = s.db.execute("select id from volunteer where token_hash=?", (H(tok or ""),)).fetchone()
        if not r: raise PermissionError("unknown token")
        return r[0]
    def claim(s, tok, kinds, now=None, ip="unknown"):
        now = now or time.time(); v = s.who(tok); n = s.net(ip)
        with s.lock:
            own = s.db.execute("select wu from assign where volunteer=? and submitted is null and lease>?", (v, now)).fetchone()
            if own: return s._wu(own[0])
            marks = ",".join("?" * len(kinds))
            # A unit is available if it needs more results than it has live or finished assignments, and this volunteer hasn't touched it.
            rows = s.db.execute(f"""select w.id, w.rounds from wu w where w.state='open' and w.kind in ({marks})
              and not exists(select 1 from assign a where a.wu=w.id and (a.volunteer=? or a.net=?))
              and (select count(*) from assign a where a.wu=w.id and (a.submitted is not null or a.lease>?)) < ? + w.rounds
              order by (select count(*) from assign a where a.wu=w.id and a.submitted is not null) desc, w.id limit 1""", (*kinds, v, n, now, s.q)).fetchone()
            if not rows: return None
            s.db.execute("insert into assign(wu, volunteer, lease, net) values(?,?,?,?)", (rows[0], v, now + LEASE, n)); s.db.commit()
            return s._wu(rows[0])
    def _wu(s, i):
        k, sh, a, b = s.db.execute("select kind, shard, rg_start, rg_end from wu where id=?", (i,)).fetchone()
        return {"wu": i, "kind": k, "shard": sh, "rg_start": a, "rg_end": b}
    def submit(s, tok, wu, rows):
        v = s.who(tok)
        kind = s.db.execute("select kind from wu where id=?", (wu,)).fetchone()
        if not kind: raise PermissionError("no such unit")
        validate_results(rows, kind[0])
        with s.lock:
            a = s.db.execute("select submitted from assign where wu=? and volunteer=?", (wu, v)).fetchone()
            if not a: raise PermissionError("this unit isn't assigned to you")
            if a[0]: raise ValueError("already submitted")
            path = os.path.join(s.dir, "raw", f"{int(wu)}_{v}.json.gz")
            with gzip.open(path, "wt") as f: json.dump(rows, f)
            s.db.execute("update assign set submitted=?, path=?, docs=? where wu=? and volunteer=?", (time.time(), path, len(rows), wu, v)); s.db.commit()
            return s._check(wu)
    def _check(s, wu):
        kind, state, rounds = s.db.execute("select kind, state, rounds from wu where id=?", (wu,)).fetchone()
        if state != "open": return state
        subs = s.db.execute("select volunteer, path from assign where wu=? and submitted is not null order by submitted", (wu,)).fetchall()
        if len(subs) < s.q: return "waiting"
        loaded = [(v, json.load(gzip.open(p, "rt"))) for v, p in subs]; lo, hi = THRESH[kind]
        for i in range(len(loaded)):
            for j in range(i + 1, len(loaded)):
                agree, tv = compare(loaded[i][1], loaded[j][1])
                if agree >= lo and tv <= hi:
                    with gzip.open(os.path.join(s.dir, "agreed", f"{wu}.json.gz"), "wt") as f: json.dump(loaded[i][1], f)
                    s.db.execute("update wu set state='agreed', docs=?, agreed_at=? where id=?", (len(loaded[i][1]), time.time(), wu))
                    for v, _ in loaded:
                        ok = v in (loaded[i][0], loaded[j][0])
                        s.db.execute(f"update volunteer set {'accepted' if ok else 'rejected'}={'accepted' if ok else 'rejected'}+1 where id=?", (v,))
                    s.db.commit(); return "agreed"
        rounds += 1
        state = "disputed" if rounds >= s.max_rounds else "open"
        s.db.execute("update wu set rounds=?, state=? where id=?", (rounds, state, wu)); s.db.commit(); return state
    def stats(s, now=None):
        """Public progress: overall, per shard, a leaderboard of pseudonyms and recent activity. Cached for 15 seconds."""
        now = now or time.time()
        if s._cache[1] and now - s._cache[0] < 15: return s._cache[1]
        q = lambda sql, *a: s.db.execute(sql, a).fetchall()
        units = dict(q("select state, count(*) from wu group by state")); total = sum(units.values())
        docs = q("select coalesce(sum(docs), 0) from wu where state='agreed'")[0][0]
        waiting = q("select count(distinct a.wu) from assign a join wu w on w.id=a.wu where a.submitted is not null and w.state='open'")[0][0]
        submitted_docs = q("select coalesce(sum(docs), 0) from assign where submitted is not null")[0][0]
        shards = [{"shard": sh.split("/")[-1].removesuffix(".parquet"), "agreed": ag, "total": n}
                  for sh, ag, n in q("select shard, sum(state='agreed'), count(*) from wu group by shard order by shard")]
        board = [dict(zip(("name", "accepted", "docs", "pending", "rejected", "joined"), r)) for r in q("""
            select v.name, v.accepted, coalesce((select sum(a.docs) from assign a join wu w on w.id=a.wu where a.volunteer=v.id and w.state='agreed'), 0),
                   (select count(*) from assign a join wu w on w.id=a.wu where a.volunteer=v.id and a.submitted is not null and w.state='open'),
                   v.rejected, v.joined from volunteer v
            order by v.accepted desc, (select count(*) from assign a where a.volunteer=v.id and a.submitted is not null) desc, v.joined limit 100""")]
        active = q("select count(distinct volunteer) from assign where submitted > ? or (submitted is null and lease > ?)", now - 3600, now)[0][0]
        hour = q("select count(*) from wu where agreed_at > ?", now - 3600)[0][0]
        recent = [dict(zip(("name", "unit", "docs", "at"), r)) for r in q("""
            select v.name, a.wu, a.docs, a.submitted from assign a join volunteer v on v.id=a.volunteer
            where a.submitted is not null order by a.submitted desc limit 12""")]
        out = {"units": units, "total_units": total, "agreed_docs": docs, "units_waiting": waiting, "submitted_docs": submitted_docs, "volunteers": q("select count(*) from volunteer")[0][0],
               "active_last_hour": active, "units_last_hour": hour, "shards": shards, "leaderboard": board, "recent": recent, "generated": now}
        s._cache = (now, out); return out
    def me(s, tok):
        """A volunteer's own record, for their worker to show."""
        v = s.who(tok)
        name, acc, rej = s.db.execute("select name, accepted, rejected from volunteer where id=?", (v,)).fetchone()
        docs = s.db.execute("select coalesce(sum(a.docs), 0) from assign a join wu w on w.id=a.wu where a.volunteer=? and w.state='agreed'", (v,)).fetchone()[0]
        pending = s.db.execute("select count(*) from assign a join wu w on w.id=a.wu where a.volunteer=? and a.submitted is not null and w.state='open'", (v,)).fetchone()[0]
        subs = lambda who: f"(select count(*) from assign a where a.volunteer={who} and a.submitted is not null)"
        mine = s.db.execute(f"select {subs('?')}", (v,)).fetchone()[0]
        rank = s.db.execute(f"select count(*) + 1 from volunteer v where v.accepted > ? or (v.accepted = ? and {subs('v.id')} > ?)", (acc, acc, mine)).fetchone()[0]
        return {"name": name, "accepted": acc, "docs": docs, "pending": pending, "rejected": rej, "rank": rank, "volunteers": s.db.execute("select count(*) from volunteer").fetchone()[0]}

PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.html")).read()

def serve(c, host, port):
    class H_(http.server.BaseHTTPRequestHandler):
        def log_message(s, *a): pass
        def send(s, code, obj):
            b = json.dumps(obj).encode(); s.send_response(code); s.send_header("Content-Type", "application/json"); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
        def do_GET(s):
            if s.path in ("/", "/index.html"):
                b = PAGE.encode(); s.send_response(200); s.send_header("Content-Type", "text/html; charset=utf-8"); s.send_header("Content-Length", str(len(b)))
                s.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'"); s.end_headers(); s.wfile.write(b); return
            if s.path == "/stats": return s.send(200, c.stats())
            s.send(404, {"error": "not found"})
        def do_POST(s):
            try:
                n = int(s.headers.get("Content-Length", 0))
                if n > MAX_BODY: return s.send(413, {"error": "too large"})
                raw = s.rfile.read(n)
                if s.headers.get("Content-Encoding") == "gzip":
                    import zlib
                    d = zlib.decompressobj(31); raw = d.decompress(raw, MAX_RAW)
                    if d.unconsumed_tail: return s.send(413, {"error": "too large"})
                d = json.loads(raw); tok = (s.headers.get("Authorization") or "").removeprefix("Bearer ")
                # Behind a Cloudflare tunnel every request arrives from loopback, and Cloudflare sets the real address.
                # Only trust that header from loopback; the coordinator itself listens on loopback only.
                peer = s.client_address[0]; ip = (s.headers.get("CF-Connecting-IP") or peer) if peer in ("127.0.0.1", "::1") else peer
                if s.path == "/register": return s.send(200, {"token": c.register(d.get("name"), d.get("invite"))})
                if s.path == "/me": return s.send(200, c.me(tok))
                if s.path == "/claim": return s.send(200, {"unit": c.claim(tok, [k for k in d.get("kinds", []) if k in KINDS], ip=ip)})
                if s.path == "/submit": return s.send(200, {"state": c.submit(tok, int(d["wu"]), d["results"])})
                s.send(404, {"error": "not found"})
            except PermissionError as e: s.send(403, {"error": str(e)})
            except (ValueError, KeyError, TypeError, OSError) as e: s.send(400, {"error": str(e)[:200]})
    srv = http.server.ThreadingHTTPServer((host, port), H_); return srv

if __name__ == "__main__":
    a = argparse.ArgumentParser(); sub = a.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("--db", required=True); i.add_argument("--kind", choices=sorted(KINDS), required=True); i.add_argument("--shards", default="0-139", help="positions in tasks/edu_100bt_files.json"); i.add_argument("--span", type=int, default=50); i.add_argument("--rowgroups", type=int, default=726, help="row groups per shard; workers tolerate a range that runs past the end")
    v = sub.add_parser("serve"); v.add_argument("--db", required=True); v.add_argument("--results", default="swarm_results"); v.add_argument("--host", default="127.0.0.1"); v.add_argument("--port", type=int, default=8770)
    x = a.parse_args()
    if x.cmd == "init":
        lo, hi = map(int, x.shards.split("-")); c = Coord(x.db, os.path.dirname(os.path.abspath(x.db))); n = 0
        files = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks", "edu_100bt_files.json")))
        for name in files[lo:hi + 1]:
            assert SHARD_RE.match(name)
            for st in range(0, x.rowgroups, x.span):
                c.db.execute("insert into wu(kind, shard, rg_start, rg_end) values(?,?,?,?)", (x.kind, name, st, st + x.span)); n += 1
        c.db.commit(); print("created", n, "units")
    else:
        srv = serve(Coord(x.db, x.results), x.host, x.port); print(f"coordinator on {x.host}:{x.port}"); srv.serve_forever()
