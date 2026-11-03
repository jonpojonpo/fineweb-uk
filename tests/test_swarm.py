import gzip, json, os, sys, tempfile, unittest
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "swarm")]
from common import HEADS, validate_results, compare
from coordinator import Coord, LEASE
from worker import check_unit
def lab(flip=False):
    p = {str(k): 0.0 for k in range(6)}; p["0" if flip else "5"] = 1.0
    return {"croydon": p}
ID = "<urn:uuid:%s>"; I = lambda n: ID % f"00000000-0000-0000-0000-{n:012d}"
def rows(flip=False, n=3): return [{"id": I(i), "sha256": "a" * 64, "labels": lab(flip)} for i in range(n)]
class T(unittest.TestCase):
    def setUp(s):
        s.d = tempfile.mkdtemp(); s.c = Coord(os.path.join(s.d, "x.db"), s.d)
        s.c.db.execute("insert into wu(kind, shard, rg_start, rg_end) values('croydon','sample/100BT/000_00000.parquet',0,50)"); s.c.db.commit()
        s.t = [s.c.register(f"v{i}x") for i in range(5)]
    def test_two_agree(s):
        u = s.c.claim(s.t[0], ["croydon"], ip="81.2.69.1"); s.assertEqual(s.c.submit(s.t[0], u["wu"], rows()), "waiting")
        u2 = s.c.claim(s.t[1], ["croydon"], ip="81.2.69.2"); s.assertEqual(u2["wu"], u["wu"]); s.assertEqual(s.c.submit(s.t[1], u["wu"], rows()), "agreed")
        s.assertTrue(os.path.exists(os.path.join(s.d, "agreed", f"{u['wu']}.json.gz")))
    def test_same_volunteer_cant_confirm_self(s):
        u = s.c.claim(s.t[0], ["croydon"]); s.c.submit(s.t[0], u["wu"], rows()); s.assertIsNone(s.c.claim(s.t[0], ["croydon"]))
        with s.assertRaises(ValueError): s.c.submit(s.t[0], u["wu"], rows())
    def test_cant_submit_unassigned(s):
        with s.assertRaises(PermissionError): s.c.submit(s.t[0], 1, rows())
        with s.assertRaises(PermissionError): s.c.claim("forged", ["croydon"])
    def test_disagreement_needs_third_and_rejects_liar(s):
        u = s.c.claim(s.t[0], ["croydon"], ip="81.2.69.1"); s.c.submit(s.t[0], u["wu"], rows())
        s.c.claim(s.t[1], ["croydon"], ip="81.2.69.2"); s.assertEqual(s.c.submit(s.t[1], u["wu"], rows(True)), "open")
        s.c.claim(s.t[2], ["croydon"], ip="81.2.69.3"); s.assertEqual(s.c.submit(s.t[2], u["wu"], rows()), "agreed")
        st = {x["name"]: x for x in s.c.stats()["leaderboard"]}
        s.assertEqual(st["v1x"]["rejected"], 1); s.assertEqual(st["v0x"]["accepted"], 1)
    def test_different_docs_never_agree(s):
        s.assertEqual(compare(rows(), rows(n=2))[0], 0.0)
        tampered = rows(); tampered[0]["sha256"] = "b" * 64; s.assertEqual(compare(rows(), tampered)[0], 0.0)
    def test_lease_expiry_reissues(s):
        u = s.c.claim(s.t[0], ["croydon"], now=1000, ip="81.2.69.1"); s.c.claim(s.t[1], ["croydon"], now=1000, ip="81.2.69.2")
        s.assertIsNone(s.c.claim(s.t[2], ["croydon"], now=1001, ip="81.2.69.3"))
        s.assertEqual(s.c.claim(s.t[2], ["croydon"], now=1000 + LEASE + 1, ip="81.2.69.3")["wu"], u["wu"])
    def test_rejects_page_text_and_bad_values(s):
        r = rows(); r[0]["text"] = "page"
        with s.assertRaises(ValueError): validate_results(r)
        r = rows(); r[0]["labels"]["croydon"]["3"] = 7
        with s.assertRaises(ValueError): validate_results(r)
        r = rows(); r[1]["id"] = r[0]["id"]
        with s.assertRaises(ValueError): validate_results(r)
    def test_worker_refuses_other_files(s):
        for bad in ("../../etc/passwd", "sample/100BT/000_00000.parquet/../../x", "data/x.parquet"):
            with s.assertRaises(ValueError): check_unit({"shard": bad, "rg_start": 0, "rg_end": 5, "kind": "croydon"})
        with s.assertRaises(ValueError): check_unit({"shard": "sample/100BT/000_00000.parquet", "rg_start": 0, "rg_end": 100000, "kind": "croydon"})
        with s.assertRaises(ValueError): check_unit({"shard": "sample/100BT/000_00000.parquet", "rg_start": 0, "rg_end": 5, "kind": "student"})
        check_unit({"shard": "sample/100BT/000_00000.parquet", "rg_start": 0, "rg_end": 50, "kind": "croydon"})
    def test_same_network_cant_confirm(s):
        u = s.c.claim(s.t[0], ["croydon"], ip="81.2.69.7"); s.c.submit(s.t[0], u["wu"], rows())
        s.assertIsNone(s.c.claim(s.t[1], ["croydon"], ip="81.2.69.7"))  # a second account at the same address
        s.assertEqual(s.c.claim(s.t[1], ["croydon"], ip="2a00:23c6:1:2::5")["wu"], u["wu"])  # a different network may
        s.c.db.execute("delete from assign"); s.c.db.commit()
        u = s.c.claim(s.t[0], ["croydon"], ip="2a00:23c6:1:2::5"); s.c.submit(s.t[0], u["wu"], rows())
        s.assertIsNone(s.c.claim(s.t[1], ["croydon"], ip="2a00:23c6:1:2:ffff::9"))  # same IPv6 /64
        s.assertEqual(s.c.claim(s.t[1], ["croydon"], ip="2a00:23c6:1:3::9")["wu"], u["wu"])  # different /64
        stored = [r[0] for r in s.c.db.execute("select net from assign")]
        s.assertTrue(all("2a00" not in x and "81.2" not in x for x in stored))  # never the address itself
    def test_progress_and_me(s):
        u = s.c.claim(s.t[0], ["croydon"], ip="81.2.69.1"); s.c.submit(s.t[0], u["wu"], rows())
        s.c.claim(s.t[1], ["croydon"], ip="81.2.69.2"); s.c.submit(s.t[1], u["wu"], rows())
        st = s.c.stats(now=1e12)
        s.assertEqual(st["units"]["agreed"], 1); s.assertEqual(st["agreed_docs"], 3); s.assertEqual(st["shards"][0]["agreed"], 1)
        top = {r["name"]: r for r in st["leaderboard"]}; s.assertEqual(top["v0x"]["docs"], 3); s.assertEqual(top["v0x"]["accepted"], 1)
        me = s.c.me(s.t[0]); s.assertEqual((me["accepted"], me["docs"], me["pending"], me["rank"]), (1, 3, 0, 1))
        s.assertEqual(s.c.me(s.t[4])["rank"], 3)
        with s.assertRaises(PermissionError): s.c.me("forged")
    def test_invite(s):
        os.environ["SWARM_INVITE"] = "abc"
        try:
            with s.assertRaises(PermissionError): s.c.register("newbie")
            s.assertTrue(s.c.register("newbie", "abc"))
        finally: del os.environ["SWARM_INVITE"]
if __name__ == "__main__": unittest.main()
