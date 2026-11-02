"""Run with: python -m unittest discover tests"""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from label_tool import Store, order
from score import kappa, ece
GOOD = {"quality": "2", "uk_relevance": "non_uk", "genre": "news", "spam_or_boilerplate": "no", "pii_risk": "no"}
class T(unittest.TestCase):
    def setUp(s):
        s.d = tempfile.mkdtemp(); os.makedirs(s.d + "/data"); s.t = s.d + "/t.jsonl"
        with open(s.t, "w") as f, open(s.d + "/data/docs_t.jsonl", "w") as g:
            for i in range(3):
                f.write(json.dumps({"id": f"d{i}", "sha256": "x", "shard": "s", "stratum": "random"}) + "\n")
                g.write(json.dumps({"id": f"d{i}", "text": "x" * (7000 if i == 0 else 5)}) + "\n")
    def test_save_resume_latest_wins(s):
        a = Store(s.t, "ann", s.d); first = a.tasks[0]["id"]
        a.save(0, GOOD, None); a.save(0, {**GOOD, "quality": "4"}, None)
        b = Store(s.t, "ann", s.d)
        s.assertEqual(b.done[first]["labels"]["quality"], "4"); s.assertEqual(b.first_open(), 1)
    def test_rejects_incomplete_and_bad(s):
        a = Store(s.t, "ann", s.d)
        for bad in ({"quality": "2"}, {**GOOD, "genre": "nope"}):
            with s.assertRaises(ValueError): a.save(0, bad, None)
        with s.assertRaises(ValueError): a.save(9, GOOD, None)
        with s.assertRaises(ValueError): a.save(0, None, "because")
    def test_skip_saved(s):
        a = Store(s.t, "ann", s.d); a.save(0, None, "upsetting"); s.assertEqual(a.first_open(), 1)
    def test_cut_flag_and_no_stratum_leak(s):
        a = Store(s.t, "ann", s.d)
        got = [a.doc(i) for i in range(3)]
        s.assertEqual(sum(g["cut"] for g in got), 1); s.assertTrue(all(len(g["text"]) <= 6000 and "stratum" not in g for g in got))
    def test_order_differs_by_annotator(s):
        ts = [{"id": f"d{i}"} for i in range(30)]; s.assertNotEqual(order(ts, "a1"), order(ts, "a2"))
    def test_kappa(s):
        s.assertEqual(kappa(list("aabb"), list("aabb")), 1.0)
        s.assertAlmostEqual(kappa(list("aabb"), list("abab")), 0.0)
        s.assertAlmostEqual(kappa(list("aaabbb"), list("aaabbb"), lambda x, y: float(x != y)), 1.0)
    def test_ece(s):
        s.assertAlmostEqual(ece([(.9, 1.0)] * 9 + [(.9, 0.0)]), 0.0)
        s.assertAlmostEqual(ece([(.9, 0.0)] * 10), .9)
if __name__ == "__main__": unittest.main()
