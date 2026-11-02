"""Check an extraction against sample v0: same documents, same text. Hashes only, no page text."""
import hashlib, sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1], columns=["sha256"])["sha256"].to_pylist()
got = (len(t), hashlib.sha256("\n".join(sorted(t)).encode()).hexdigest())
want = open("tasks/sample_v0.fingerprint").read().split()
print("docs", got[0], "fingerprint", got[1])
sys.exit(0 if [str(got[0]), got[1]] == want else "does not match tasks/sample_v0.fingerprint")
