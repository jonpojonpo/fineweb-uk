#!/bin/bash
# Re-label with official Qwen, stream sample/100BT alongside, train student v2, then label everything with it. Rerun to resume.
# Usage: distill/run_official.sh /path/to/official.gguf
set -u; cd "$(dirname "$0")/.."; G=$1; D=data/edu_v0; B=data/edu_100bt/shards; PY=${PY:-python3}; SQ=${SQ:-sq}; mkdir -p $B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEACHER_NAME="Qwen3.8-27B official unsloth IQ4_XS" TEACHER_URL=http://127.0.0.1:8092
say() { echo "[$(date +%H:%M)] $*"; }
( for i in $(seq 0 139); do n=$(printf %03d $i); [ -f $B/$n.parquet ] && continue
    for a in 1 2 3 4 5; do $PY distill/extract_shard.py $i $B/$n.parquet 100BT 2>&1 | grep -E '^\{' && break; echo "shard $i attempt $a failed"; sleep 120; done; done; touch $B/EXTRACT_DONE ) > data/edu_100bt/extract.log 2>&1 & X=$!
if [ ! -d $D/student_v2 ]; then
  $SQ down > /dev/null 2>&1; sleep 5
  ${LLAMA_SERVER:-llama-server} -m "$G" -ngl 99 -c 8192 --parallel 1 --port 8092 --host 127.0.0.1 > $D/official_server.log 2>&1 & P=$!
  for i in $(seq 1 120); do curl -s -m 2 localhost:8092/health | grep -q '"ok"' && break; sleep 1; done
  say "official teacher: likely spam and PII first"
  $PY distill/teacher_qwen.py $D/priority.parquet $D/teacher_official.jsonl --n 99999 2>&1 | grep -v Skipping | tail -1
  say "official teacher: random documents"
  $PY distill/teacher_qwen.py $D/docs.parquet $D/teacher_official.jsonl --n 24128 2>&1 | grep -v Skipping | tail -1
  kill $P; sleep 5; say "official labels: $(wc -l < $D/teacher_official.jsonl)"
  $PY distill/train_student.py --docs $D/docs.parquet $D/priority.parquet --teacher $D/teacher_official.jsonl --priority $D/priority_ids.json --check "" --test 1000 --out $D/student_v2 > $D/train_v2.log 2>&1; say "train exit $?"
  $PY distill/export_student.py $D/student_v2 $D/hf_student_v2 2>&1 | tail -1 | cut -c1-120
  $SQ up > /dev/null 2>&1; say "speed-qwen back up"
fi
wait $X; say "100BT extract finished: $(ls $B | grep -cE '^[0-9]+\.parquet$') shards"
for f in $D/shards/[0-9][0-9][0-9].parquet $B/[0-9][0-9][0-9].parquet; do o=${f%.parquet}.v2.parquet; [ -f "$o" ] || $PY distill/apply_student.py "$f" $D/student_v2 "$o" 2>&1 | grep -E "^wrote|Error" | cut -c1-110; done
say "ALL DONE"; touch data/edu_100bt/OFFICIAL_DONE
