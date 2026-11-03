#!/bin/bash
# Label with Qwen until the deadline, widen the pool with the current student, then train and apply a new student.
# Usage: distill/run_unattended.sh HOURS
set -u; cd "$(dirname "$0")/.."; D=data/edu_v0; PY=${PY:-python3}; SQ=${SQ:-sq}; mkdir -p $D/shards
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEACHER_NAME="HauhauCS Qwen3.8-27B uncensored IQ4_XS"
END=$(( $(date +%s) + $(python3 -c "print(int(float('$1')*3600))") )); say() { echo "[$(date +%H:%M)] $*"; }
say "phase 1: random labels while widening the pool"
$PY distill/teacher_qwen.py $D/docs.parquet $D/teacher.jsonl --n 24128 > $D/teacher_random.log 2>&1 & T=$!
cp -n $D/docs.parquet $D/shards/000.parquet
for i in $(seq 1 13); do n=$(printf %03d $i); [ -f $D/shards/$n.parquet ] && continue
  for a in 1 2 3; do $PY distill/extract_shard.py $i $D/shards/$n.parquet 2>&1 | grep -E '^\{' && break; say "shard $i attempt $a failed"; sleep 60; done; done
for f in $D/shards/[0-9][0-9][0-9].parquet; do o=${f%.parquet}.labelled.parquet; [ -f $o ] || $PY distill/apply_student.py $f $D/student $o 2>&1 | grep -E "^wrote|Error" | cut -c1-120; done
$PY distill/make_queue.py $D 1500 2>&1 | grep -E "^pool|Error"
kill $T 2>/dev/null; wait $T 2>/dev/null; say "random labels so far: $(wc -l < $D/teacher.jsonl)"
say "phase 2: likely spam and PII"
left=$(( END - $(date +%s) )); [ $left -gt 0 ] && timeout $left $PY distill/teacher_qwen.py $D/priority.parquet $D/teacher.jsonl --n 99999 > $D/teacher_priority.log 2>&1
say "labels so far: $(wc -l < $D/teacher.jsonl)"
say "phase 3: random labels until the deadline"
left=$(( END - $(date +%s) )); [ $left -gt 0 ] && timeout $left $PY distill/teacher_qwen.py $D/docs.parquet $D/teacher.jsonl --n 24128 >> $D/teacher_random.log 2>&1
say "teacher finished: $(wc -l < $D/teacher.jsonl) labels"
say "phase 4: train student v1 and label every shard"
$SQ down > /dev/null 2>&1; sleep 5
$PY distill/train_student.py --docs $D/docs.parquet $D/priority.parquet --priority $D/priority_ids.json --test 1000 --out $D/student_v1 > $D/train_v1.log 2>&1; say "train exit $?"
if [ -f $D/student_v1/student.pt ]; then
  for f in $D/shards/[0-9][0-9][0-9].parquet; do $PY distill/apply_student.py $f $D/student_v1 ${f%.parquet}.v1.parquet 2>&1 | grep -E "^wrote|Error" | cut -c1-120; done
fi
$SQ up > /dev/null 2>&1; say "ALL DONE"; touch $D/UNATTENDED_DONE
