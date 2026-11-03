#!/bin/bash
# Fetch every 100BT file named in tasks/edu_100bt_files.json, keep .uk text, then label what the main run hasn't. Rerun to resume.
set -u; cd "$(dirname "$0")/.."; B=data/edu_100bt/shards; D=data/edu_v0; PY=${PY:-python3}; mkdir -p $B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEACHER_NAME="Qwen3.8-27B official unsloth IQ4_XS"
for s in $(python3 -c "import json;print(' '.join(json.load(open('tasks/edu_100bt_files.json'))))"); do
  n=$(basename $s .parquet); short=${n%_00000}; [ -f $B/$n.parquet ] && continue; [ "$short" != "$n" ] && [ -f $B/$short.parquet ] && continue
  for a in 1 2 3 4 5; do $PY distill/extract_shard.py $s $B/$n.parquet 2>&1 | grep -E '^\{' && break; echo "$s attempt $a failed"; sleep 120; done
done
have=$(ls $B | grep -vE 'v2|tmp|part|DONE' | grep -c parquet); echo "have $have of 140"; [ "$have" = 140 ] && touch $B/EXTRACT_DONE
until [ -d $D/student_v2 ] && [ -f data/edu_100bt/OFFICIAL_DONE ]; do sleep 120; done
for f in $B/*.parquet; do case $f in *.v2.parquet) continue;; esac; o=${f%.parquet}.v2.parquet; [ -f "$o" ] || $PY distill/apply_student.py "$f" $D/student_v2 "$o" 2>&1 | grep -E "^wrote|Error" | cut -c1-110; done
echo "labelled $(ls $B | grep -c 'v2.parquet') shards"; touch data/edu_100bt/ALL_DONE
