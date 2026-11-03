#!/bin/bash
# After the Croydon teacher finishes: train Croydon, score every shard, draw the grid. Rerun to resume.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python3}; SQ=${SQ:-sq}; C=data/croydon; M=$C/croydon_v0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEACHER_NAME="Qwen3.8-27B official unsloth IQ4_XS"
say() { echo "[$(date +%H:%M)] $*"; }
while ps -eo args | grep -q '[t]eacher_croydon.py'; do sleep 60; done; say "teacher done: $(wc -l < $C/teacher.jsonl) labels"
pkill -f "llama-server.*--port 8092"; sleep 5
# plain FineWeb's Edu scores, for the grid
[ -f data/fineweb_v0/scored.parquet ] || $PY -c "
import pyarrow.parquet as pq
pq.write_table(pq.read_table('data/sample_v0/fineweb_uk_sample_v0.parquet',columns=['id','text','token_count','edu_score']),'data/fineweb_v0/scored.parquet')"
[ -f $M/model.safetensors ] || { $PY distill/train_croydon.py --out $M > $C/train.log 2>&1; say "train exit $?"; }
grep -E "^---|^speed" $C/train.log | cut -c1-220
$SQ up > /dev/null 2>&1; say "speed-qwen back up"
until [ -f data/edu_100bt/shards/EXTRACT_DONE ] || ! ps -eo args | grep -q '[e]xtract_100bt.sh'; do sleep 120; done
for f in data/fineweb_v0/scored.parquet data/edu_v0/shards/*.parquet data/edu_100bt/shards/*.parquet; do
  [[ $(basename $f) =~ ^([0-9]{3}(_[0-9]{5})?|scored)\.parquet$ ]] || continue  # source shards only, not derived files
  o=${f%.parquet}.croydon.parquet; [ -f "$o" ] || $PY distill/apply_croydon.py "$f" $M "$o" 2>&1 | grep -E "^wrote|Error" | cut -c1-120; done
$PY distill/grid.py 2>&1 | grep -v Skipping > $C/grid.txt; cat $C/grid.txt; say "CROYDON DONE"; touch $C/DONE
