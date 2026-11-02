#!/bin/bash
# Rebuild sample v0 from pinned sources. Needs a CUDA GPU and llama-server at LLAMA_CPP_COMMIT or later.
# Usage: LLAMA_SERVER=/path/to/llama-server pipeline/run_sample.sh   (rerun to resume)
set -eu; cd "$(dirname "$0")/.."; . ./pins.env
PY=${PY:-python3}; LLAMA_SERVER=${LLAMA_SERVER:-llama-server}; D=data/sample_v0; mkdir -p "$D"
RAW=$($PY -c "from huggingface_hub import hf_hub_download as h;print(h('$FINEWEB_REPO','$SHARD',repo_type='dataset',revision='$FINEWEB_REV'))")
echo "$SHARD_SHA256  $RAW" | sha256sum -c -
[ -f $D/docs.parquet ] || $PY pipeline/extract.py "$RAW" $D/docs.parquet "$SHARD"
$PY pipeline/verify.py $D/docs.parquet
[ -f $D/edu.jsonl ] || $PY pipeline/label_edu.py $D/docs.parquet $D/edu.jsonl "$EDU_REV"
G=$($PY -c "from huggingface_hub import hf_hub_download as h;print(h('$D1_GGUF_REPO','$D1_GGUF_FILE',revision='$D1_GGUF_REV'))")
"$LLAMA_SERVER" -m "$G" -ngl 99 -c 32768 --parallel 8 --port 8091 --host 127.0.0.1 > $D/llama.log 2>&1 & P=$!
trap 'kill $P' EXIT
for i in $(seq 1 60); do curl -s -m 2 localhost:8091/health | grep -q ok && break; sleep 1; done
$PY pipeline/label_d1.py $D/docs.parquet $D/d1.jsonl
$PY pipeline/merge.py $D
