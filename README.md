# FineWeb-UK

Tools and task sets for testing whether UK-tuned filtering of web text trains better British language models. This is an early pilot.

**This repo contains no web page text.** It has code, pinned source versions, and task lists of document IDs and hashes. Scripts fetch the text from the source dataset onto your own machine.

## Label documents

You need Python 3.10+ and about 2 GB of disk space.

```bash
pip install -r requirements.txt
python fetch_docs.py                                  # downloads the shard, verifies 300 documents
python label_tool.py --annotator your-pseudonym       # opens the tool in your browser
```

Read [LABELLING.md](LABELLING.md) first, then [CONTRIBUTING.md](CONTRIBUTING.md) to submit your labels. Expect about an hour per 100 documents. Pages come from the open web, so some may contain adult or upsetting material; you can skip any document.

## Check agreement

```bash
python score.py
```

This reports Cohen's kappa for each decision. A decision below 0.6 needs its definition rewritten. Add `--model data/sample_v0/d1.jsonl` after rebuilding the sample to score the model's accuracy and calibration against the human labels.

## Task sets

| File | Documents | Purpose |
|---|---|---|
| `tasks/shared_300.jsonl` | 300 (90 random, 210 rare-label) | Three annotators label all of them, to test the definitions |
| `tasks/gold_1000.jsonl` | 1,000 (700 random, 300 rare-label) | Measures model accuracy and calibration |

The sets don't overlap. `pipeline/make_tasks.py` drew them with a fixed seed from sample v0.

## Rebuild sample v0

```bash
LLAMA_SERVER=/path/to/llama-server pipeline/run_sample.sh
```

This needs a CUDA GPU and takes about an hour on an RTX 4090. It filters one FineWeb shard to `.uk` hosts, scores each document with the FineWeb-Edu classifier and d1-3B, and writes `data/sample_v0/fineweb_uk_sample_v0.parquet`. Versions are pinned in `pins.env`. The script checks the shard's hash and the extracted documents against `tasks/sample_v0.fingerprint`. Labels from the models may differ slightly between machines.

Sample v0 has 51,158 documents and 30.4M tokens. Its labels haven't been validated by people yet.

## Licences and page rights

- **Code:** Apache 2.0, see [LICENSE](LICENSE).
- **Labels in `labels/`:** CC BY 4.0.
- **Source data:** [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) is released under ODC-By 1.0 and is also subject to Common Crawl's terms. Those terms don't clear the rights in each underlying page, so please don't republish document text with your labels.
- **d1-3B:** the [LFM Open License](https://huggingface.co/LiquidAI/d1-3B), which doesn't license commercial use by organisations above its revenue threshold. Labelling by hand doesn't use this model.
