#!/usr/bin/env python
"""Export a trained student to a Hugging Face folder: safetensors weights (which can't run code), tokenizer and label config."""
import json, os, shutil, sys, torch
from safetensors.torch import save_file
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
ck = torch.load(os.path.join(src, "student.pt"), map_location="cpu", weights_only=True)
save_file({k: v.clone() for k, v in ck["state"].items()}, os.path.join(out, "model.safetensors"))
json.dump({"base": ck["base"], "heads": ck["heads"], "max_length": 512, "pooling": "mean"}, open(os.path.join(out, "fineweb_uk_student.json"), "w"), indent=1)
for f in os.listdir(src):
    if f.startswith("tokenizer") or f.endswith(".jinja") or f == "special_tokens_map.json": shutil.copy(os.path.join(src, f), out)
if os.path.exists(os.path.join(src, "metrics.json")): shutil.copy(os.path.join(src, "metrics.json"), out)
print("exported", sorted(os.listdir(out)))
