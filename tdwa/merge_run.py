"""Write a teaching run's final student as a plain Hugging Face model directory (for vLLM).

  python -m tdwa.merge_run runs/<name> [--out cache/merged/<name>] [--dtype float32]

The directory holds the original config/tokenizer files plus one safetensors file with every original tensor,
replaced by the run's final edited tensors (runs/<name>/final/edit.safetensors). float32 is the default so that small
edits are not rounded away (bfloat16 keeps only ~3 significant digits). Added bias vectors cannot be expressed in
the original architecture: the script refuses if a run kept a non-zero one (evaluate such runs with tdwa.evaluate).
"""
import argparse, glob, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.environ.get("TDWA_HF", "/mnt/data2/users/lang.gao/huggingface_cache")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out"); ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    run = os.path.abspath(a.run); name = os.path.basename(run.rstrip("/"))
    cfg = json.load(open(os.path.join(run, "config.json")))
    snap = glob.glob(os.path.join(HF, "models--" + cfg["student"].replace("/", "--"), "snapshots", "*"))[0]
    out = a.out or os.path.join(ROOT, "cache", "merged", name)
    edit = load_file(os.path.join(run, "final", "edit.safetensors")); edit.pop("_empty", None)
    bias = {k: v for k, v in edit.items() if k.startswith("tdwa_bias__")}
    if any(bool(v.abs().sum()) for v in bias.values()):
        sys.exit(f"{name} kept a non-zero added bias ({list(bias)}); vLLM cannot load it. Use tdwa.evaluate.")
    edit = {k: v for k, v in edit.items() if k not in bias}
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(snap):
        if not f.endswith(".safetensors") and not f.endswith(".safetensors.index.json"):
            shutil.copy(os.path.realpath(os.path.join(snap, f)), os.path.join(out, f))
    dt = getattr(torch, a.dtype); tensors = {}; used = set()
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                t = edit[k] if k in edit else h.get_tensor(k)
                used.add(k) if k in edit else None
                tensors[k] = t.to(dt).contiguous()
    missing = set(edit) - used
    if missing:
        sys.exit(f"edited tensors not found in the checkpoint: {sorted(missing)[:5]}")
    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    c = json.load(open(os.path.join(out, "config.json")))
    c["torch_dtype"] = a.dtype
    if "text_config" in c:
        c["text_config"]["torch_dtype"] = a.dtype
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(out, len(tensors), "tensors,", len(edit), "edited")


if __name__ == "__main__":
    main()
