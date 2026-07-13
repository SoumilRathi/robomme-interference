"""Embed all history frames + query frame per episode; save fp16 npz. One-time GPU pass."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dump-dirs", type=Path, nargs="+", required=True)
ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args()

import open_clip
model, _, pre = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
model = model.to("cuda").eval()

def embed(frames):
    out = []
    with torch.inference_mode():
        for s in range(0, len(frames), args.batch):
            imgs = torch.stack([pre(Image.fromarray(f)) for f in frames[s:s+args.batch]]).to("cuda")
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.half().cpu().numpy())
    return np.concatenate(out, axis=0)

for dump_dir in args.dump_dirs:
    for ep_dir in sorted(p for p in dump_dir.iterdir() if (p / "manifest.json").exists()):
        out = ep_dir / "emb.npz"
        if out.exists():
            continue
        man = json.loads((ep_dir / "manifest.json").read_text())
        frames = np.load(ep_dir / "frames.npz")["frames"]
        emb = embed(frames[:man["exec_start_idx"]])
        qemb = embed(frames[man["query_index"]:man["query_index"]+1])
        np.savez(out, emb=emb, query=qemb[0])
    print(f"[embed] {dump_dir.name} done", flush=True)
print("[embed] ALL_DONE", flush=True)
