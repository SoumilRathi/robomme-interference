"""CPU-only: calibrate absolute boundary threshold, build indices, gate on purity+coverage."""
import argparse, json, glob
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dumps-root", type=Path, required=True, help="root holding <family>_k<k>/<episode>/ dumps")
ap.add_argument("--index-dir", type=Path, required=True, help="output dir for per-cell index jsonl")
ap.add_argument("--gate-report", type=Path, required=True, help="file to append purity/coverage lines to")
ap.add_argument("--calibrate-only", action="store_true")
ap.add_argument("--threshold", type=float, default=None)
args = ap.parse_args()

def adjacent(emb):
    return np.sum(emb[:-1].astype(np.float32) * emb[1:].astype(np.float32), axis=1)

# ---- calibration over all episodes with embeddings ----
true_cut_sims, lesson_min_sims = [], []
for man_path in sorted(glob.glob(str(args.dumps_root / "*_k*" / "*" / "manifest.json"))):
    ep_dir = Path(man_path).parent
    if not (ep_dir / "emb.npz").exists():
        continue
    man = json.loads(Path(man_path).read_text())
    emb = np.load(ep_dir / "emb.npz")["emb"]
    adj = adjacent(emb)
    L0, L1 = man["lesson_span"]
    if L1 - 1 < len(adj):
        # true boundaries: lesson end + every filler edge
        cuts = [L1] + [s["start"] for s in man["filler_sessions"][1:]]
        for c in cuts:
            if 0 < c <= len(adj):
                true_cut_sims.append(float(adj[c - 1]))
    if L1 - 1 > 2:
        lesson_min_sims.append(float(adj[:L1 - 1].min()))

tc = np.array(true_cut_sims); lm = np.array(lesson_min_sims)
print(f"calibration: {len(tc)} true cuts, {len(lm)} lesson-min samples")
if len(tc):
    print(f"  true-cut sims:   p50={np.median(tc):.3f} p95={np.percentile(tc,95):.3f} max={tc.max():.3f}")
if len(lm):
    print(f"  lesson-min sims: p50={np.median(lm):.3f} p05={np.percentile(lm,5):.3f} min={lm.min():.3f}")
if args.threshold is None:
    thr = float((np.percentile(tc, 99) + np.percentile(lm, 1)) / 2) if len(tc) and len(lm) else 0.8
else:
    thr = args.threshold
print(f"  chosen absolute threshold: {thr:.3f}")
if args.calibrate_only:
    raise SystemExit

# ---- index + gate every cell that has full embeddings ----
args.index_dir.mkdir(exist_ok=True)
for cell_dir in sorted(args.dumps_root.glob("*_k*")):
    if not cell_dir.is_dir():
        continue
    d = cell_dir.name
    k = d.rsplit("_k", 1)[1]
    ep_dirs = sorted(p for p in cell_dir.iterdir() if (p / "manifest.json").exists())
    if len(ep_dirs) < 50 or not all((p / "emb.npz").exists() for p in ep_dirs):
        continue
    out_path = args.index_dir / f"{d}.jsonl"
    if out_path.exists():
        continue
    purs, covs, hits, rows = [], [], 0, []
    for ep_dir in ep_dirs:
        man = json.loads((ep_dir / "manifest.json").read_text())
        z = np.load(ep_dir / "emb.npz")
        emb, q = z["emb"].astype(np.float32), z["query"].astype(np.float32)
        adj = adjacent(emb)
        n = len(emb)
        cuts = sorted(set([0] + [int(i + 1) for i in np.where(adj < thr)[0]] + [n]))
        segs = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= 4]
        scores = [float(np.max(emb[lo:hi] @ q)) for lo, hi in segs]
        lo, hi = segs[int(np.argmax(scores))]
        idx = list(range(lo, hi))
        rows.append({"family": man["family"], "episode": man["episode"],
                     "condition": f"k{k}-compact-seg",
                     "retrieval_model": "siglip-so400m/visual-episodic-absthr",
                     "retrieval_method": f"visual-maxsim-absthr{thr:.2f}",
                     "candidate_frames": n, "exec_start_idx": n,
                     "segment": [lo, hi], "n_segments": len(segs),
                     "selected_ranked_indices": idx, "selected_indices": idx})
        L0, L1 = man["lesson_span"]
        inl = sum(1 for i in idx if L0 <= i < L1)
        purs.append(inl / len(idx))
        covs.append(sum(1 for i in idx if L0 <= i < L1) / (L1 - L0))
        hits += inl > 0
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mp, mc = float(np.mean(purs)), float(np.mean(covs))
    verdict = "PASS" if (mp >= 0.9 and mc >= 0.9 and hits >= int(0.9 * len(purs))) else "HOLD"
    line = (f"{d} | k{k}-compact-seg: {len(rows)} rows | purity {mp:.3f} | coverage {mc:.3f} "
            f"(nonzero {hits}/{len(purs)}) | {verdict}")
    with args.gate_report.open("a") as g:
        g.write(line + "\n")
    print("[gate]", line, flush=True)
