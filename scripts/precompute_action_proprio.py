#!/usr/bin/env python3
"""Precompute normalized action/proprio cache for the LIBERO dataset (FAST batch version).

Instead of calling ds[idx] 277k times (each ~519ms due to slow hf_dataset.select), this:
  1. Batch-extracts raw action/state tensors from each suite's hf_dataset (~2s/suite)
  2. Vectorized slicing with delta_indices + episode boundary clip (seconds)
  3. Reuses the processor's transform/normalizer/merger (identical to training)
  4. Saves normalized action/proprio/action_is_pad to a single file (~0.5GB)

The normalization is byte-identical to ds[idx] because we reuse the SAME processor methods
(action_state_transform / normalizer.forward / action_state_merger.forward), only replacing
the slow hf_dataset.select with vectorized tensor indexing.

Output: data/action_proprio_cache/libero.pt
  {idx: {"action": [32,7], "proprio": [32,8], "action_is_pad": [32]}}

Usage:
  python scripts/precompute_action_proprio.py --task libero_uncond_2cam224_1e-4 --backbone wan21
  # then verify (compares 50 random idx vs ds[idx]):
  python scripts/precompute_action_proprio.py --verify data/action_proprio_cache/libero.pt
"""
import argparse
import sys
import time
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_dataset_and_processor(task: str, backbone: str = "wan21"):
    """Build dataset (skip_images to avoid mp4 decode) and return the processor for reuse."""
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}", f"model.backbone={backbone}"])
    OmegaConf.resolve(cfg)
    cfg.data.train.is_training_set = True
    OmegaConf.set_struct(cfg.data.train, False)
    cfg.data.train.pretrained_norm_stats = "data/vae_latents_cache/dataset_stats.json"
    cfg.data.train.vae_latent_cache_dir = None  # no latent pre-load; we set skip_images manually
    ds = instantiate(cfg.data.train)
    ds.lerobot_dataset.skip_images = True  # don't decode video; we only need action/proprio
    # The processor is set on BaseLerobotDataset (ds.lerobot_dataset), not on sub-datasets.
    processor = ds.lerobot_dataset.processor
    return ds, processor, cfg


def extract_suite_data(ds):
    """Batch-extract raw action/state + episode boundaries + task from each suite.

    Returns a list (per suite) of dicts with:
      action_tensor: [N_suite, action_dim]  (raw, unnormalized)
      state_tensor:  [N_suite, state_dim]   (raw, unnormalized)
      tasks: list[str]  (task/instruction string per frame)
      ep_from: list[int]  (episode start frame indices, global within suite)
      ep_to:   list[int]  (episode end frame indices)
      suite_start: int    (this suite's start idx in the global MultiLeRobotDataset)
      suite_len: int       (number of frames in this suite)
    """
    multi = ds.lerobot_dataset.multi_dataset
    suites = []
    suite_start = 0
    for d in multi._datasets:
        hf = d.hf_dataset
        suite_len = len(hf)
        # Batch extract (fast: ~2s per suite)
        action_list = hf["action"]  # list of tensors [action_dim]
        state_list = hf["observation.state"]  # list of tensors [state_dim]
        action_tensor = torch.stack([torch.as_tensor(a) for a in action_list])  # [N, 7]
        state_tensor = torch.stack([torch.as_tensor(s) for s in state_list])    # [N, 8]

        ep_from = d.episode_data_index["from"].tolist()  # episode start frames
        ep_to = d.episode_data_index["to"].tolist()      # episode end frames

        # Extract task/instruction per frame: task_index -> task string via meta.tasks
        task_indices = hf["task_index"]  # list of tensors
        tasks = [d.meta.tasks[int(ti)] for ti in task_indices]  # list[str]

        suites.append({
            "action_tensor": action_tensor,
            "state_tensor": state_tensor,
            "tasks": tasks,
            "ep_from": ep_from,
            "ep_to": ep_to,
            "suite_start": suite_start,
            "suite_len": suite_len,
            "delta_action": d.delta_indices["action"],      # [0,1,...,31]
            "delta_state": d.delta_indices["observation.state"],  # [0,...,32]
            "ep_idx_of_frame": _build_frame_to_episode(ep_from, ep_to, suite_len),
        })
        suite_start += suite_len
    return suites


def _build_frame_to_episode(ep_from, ep_to, n_frames):
    """Return array where frame_to_ep[f] = episode index containing frame f."""
    frame_to_ep = np.zeros(n_frames, dtype=np.int64)
    for ep_i, (s, e) in enumerate(zip(ep_from, ep_to)):
        frame_to_ep[s:e] = ep_i
    return frame_to_ep


def vectorized_slice(suites):
    """For each global idx, slice action/state with delta_indices + episode boundary clip.

    Returns:
      all_action: [N_total, 32, 7]   (raw, clipped to episode)
      all_state:  [N_total, 33, 8]  (raw, clipped to episode)
      all_action_is_pad: [N_total, 32] bool
    """
    all_action = []
    all_state = []
    all_action_is_pad = []
    for su in suites:
        N = su["suite_len"]
        action_t = su["action_tensor"]      # [N, 7]
        state_t = su["state_tensor"]        # [N, 8]
        delta_a = su["delta_action"]        # [0..31], len 32
        delta_s = su["delta_state"]         # [0..32], len 33
        ep_from = su["ep_from"]
        ep_to = su["ep_to"]
        frame_to_ep = su["ep_idx_of_frame"]

        # For each frame f, its episode boundaries:
        ep_for_frame = frame_to_ep  # [N]
        ep_start = np.array([ep_from[e] for e in ep_for_frame])  # [N]
        ep_end = np.array([ep_to[e] for e in ep_for_frame])      # [N]

        # Build index arrays for action: for each frame f, indices = clip(f + delta_a, ep_start, ep_end-1)
        f_idx = np.arange(N)  # [N]
        # action: [N, 32]
        a_query = f_idx[:, None] + np.array(delta_a)[None, :]  # [N, 32]
        a_query_clipped = np.maximum(ep_start[:, None], np.minimum(ep_end[:, None] - 1, a_query))
        a_is_pad = (a_query < ep_start[:, None]) | (a_query >= ep_end[:, None])  # [N, 32]

        # state: [N, 33]
        s_query = f_idx[:, None] + np.array(delta_s)[None, :]  # [N, 33]
        s_query_clipped = np.maximum(ep_start[:, None], np.minimum(ep_end[:, None] - 1, s_query))

        # Gather from tensors
        a_slices = action_t[a_query_clipped]  # [N, 32, 7]
        s_slices = state_t[s_query_clipped]   # [N, 33, 8]

        all_action.append(a_slices)
        all_state.append(s_slices)
        all_action_is_pad.append(torch.from_numpy(a_is_pad))

    all_action = torch.cat(all_action, dim=0)          # [N_total, 32, 7]
    all_state = torch.cat(all_state, dim=0)            # [N_total, 33, 8]
    all_action_is_pad = torch.cat(all_action_is_pad, dim=0)  # [N_total, 32]
    return all_action, all_state, all_action_is_pad


def normalize_batch(all_action, all_state, all_action_is_pad, processor, batch_size=4096):
    """Apply processor's normalization (action_state_transform + normalizer + merger) per-sample.

    The merger's _concat/_pad only support 2D ([T, dim]), so we process one sample at a time
    (each [32,7] / [33,8]). Still fast because normalization is a linear transform.
    Reuses the EXACT same processor methods as training, so output is identical.
    """
    N = all_action.shape[0]
    out_action = []
    out_proprio = []
    out_action_is_pad = []
    t0 = time.time()
    for idx in range(N):
        a = all_action[idx]       # [32, 7]
        s = all_state[idx]        # [33, 8]
        pad = all_action_is_pad[idx]  # [32]

        # Build data dict in the format processor.preprocess expects (per-key dict, 2D tensors).
        data = {
            "action": {"default": a.clone()},   # [32,7]
            "state": {"default": s.clone()},    # [33,8]
            "action_is_pad": pad,               # [32]
        }
        # Apply delta_action_dim_mask (zero padded delta actions) — same as preprocess L252-260
        if processor.delta_action_dim_mask is not None and bool(pad.any()):
            for key, dim_mask in processor.delta_action_dim_mask.items():
                cur_action = data["action"][key]  # [32,7]
                cur_dim_mask = dim_mask.to(device=cur_action.device)  # [7]
                pad_delta_mask = pad.unsqueeze(-1) & cur_dim_mask.unsqueeze(0)  # [32,7]
                cur_action[pad_delta_mask] = 0.0

        # Reuse processor's transform/normalizer/merger (identical to training)
        data = processor.action_state_transform(data)
        data = processor.normalizer.forward(data)
        data = processor.action_state_merger.forward(data)

        out_action.append(data["action"])          # [32,7]
        out_proprio.append(data["state"][:-1])     # [32,8] (proprio = state[:-1], matching _get)
        out_action_is_pad.append(data["action_is_pad"])  # [32]

        if (idx + 1) % 20000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (N - idx - 1) / rate
            print(f"  normalized {idx+1}/{N} rate={rate:.0f}/s eta={eta:.0f}s")

    out_action = torch.stack(out_action, dim=0)
    out_proprio = torch.stack(out_proprio, dim=0)
    out_action_is_pad = torch.stack(out_action_is_pad, dim=0)
    return out_action, out_proprio, out_action_is_pad


def main():
    ap = argparse.ArgumentParser(description="Precompute normalized action/proprio (batch, fast).")
    ap.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    ap.add_argument("--backbone", default="wan21")
    ap.add_argument("--output", default="data/action_proprio_cache/libero.pt")
    ap.add_argument("--verify", default=None, help="Verify cached file vs ds[idx] (path to cached .pt)")
    ap.add_argument("--n-verify", type=int, default=50, help="Number of random idx to verify")
    args = ap.parse_args()

    if args.verify:
        return verify(args.verify, args.task, args.backbone, args.n_verify)

    print(f"=== Precompute action/proprio cache (batch) ===")
    print(f"Building dataset (skip_images, no latent pre-load)...")
    t0 = time.time()
    ds, processor, _ = build_dataset_and_processor(args.task, args.backbone)
    print(f"Dataset built in {time.time()-t0:.1f}s")

    print(f"\nExtracting raw action/state from hf_dataset (batch)...")
    t0 = time.time()
    suites = extract_suite_data(ds)
    total = sum(s["suite_len"] for s in suites)
    print(f"  {len(suites)} suites, {total} total frames, extracted in {time.time()-t0:.1f}s")

    print(f"\nVectorized slicing (delta_indices + episode clip)...")
    t0 = time.time()
    all_action, all_state, all_action_is_pad = vectorized_slice(suites)
    print(f"  action: {tuple(all_action.shape)}, state: {tuple(all_state.shape)}, "
          f"action_is_pad: {tuple(all_action_is_pad.shape)}, in {time.time()-t0:.1f}s")

    print(f"\nNormalizing (reusing processor methods)...")
    t0 = time.time()
    norm_action, norm_proprio, norm_action_is_pad = normalize_batch(
        all_action, all_state, all_action_is_pad, processor)
    print(f"  normalized action: {tuple(norm_action.shape)}, proprio: {tuple(norm_proprio.shape)}, "
          f"in {time.time()-t0:.1f}s")

    # Build idx->data dict and save (include task/instruction for each idx)
    print(f"\nSaving to {args.output}...")
    # Collect tasks per global idx
    all_tasks = []
    for su in suites:
        all_tasks.extend(su["tasks"])
    cache = {}
    for idx in range(total):
        cache[idx] = {
            "action": norm_action[idx],
            "proprio": norm_proprio[idx],
            "action_is_pad": norm_action_is_pad[idx],
            "task": all_tasks[idx],  # instruction string (for context lookup)
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    sz = Path(args.output).stat().st_size
    print(f"Saved {len(cache)} samples, {sz/1e6:.0f} MB -> {args.output}")
    print(f"\n=== DONE. Verify with: python {__file__} --verify {args.output} ===")


def verify(cache_path, task, backbone, n_verify):
    """Compare cached action/proprio vs ds[idx] for n_verify random indices."""
    print(f"=== Verify: {cache_path} vs ds[idx] ({n_verify} random idx) ===")
    print("Loading cache...")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"Cache: {len(cache)} samples")

    print("Building dataset (with processor)...")
    ds, _, _ = build_dataset_and_processor(task, backbone)
    # For verification, ds[idx] must go through the FULL original path (decode video + processor)
    # to produce the reference action/proprio. So disable skip_images and latent cache.
    ds.lerobot_dataset.skip_images = False
    ds.vae_latent_cache_dir = None
    ds._latent_cache = None

    import random
    random.seed(42)
    indices = random.sample(range(len(cache)), min(n_verify, len(cache)))

    max_diff_action = 0.0
    max_diff_proprio = 0.0
    max_diff_pad = 0
    mismatches = 0
    for idx in indices:
        s = ds[idx]
        ref_action = s["action"]          # [32,7]
        ref_proprio = s["proprio"]        # [32,8]
        ref_pad = s["action_is_pad"]      # [32]

        cached = cache[idx]
        c_action = cached["action"]
        c_proprio = cached["proprio"]
        c_pad = cached["action_is_pad"]

        d_action = (ref_action.float() - c_action.float()).abs().max().item()
        d_proprio = (ref_proprio.float() - c_proprio.float()).abs().max().item()
        d_pad = (ref_pad != c_pad).sum().item()

        max_diff_action = max(max_diff_action, d_action)
        max_diff_proprio = max(max_diff_proprio, d_proprio)
        max_diff_pad = max(max_diff_pad, d_pad)
        if d_action > 1e-5 or d_proprio > 1e-5 or d_pad > 0:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH idx={idx}: action_diff={d_action:.6f} proprio_diff={d_proprio:.6f} pad_diff={d_pad}")

    print(f"\n=== Verify result ({n_verify} samples) ===")
    print(f"  max action diff:     {max_diff_action:.8f}")
    print(f"  max proprio diff:    {max_diff_proprio:.8f}")
    print(f"  max action_is_pad diff: {max_diff_pad}")
    print(f"  mismatches: {mismatches}/{n_verify}")
    if mismatches == 0 and max_diff_action < 1e-5 and max_diff_proprio < 1e-5:
        print("  ✓ PASS: cached action/proprio identical to ds[idx]")
    else:
        print("  ✗ FAIL: differences found, do not use cache")


if __name__ == "__main__":
    main()
