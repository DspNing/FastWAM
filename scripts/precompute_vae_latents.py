#!/usr/bin/env bash
# Precompute and cache Wan2.1 VAE latents for the LIBERO video dataset.
#
# This removes the CPU mp4-decode + VAE-encode cost from the training loop, which is the
# main training bottleneck (pyav decode ~2.1s/sample). After caching, the dataset reads a
# ~110KB latent tensor per sample instead of decoding 66 mp4 frames + running the 127M VAE.
#
# Usage (single GPU):
#   CUDA_VISIBLE_DEVICES=0 python scripts/precompute_vae_latents.py \
#       --task libero_uncond_2cam224_1e-4 --backbone wan21 \
#       --output-dir data/vae_latents_cache/libero_wan21_112x224
#
# Multi-GPU (shard by rank, run one process per GPU in parallel):
#   for g in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$g python scripts/precompute_vae_latents.py \
#       --task libero_uncond_2cam224_1e-4 --backbone wan21 \
#       --output-dir data/vae_latents_cache/libero_wan21_112x224 \
#       --num-shards 4 --shard $g &
#   done; wait
#
# Output: one .pt file per sample, named latent_{idx:07d}.pt, each containing
#   {"latent": [1, z_dim, T_lat, H_lat, W_lat] bf16, "image_is_pad": [T_video] bool}
# Total size ~31 GB (bf16) for 277713 samples at 112x224 (wan21 z16, 8x spatial).

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Make `src` importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components  # noqa: E402


def _load_vae(backbone: str, device: str, torch_dtype: torch.dtype):
    """Load only the VAE for the given backbone (DiT/text-encoder skipped)."""
    # A minimal dit_config is still required by the loader; we pass wan21/wan22 preset values
    # but skip_dit_load_from_pretrain=True so no DiT weights are downloaded/loaded.
    if backbone == "wan21":
        dit_config = dict(
            has_image_input=False, patch_size=[1, 2, 2], in_dim=16, hidden_dim=1536,
            ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12,
            attn_head_dim=128, num_layers=30, eps=1e-6, seperated_timestep=True,
            require_clip_embedding=False, require_vae_embedding=False,
            fuse_vae_embedding_in_latents=True, video_attention_mask_mode="first_frame_causal",
            action_conditioned=False, action_group_causal_mask_mode="group_diagonal",
        )
    elif backbone == "wan22":
        dit_config = dict(
            has_image_input=False, patch_size=[1, 2, 2], in_dim=48, hidden_dim=3072,
            ffn_dim=14336, freq_dim=256, text_dim=4096, out_dim=48, num_heads=24,
            attn_head_dim=128, num_layers=30, eps=1e-6, seperated_timestep=True,
            require_clip_embedding=False, require_vae_embedding=False,
            fuse_vae_embedding_in_latents=True, video_attention_mask_mode="first_frame_causal",
            action_conditioned=False, action_group_causal_mask_mode="group_diagonal",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    components = load_wan22_ti2v_5b_components(
        device=device,
        torch_dtype=torch_dtype,
        model_id="Wan-AI/Wan2.2-TI2V-5B",  # ignored for VAE; preset overrides per backbone
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
        dit_config=dit_config,
        skip_dit_load_from_pretrain=True,  # do not load DiT weights
        load_text_encoder=False,
        backbone=backbone,
    )
    vae = components.vae.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)
    return vae


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Precompute Wan2.1 VAE latents for LIBERO video dataset.")
    ap.add_argument("--task", required=True, help="Task config name under configs/task/ (without .yaml)")
    ap.add_argument("--backbone", default="wan21", choices=["wan21", "wan22"], help="VAE backbone preset")
    ap.add_argument("--output-dir", required=True, help="Directory to store cached latent .pt files")
    ap.add_argument("--device", default="cuda", help="Device for VAE encode")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards (for multi-GPU)")
    ap.add_argument("--shard", type=int, default=0, help="This process's shard id (0..num-shards-1)")
    ap.add_argument("--start-idx", type=int, default=0, help="Start sample index (for resuming)")
    ap.add_argument("--end-idx", type=int, default=-1, help="End sample index (exclusive); -1 = all")
    ap.add_argument("--overwrite", action="store_true", help="Re-encode even if cache file exists")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers for parallel mp4 decode (CPU-bound). More workers = faster decode, but watch CPU load.")
    ap.add_argument("--pretrained-norm-stats", default=None,
                    help="Path to dataset_stats.json to reuse (skip the ~40s stats scan). "
                         "If None and --stats-output is set, computes once and saves there.")
    ap.add_argument("--stats-output", default=None,
                    help="Where to save computed dataset_stats.json (so other shards can reuse via --pretrained-norm-stats).")
    args = ap.parse_args()

    # Validate shard range: shard id must be in [0, num_shards). (Common mistake: passing the
    # GPU id as --shard, e.g. --num-shards 4 --shard 4, which makes every shard process 0 samples.)
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError(
            f"--shard must be in [0, --num-shards={args.num_shards}), got --shard={args.shard}. "
            f"Pass the shard index (0..{args.num_shards-1}), NOT the GPU id."
        )

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset config the SAME way training does (hydra compose + instantiate), so the
    # decoded/normalized video is byte-identical to what training's VAE would see.
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}", f"model.backbone={args.backbone}"])
    OmegaConf.resolve(cfg)
    cfg.data.train.is_training_set = True
    # Reuse dataset stats across shards to avoid every shard re-scanning the dataset (~40s each).
    if args.pretrained_norm_stats:
        OmegaConf.set_struct(cfg.data.train, False)
        cfg.data.train.pretrained_norm_stats = args.pretrained_norm_stats
        OmegaConf.set_struct(cfg.data.train, True)
        print(f"[INFO] Reusing dataset stats from {args.pretrained_norm_stats}")

    print(f"[INFO] Building dataset (task={args.task}, backbone={args.backbone})...")
    dataset = instantiate(cfg.data.train)
    total = len(dataset)

    # If asked, persist the computed stats so sibling shards can reuse them.
    if args.stats_output and not args.pretrained_norm_stats:
        from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
        from fastwam.utils import misc
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        # The dataset stored stats at work_dir/dataset_stats.json during construction; copy it out.
        work_dir = Path(misc.get_work_dir())
        src = work_dir / "dataset_stats.json"
        if src.exists():
            import shutil
            shutil.copy(src, stats_path)
            print(f"[INFO] Saved dataset stats to {stats_path} (reuse with --pretrained-norm-stats)")
    end_idx = total if args.end_idx < 0 else args.end_idx
    print(f"[INFO] Dataset size: {total}. Encoding indices [{args.start_idx}, {end_idx}) shard {args.shard}/{args.num_shards}.")

    # Load VAE for the chosen backbone.
    print(f"[INFO] Loading VAE (backbone={args.backbone}, dtype={torch_dtype})...")
    vae = _load_vae(args.backbone, device, torch_dtype)
    print(f"[INFO] VAE z_dim={vae.z_dim}, upsampling_factor={vae.upsampling_factor}")

    # Shard the index range across processes.
    indices = list(range(args.start_idx, end_idx))
    my_indices = [idx for i, idx in enumerate(indices) if i % args.num_shards == args.shard]
    print(f"[INFO] This shard processes {len(my_indices)} samples.")

    # Filter out already-cached indices (resume support) before launching workers, so workers
    # don't waste CPU decoding samples we already have.
    todo_indices = [idx for idx in my_indices if args.overwrite or not (output_dir / f"latent_{idx:07d}.pt").exists()]
    n_already = len(my_indices) - len(todo_indices)
    print(f"[INFO] Already cached: {n_already}, to encode: {len(todo_indices)}")

    # Use a DataLoader with multiple workers to parallelize the CPU-bound mp4 decode + resize.
    # The main process only does VAE encode + save, so the GPU is fed by N decode workers.
    from torch.utils.data import DataLoader, Subset

    class _IdxDataset:
        """Wrap the RobotVideoDataset so __getitem__ returns (idx, sample) for our todo list."""
        def __init__(self, base, idx_list):
            self.base = base
            self.idx_list = idx_list
        def __len__(self):
            return len(self.idx_list)
        def __getitem__(self, i):
            idx = self.idx_list[i]
            sample = self.base[idx]
            return idx, sample

    loader = DataLoader(
        _IdxDataset(dataset, todo_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=lambda batch: batch[0],  # keep as (idx, sample), no stacking
    )

    t0 = time.time()
    n_done = 0
    n_skipped = n_already
    errors = []
    for idx, sample in loader:
        out_path = output_dir / f"latent_{idx:07d}.pt"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue
        try:
            video = sample["video"]  # [C, T_video, H, W], range [-1, 1]
            image_is_pad = sample.get("image_is_pad", None)
            # VAE.encode expects a list of [C,T,H,W] tensors; returns [B, z_dim, T_lat, H_lat, W_lat].
            video = video.to(device=device, dtype=torch_dtype).unsqueeze(0)  # [1, C, T, H, W]
            latent = vae.encode([video[0]], device=device, tiled=False)  # [1, z_dim, T_lat, H_lat, W_lat]
            latent = latent.to("cpu", dtype=torch_dtype)
            payload = {"latent": latent}
            if image_is_pad is not None:
                payload["image_is_pad"] = image_is_pad.to("cpu")
            torch.save(payload, out_path)
            n_done += 1
        except Exception as e:
            errors.append((int(idx), str(e)))
            print(f"[ERROR] idx={int(idx)}: {e}")

        if (n_done + n_skipped) % 100 == 0:
            elapsed = time.time() - t0
            rate = (n_done + n_skipped) / max(elapsed, 1e-6)
            print(f"[progress] shard={args.shard} done={n_done} skipped={n_skipped} "
                  f"errors={len(errors)} rate={rate:.2f} samples/s elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"[DONE] shard={args.shard} encoded={n_done} skipped={n_skipped} errors={len(errors)} "
          f"in {elapsed:.0f}s ({(n_done+n_skipped)/max(elapsed,1e-6):.2f} samples/s)")
    if errors:
        print(f"[WARN] {len(errors)} errors. First 5: {errors[:5]}")
        # Write error log for re-running just the failed indices.
        err_path = output_dir / f"errors_shard{args.shard}.txt"
        with open(err_path, "w") as f:
            for idx, msg in errors:
                f.write(f"{idx}\t{msg}\n")
        print(f"[WARN] Error indices written to {err_path}")


if __name__ == "__main__":
    main()
