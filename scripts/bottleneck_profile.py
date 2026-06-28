"""Precise bottleneck breakdown for the FastWAM data pipeline.

Measures each stage of __getitem__ and DataLoader throughput to locate the real
bottleneck (parquet read / processor normalize / latent load / collate / main-process).
Run on CPU (CUDA_VISIBLE_DEVICES="") so it does not disturb training.
"""
import sys, time, os
sys.path.insert(0, "src")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader

with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=libero_uncond_2cam224_1e-4", "model.backbone=wan21"])
OmegaConf.resolve(cfg)
cfg.data.train.is_training_set = True
OmegaConf.set_struct(cfg.data.train, False)
cfg.data.train.pretrained_norm_stats = "data/vae_latents_cache/dataset_stats.json"
cfg.data.train.vae_latent_cache_dir = "data/vae_latents_cache/libero_wan21_112x224"

print("Building dataset (with in-memory latent cache)...")
t0 = time.time()
ds = instantiate(cfg.data.train)
print(f"Dataset built in {time.time()-t0:.1f}s\n")

idx = 500

# --- Stage 1: parquet read only (lerobot_dataset, skip_images) ---
ds.lerobot_dataset.skip_images = True
t0 = time.time()
for i in range(20):
    s = ds.lerobot_dataset[idx + i]
print(f"[1] parquet read (skip_images, 20 samples): {(time.time()-t0)/20*1000:.1f} ms/sample")

# --- Stage 2: processor.preprocess only (use a fresh raw sample each time) ---
t0 = time.time()
for i in range(20):
    s_raw_i = ds.lerobot_dataset[idx + i]  # fresh raw sample each time
    s_proc = proc.preprocess(s_raw_i)
print(f"[2] parquet+processor.preprocess (20x): {(time.time()-t0)/20*1000:.1f} ms/sample")

# --- Stage 2b: processor.preprocess only (reuse raw, isolate normalize) ---
s_raw = ds.lerobot_dataset[idx]
t0 = time.time()
for _ in range(20):
    s_proc = proc.preprocess(s_raw)
# (may error if processor mutates; if so, skip)
try:
    print(f"[2b] processor.preprocess reuse (20x): {(time.time()-t0)/20*1000:.1f} ms/sample")
except Exception:
    print("[2b] skipped (processor mutates input)")

# --- Stage 3: latent from in-memory cache ---
t0 = time.time()
for i in range(20):
    _ = ds._latent_cache.get(idx + i)
print(f"[3] latent from memory dict (20x): {(time.time()-t0)/20*1000:.3f} ms/sample")

# --- Stage 4: full __getitem__ ---
for i in range(3): ds[idx+i]  # warmup
t0 = time.time()
for i in range(20):
    s = ds[idx + i]
print(f"[4] full _get (20 samples): {(time.time()-t0)/20*1000:.1f} ms/sample")

# --- Stage 5: DataLoader throughput (8 workers, real parallel) ---
print("\n--- DataLoader throughput (8 workers, batch=8) ---")
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=False, prefetch_factor=4)
it = iter(loader)
# warmup
for _ in range(3): next(it)
t0 = time.time()
n = 0
for batch in it:
    n += 1
    if n >= 20: break
dt = time.time() - t0
print(f"20 batches (160 samples) in {dt:.2f}s -> {160/dt:.1f} samples/s with 8 workers")
print(f"  per-batch: {dt/20*1000:.1f} ms")

# --- Stage 6: collate cost (default_collate on a batch) ---
samples = [ds[idx+i] for i in range(8)]
from torch.utils.data._utils.collate import default_collate
t0 = time.time()
for _ in range(20):
    default_collate(samples)
print(f"\n[6] default_collate (8 samples, 20x): {(time.time()-t0)/20*1000:.1f} ms/batch")
