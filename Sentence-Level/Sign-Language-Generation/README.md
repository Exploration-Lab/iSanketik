# ISL Pose Generation

Text-to-pose generation for Indian Sign Language (ISL) using a two-phase
latent flow matching pipeline: a pose VAE compresses raw skeletal pose
sequences into a compact latent space and a Diffusion Transformer (DiT)
learns to generate latents conditioned on text via rectified flow matching.

## Pipeline

**Phase 1: Pose VAE**
Trains a 1D-convolutional VAE that compresses 190-dim pose sequences
(upper-body, hands, and a reduced facial landmark set) into a 128-dim
latent space at 4x temporal downsampling.

- `src/data.py` — dataset, length-aware sampler, dataloader
- `src/pose_vae.py` — encoder/decoder architecture, VAE loss
- `src/Phase 1/train_vae.py` — training entry point

**Phase 2: Flow Matching with DiT**
Trains a DiT to predict the rectified-flow velocity field over the frozen
VAE's latent space, conditioned on Flan-T5 text embeddings. A length
predictor is trained jointly to estimate output pose duration from text.

- `src/data.py`, `src/pose_vae.py` — shared with Phase 1 (VAE loaded frozen)
- `src/Phase 2/text_encoder.py` — Flan-T5 text encoder wrapper (+ null encoder for CFG)
- `src/Phase 2/dit.py` — DiT denoiser and length predictor
- `src/Phase 2/flow_matching.py` — rectified flow forward process, loss and sampler
- `src/Phase 2/train_flow.py` — training entry point

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for practical training times; both training
scripts fall back to CPU automatically if CUDA is unavailable.

## Data format

Each split (train/val/test) requires:
- A directory of `.pt` pose shards, each containing `X` (pose), `C`
  (per-joint confidence), `root`, `scale`, and `uid` tensors/lists.
- A CSV file with `uid` and `text` columns mapping each pose sample to its
  text translation.
- Optionally, a prebuilt shard index (`.pt`) to skip re-scanning shards on
  every run; one is built and cached automatically on first use if not
  provided.

## Usage

**Phase 1 — train the pose VAE:**

```bash
python src/Phase 1/train_vae.py \
  --train_shard_dir /path/to/train_shards \
  --val_shard_dir   /path/to/val_shards \
  --train_csv       /path/to/train.csv \
  --val_csv         /path/to/val.csv \
  --output_dir      ./checkpoints/vae
```

**Phase 2 — train the flow matching model** (requires a trained VAE checkpoint):

```bash
python src/Phase 2/train_flow.py \
  --train_shard_dir /path/to/train_shards \
  --val_shard_dir   /path/to/val_shards \
  --train_csv       /path/to/train.csv \
  --val_csv         /path/to/val.csv \
  --vae_ckpt        ./checkpoints/vae/best_vae.pt \
  --latent_stats_path /path/to/latent_mean_std.pt \
  --output_dir      ./checkpoints/flow
```

Run either script with `--help` to see the full list of configurable
arguments (model size, augmentation, loss weights etc.).

Both scripts log to a file under `--output_dir` and support `--resume` to
continue from the last saved checkpoint.

## Repository structure

```
isl-pose-generation/
├── README.md
├── requirements.txt
└── helpers/
    ├── data.py
    ├── pose_vae.py
    ├── text_encoder.py
    ├── dit.py
    ├── flow_matching.py
└── phase1.py
└── phase2.py
```