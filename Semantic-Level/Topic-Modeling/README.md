# Self-Supervised Pose Embeddings for ISL Topic Modeling

Topic Modeling is implemented as a four-stage pipeline that learns self-supervised
representations of Indian Sign Language (ISL) video paragraphs from 3D pose
sequences and uses those representations to discover latent semantic
topics in the underlying signed content.

The pipeline is organized as:

```
Step 1: train_pose_encoder.py     Self-supervised MoCo pretraining of a pose encoder
Step 2: extract_embeddings.py     Video-level embedding extraction from a trained checkpoint
Step 3: checkpoint_selection.ipynb          Checkpoint / epoch selection via clustering quality metrics
Step 4: topic_modeling.ipynb      BERTopic topic discovery over the selected embeddings
```

Each stage consumes the output of the previous one. The sections below
describe the purpose, inputs, outputs, and key design decisions of each
stage.

---



## Step 1 — `train_pose_encoder.py`: Self-Supervised Pose Encoder Pretraining

**Goal:** Learn a general-purpose, fixed-size embedding for variable-length
ISL pose sequences without any text or label supervision, using a
Momentum Contrast (MoCo) objective.

### Input data
- Pre-extracted per-paragraph pose tensors (`522`-dim skeletal landmarks:
  body, hands, and face) stored as `.pt` files, indexed by a
  `metadata.csv` manifest (`uid`, `file`, `length`, ...).

### Model architecture (`PoseEncoder`)
- Linear input projection (`522 → 512`) + sinusoidal positional encoding.
- A 6-layer, 8-head **pre-norm Transformer encoder** (`d_model=512`,
  `dim_feedforward=2048`) processes the padded pose sequence.
- **Attention-based temporal pooling** (`TemporalPooling`) collapses the
  frame sequence into a single embedding, respecting a per-frame validity
  mask (with a safe fallback for degenerate all-padding sequences).
- An MLP output head projects to a `256`-dim embedding
  (`output_embed_dim`), and a lightweight linear `recon_head` decodes
  per-frame pose reconstructions from the same hidden states.

### Self-supervised objective
Two losses are combined:

1. **MoCo contrastive loss (`MoCoContrastive`)** — the primary objective.
   - A query encoder/projection head and a momentum-averaged key
     encoder/projection head (EMA, `momentum=0.999`) are maintained.
   - Each pose sequence is augmented **twice independently** to produce a
     query view and a key view (see augmentations below); their
     normalized projected embeddings are pulled together via an InfoNCE
     loss (`temperature=0.07`) against a queue of `4096` negative
     embeddings from previous batches.
2. **Auxiliary frame-level reconstruction loss** — an MSE loss between the
   `recon_head` output and the original (un-augmented) pose sequence,
   computed only over valid (non-padded) entries. This is weighted at
   `0.05` relative to the contrastive loss and acts as a regularizer that
   discourages the encoder from discarding fine-grained pose information.

### Augmentations (`ParagraphDataset._augment`)
Each view is produced by a randomized stack of:
- Temporal cropping (40–100% of sequence length)
- Speed perturbation (0.75×–1.25× resampling)
- Frame dropout (keep 70–100% of frames)
- Temporal reversal (20% probability)
- Regional keypoint masking — zeroing out body, hand, or face keypoints
  (50% probability), forcing the encoder to be robust to missing
  modalities.

### Training loop (`Trainer`)
- AdamW optimizer, linear warmup (5 epochs) + cosine decay schedule, up to
  100 epochs, mixed-precision training, gradient clipping.
- Validation every epoch: the negative queue is **rebuilt from scratch in
  eval mode** before scoring, to avoid a train/eval BatchNorm mismatch
  between the validation query embeddings and the queue's key embeddings.
- Early stopping on validation contrastive loss (`patience=15`).
- Full checkpointing (model weights, MoCo queue + pointer, optimizer,
  scheduler, and bookkeeping state) enabling exact resumption via
  `cfg.resume_from_checkpoint = "latest"` or an explicit path. On resume,
  the negative queue is explicitly re-warmed (`_warm_queue`) since a
  stale queue would otherwise corrupt early post-resume loss estimates.
- `checkpoint_epoch_{NNN}.pt` is saved every epoch; `best_model.pt` /
  `best_backbone.pt` track the best validation contrastive loss.

### Output
- `checkpoints/checkpoint_epoch_{NNN}.pt` — full training checkpoints for
  every epoch (used by Step 3 for cross-epoch comparison).
- `checkpoints/best_backbone.pt` — encoder-only weights for the
  best-validation-loss epoch (used by default in Step 2).

---

## Step 2 — `extract_embeddings.py`: Video-Level Embedding Extraction

**Goal:** Run a trained (frozen) `PoseEncoder` backbone over a directory of
video-level pose tensors and produce a single embedding matrix, row-aligned
with per-video metadata, ready for downstream clustering / topic modeling.

This script is intentionally **self-contained** — it redefines the model
architecture rather than importing it from Step 1 — so that embedding
extraction has no dependency on the training codebase and can be run or
shared independently (e.g. for reproducibility alongside the paper).

### Input data
- A backbone checkpoint: either a raw `state_dict` (`best_backbone.pt`) or
  a full training checkpoint (`checkpoint_epoch_NNN.pt`, from which the
  `encoder_q_state` sub-dictionary is automatically extracted).
- A directory of per-video `.pt` files (`VideoLevelPoseDataset`), each
  containing a padded pose tensor `X`, validity mask `M`, sequence length
  `L`, `video_id`, source FPS, and segment count.

### Process
- Loads the encoder in eval mode and disables gradient computation.
- Batches videos (default `batch_size=32`), pads to the batch's max
  length, and derives a frame-level validity mask.
- Runs the forward pass under mixed precision to produce a
  `[N, 256]` embedding matrix.

### Output (written to `--output_dir`)
- `embeddings.npy` — `[N, 256]` float32 embedding matrix.
- `metadata.csv` — row-aligned per-video metadata (`video_id`, `length`,
  `src_fps`, `num_segments`, `file`).
- `embedding_stats.json` + a printed **health-check report** covering NaN
  / Inf rows, "dead" (near-zero-variance) dimensions, and embedding norm
  statistics — a sanity gate before the embeddings are handed to topic
  modeling.

### CLI usage
```bash
python extract_embeddings.py \
    --backbone   ./checkpoints/best_backbone.pt \
    --data_dir   ./video_level_pose \
    --output_dir ./train_encoder_output/embeddings \
    --batch_size 32 --device cuda
```

This step is run **once per checkpoint of interest** (e.g. `best_backbone`,
`epoch_35`, `epoch_45`, `epoch_60`) to produce the set of candidate
embedding matrices evaluated in Step 3.

---

## Step 3 — `checkpoint_selection.ipynb`: Checkpoint / Epoch Selection

**Goal:** Compare embeddings extracted from several candidate checkpoints
(Step 2 outputs) and select the checkpoint whose embedding space produces
the most coherent unsupervised clustering — used as a proxy for
representation quality, since no labeled downstream task is available.

### Method
For each candidate embedding set (`best_backbone`, `epoch_35`, `epoch_45`,
`epoch_60`):
1. L2-normalize the embeddings.
2. Reduce dimensionality with **UMAP** (cosine metric).
3. Cluster the reduced space with **HDBSCAN** (`leaf` cluster selection).
4. Score the resulting clustering with:
   - **Number of clusters found**
   - **Noise-point percentage** (points HDBSCAN leaves unassigned)
   - **Silhouette score** (cluster cohesion/separation; higher is better)
   - **Davies–Bouldin index** (average cluster similarity; lower is
     better)

Two parameter regimes are evaluated for robustness:
- *Coarse*: `UMAP(n_neighbors=60, n_components=5)`,
  `HDBSCAN(min_cluster_size=120, min_samples=30)` — favors a small number
  of broad, high-confidence clusters.
- *Fine*: `UMAP(n_neighbors=15, n_components=15)`,
  `HDBSCAN(min_cluster_size=25, min_samples=5)` — favors a larger number
  of more granular clusters, stress-testing whether structure holds at
  finer resolution.

### Output
Printed comparison tables (per checkpoint, per parameter regime) of
cluster count, noise %, silhouette, and Davies–Bouldin score. The
checkpoint with the strongest silhouette / Davies–Bouldin trade-off (and
acceptable noise %) is selected as the embedding source for Step 4.

> In the current run, `best_backbone` (epoch 94, selected by the
> training loop's own early-stopping criterion) and `epoch_45` are the
> strongest candidates; `epoch_45` was ultimately used as the embedding
> source for topic modeling in Step 4.

---

## Step 4 — `topic_modeling.ipynb`: BERTopic Topic Discovery

**Goal:** Discover and label latent semantic topics in the ISL paragraph
corpus, using the selected pose embeddings from Step 3 as the clustering
representation and the paired ASR/gloss **text transcripts** for keyword
extraction and human-readable labeling — i.e., pose embeddings drive
*where* documents cluster, while text drives *what each cluster means*.

### Input data
- `embeddings.npy` from the selected checkpoint (Step 3's winner).
- A CSV of paired text transcripts (`text` column), row-aligned with the
  embeddings.

### Method
1. **BERTopic** is configured with `embedding_model=None` — the
   pre-computed pose embeddings are supplied directly, so BERTopic's own
   text-embedding step is bypassed and only its downstream
   reduction/clustering/representation stages are used:
   - `UMAP(n_neighbors=60, n_components=5, metric="cosine")` for
     dimensionality reduction.
   - `HDBSCAN(min_cluster_size=120, min_samples=10, cluster_selection_method="leaf")`
     for clustering.
   - `CountVectorizer` with a custom stop-word list (standard English
     stop words plus filler/backchannel tokens common in ASR transcripts:
     "uh", "um", "okay", "yeah", etc.) and `ngram_range=(1, 3)`.
   - `ClassTfidfTransformer(reduce_frequent_words=True)` for class-based
     TF-IDF keyword weighting per topic.
2. **Outlier reduction** (`reduce_outliers`) reassigns documents initially
   labeled as noise (`-1`) to their nearest topic where possible, followed
   by `update_topics` to refresh topic representations.
3. **Topic labeling**: candidate topic keywords/representative documents
   are inspected, and a Gemini-based LLM helper (`GeminiKeyFailover`) is
   used to propose short (2–5 word) descriptive labels per topic from its
   representative documents, followed by manual review and correction
   into a final `manual_labels` mapping.
4. Topics are further grouped into **macro-categories** by manual
   inspection (e.g. *Mathematics & Science Education*, *Indian News &
   Current Affairs*, *Deaf Community & Sign Language*, *General Knowledge
   & Mixed*).

### Visualizations produced
- Intertopic distance map (`intertopic_distance_map.pdf/.svg`).
- Topic similarity heatmap (`topic_heatmap.pdf/.png`), computed from
  cosine similarity between topic embeddings.
- 2D UMAP projection of the pose embedding space, colored by discovered
  topic (`umap_topics.pdf`), including a noise-highlighted variant.

### Output
- `topic_model2` — the fitted BERTopic model (topics, representative
  documents, keyword weights, topic embeddings).
- Topic info table (topic ID, size, keywords, manual label).
- Figures listed above, exported at publication-ready resolution
  (300–400 DPI, PDF/SVG for vector reproduction).

---

## Environment Notes

- **Step 1 & 2** require PyTorch with CUDA support, `pandas`, `numpy`,
  `tqdm`.
- **Step 3 & 4** require `umap-learn`, `hdbscan`, `scikit-learn`,
  `bertopic`, `seaborn`, `matplotlib`, `plotly` 
  for static figure export and `google-generativeai` for
  LLM-assisted topic labeling.



---

## Reproducing the Pipeline End-to-End

```bash
# Step 1: pretrain the pose encoder
python train_pose_encoder.py

# Step 2: extract embeddings for one or more checkpoints
python extract_embeddings.py \
    --backbone ./checkpoints/checkpoint_epoch.pt \
    --data_dir ./video_level_pose \
    --output_dir ./embeddings/epoch45

python extract_embeddings.py \
    --backbone ./checkpoints/best_backbone.pt \
    --data_dir ./video_level_pose \
    --output_dir ./embeddings/best

# Step 3: compare candidate checkpoints (open in Jupyter)
jupyter notebook checkpoint_selection.ipynb

# Step 4: run topic modeling on the selected embeddings (open in Jupyter)
jupyter notebook topic_modeling.ipynb
```