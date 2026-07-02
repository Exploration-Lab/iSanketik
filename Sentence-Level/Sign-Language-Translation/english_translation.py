"""
===============
Indian Sign Language (ISL) → Text Translation — Training Script
Dataset  : iSign 1.0  (shard-based WebDataset)
Model    : BERT encoder  +  GPT-2 decoder  (HuggingFace EncoderDecoderModel)

Key techniques
  [1] Temporal positional encoding  – sinusoidal PE added to frame embeddings
                                      so the encoder sees frame order
  [2] Feature projection layer      – two-layer MLP + LayerNorm + GELU projects
                                      raw keypoints into the encoder's hidden space
  [3] Label smoothing               – redistributes a small probability mass to
                                      non-target tokens, reducing overconfidence
  [4] Gradient clipping @ 1.0       – caps the gradient norm to prevent
                                      exploding-gradient instability
  [5] Frame-level attention mask    – zero-padded frames are masked out so the
                                      encoder attends only to real frames
  [6] Decoder freeze / unfreeze     – decoder is frozen for the first N epochs
                                      while the encoder warms up; then unfrozen
                                      for end-to-end fine-tuning
  [7] R-Drop regularisation         – two stochastic forward passes are made per
                                      step; the symmetrical KL divergence between
                                      their output distributions acts as an
                                      additional consistency regulariser

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iSign 1.0  —  hyperparameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  train_csv             /DATA1007/vaibhav/tokenization/train_split_unicode_filtered.csv
  val_csv               /DATA1007/vaibhav/tokenization/val_split_unicode_filtered.csv
  test_csv              /DATA1007/vaibhav/tokenization/test_split_unicode_filtered.csv
  shard_dir             /indian-slp/dattatreya/isign1.0
  learning_rate         2e-4
  encoder/decoder       4 layers, hidden=512, 8 heads
  dropout               0.15
  MAX_FRAMES            80
  max_length            40 tokens
  vocab_size_decoder    7200  (BPE)
  num_keypoints         152
  batch_size            512
  num_epochs            150
  WEIGHT_DECAY          0.05
  warmup_steps_ratio    0.15
  GRAD_ACCUM_STEPS      1
  RDROP_ALPHA           0.5
  LABEL_SMOOTHING       0.1
  FREEZE_DECODER_EPOCHS 10
  GEN_LENGTH_PENALTY    0.6
  GEN_NO_REPEAT_NGRAM   3
  GEN_MIN_LENGTH        2
  STEP_FRAMES_ISIGN     5
  ADD_NOISE_ISIGN       False

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iSign 2.0 / iSanketik  —  hyperparameters
To switch to this dataset, replace the values in SECTION 1
with those listed below and set _RUN_TAG accordingly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  train_csv             /indian-slp/dattatreya/missing_data/isign2.1newupdatedmay24.csv
  val_csv               /DATA7/vaibhav/tokenization/val_split_unicode_filtered.csv
  test_csv              /DATA7/vaibhav/tokenization/test_split_unicode_filtered.csv
  shard_dir             /indian-slp/dattatreya/isign2.1
  learning_rate         2e-4
  encoder/decoder       4 layers, hidden=512, 8 heads
  dropout               0.15
  MAX_FRAMES            80
  max_length            40 tokens
  vocab_size_decoder    10000  (BPE — larger vocab than iSign 1.0)
  num_keypoints         152
  batch_size            512
  num_epochs            150
  WEIGHT_DECAY          0.05
  warmup_steps_ratio    0.15
  GRAD_ACCUM_STEPS      1
  RDROP_ALPHA           0.5
  LABEL_SMOOTHING       0.1
  FREEZE_DECODER_EPOCHS 10
  GEN_LENGTH_PENALTY    0.6
  GEN_NO_REPEAT_NGRAM   3
  GEN_MIN_LENGTH        2
  STEP_FRAMES_ISIGN     5
  ADD_NOISE_ISIGN       False
"""

# ── Silence noisy parallelism and CUDA fragmentation warnings ──────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import random
import sacrebleu
from tqdm import tqdm
from rouge_score import rouge_scorer

from transformers import (
    BertConfig, BertModel,               # encoder backbone
    GPT2Config, GPT2LMHeadModel,         # decoder backbone
    EncoderDecoderModel,                 # glues encoder + decoder together
    PreTrainedTokenizerFast,             # HuggingFace fast tokenizer wrapper
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers  # HF tokenizers

from helpers.bleu_cal import quick_bleu_metric    # corpus BLEU-1/2/3/4
from helpers.dataloaders import WebDatasetWrapper  # shard-based keypoint loader

import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
warnings.filterwarnings("ignore", message="Version v4.12.0 introduces a better way")

torch.cuda.empty_cache()  # free leftover GPU memory from any previous run


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – HYPERPARAMETERS  (iSign 1.0)
# All tuneable values live here so they are easy to locate and change without
# hunting through the rest of the file.
# ══════════════════════════════════════════════════════════════════════════════

# ── W&B identifiers ───────────────────────────────────────────────────────
# project_name groups all related runs on the W&B dashboard.
# run_name labels this individual experiment within that project.
project_name = "ISL_v1_baseline"
run_name     = "ISL_v1_train"

# Use only GPU 0; set before any CUDA initialisation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Visible GPUs:", torch.cuda.device_count())

# ── Dataset CSV paths (iSign 1.0) ─────────────────────────────────────────
# Each CSV contains columns: uid, video_id, text
# The actual skeleton keypoints are stored in WebDataset shards on disk.
train_csv = "/DATA1007/vaibhav/tokenization/train_split_unicode_filtered.csv"
val_csv   = "/DATA1007/vaibhav/tokenization/val_split_unicode_filtered.csv"
test_csv  = "/DATA1007/vaibhav/tokenization/test_split_unicode_filtered.csv"

# ── Optimiser ─────────────────────────────────────────────────────────────
learning_rate    = 2e-4   # peak LR; cosine schedule decays from this value
WEIGHT_DECAY     = 0.05   # AdamW L2 penalty — reduces overfitting
warmup_steps_ratio = 0.15 # fraction of total steps used for linear LR warm-up
GRAD_ACCUM_STEPS = 1      # accumulate gradients over N micro-batches before stepping

# ── Architecture ──────────────────────────────────────────────────────────
num_encoder_layers  = 4    # number of BERT transformer blocks
num_decoder_layers  = 4    # number of GPT-2 transformer blocks
encoder_hidden_size = 512  # BERT hidden dim (also the cross-attention dim)
decoder_hidden_size = 512  # GPT-2 hidden dim (must equal encoder_hidden_size here)
num_attention_heads = 8    # must divide both hidden sizes evenly (512 / 8 = 64)
dropout             = 0.15 # dropout probability for attention and hidden layers

# ── Sequence lengths ──────────────────────────────────────────────────────
MAX_FRAMES          = 80   # max skeleton frames kept per video clip (temporal axis)
max_length_train    = 40   # max decoder token sequence length during training
max_length_generate = 40   # max tokens the model may produce at inference time
vocab_size_decoder  = 7200 # BPE vocabulary size trained on iSign 1.0 texts
num_keypoints       = 152  # feature dimension per frame (2-D pose keypoints × joints)

# ── Training loop ─────────────────────────────────────────────────────────
batch_size = 512
num_epochs = 150
num_beams  = 3    # beam search width during per-epoch evaluation

# ── Regularisation ────────────────────────────────────────────────────────
RDROP_ALPHA           = 0.5   # [7] KL weight; higher → stronger consistency pressure
LABEL_SMOOTHING       = 0.1   # [3] smoothing factor; 0 = standard CE, 1 = uniform
FREEZE_DECODER_EPOCHS = 10    # [6] freeze decoder for this many epochs at the start

# ── Generation (beam search) settings ─────────────────────────────────────
GEN_LENGTH_PENALTY  = 0.6   # <1 → shorter outputs favoured; >1 → longer
GEN_NO_REPEAT_NGRAM = 3     # forbid any n-gram from repeating (reduces loops)
GEN_MIN_LENGTH      = 2     # shortest allowed output in tokens
GEN_EARLY_STOPPING  = True  # stop all beams once the top beam reaches EOS

# ── Data loading (ISL-specific) ───────────────────────────────────────────
STEP_FRAMES_ISIGN = 5     # temporal sub-sampling stride (every 5th frame)
ADD_NOISE_ISIGN   = False # add Gaussian noise to keypoints (data augmentation)
SHARD_DIR         = "/indian-slp/dattatreya/isign1.0"  # root of WebDataset shards


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – OUTPUT DIRECTORIES & FILE PATHS
# All artefacts (checkpoints, predictions, metrics) are written under a shared
# run tag so multiple experiments do not overwrite each other.
# ══════════════════════════════════════════════════════════════════════════════

# Short tag appended to every output directory name.
# Change this if you want to keep multiple experiment trees side-by-side.
_RUN_TAG = "isign1_baseline"

CKPT_DIR  = f"checkpoints_{_RUN_TAG}"   # saved model weights
PRED_DIR  = f"predictions_{_RUN_TAG}"   # per-epoch CSV predictions
MTR_DIR   = f"metrics_{_RUN_TAG}"       # XLSX + JSON metric logs
MERGE_DIR = f"mergecsv_{_RUN_TAG}"      # UID → shard mapping CSVs

for d in [CKPT_DIR, PRED_DIR, MTR_DIR, MERGE_DIR]:
    os.makedirs(d, exist_ok=True)

METRICS_FILE      = f"{MTR_DIR}/metrics_summary.xlsx"      # rolling per-epoch table
BEST_METRICS_FILE = f"{MTR_DIR}/best_metrics_summary.json" # best-so-far snapshot

# ── Checkpoint / resume settings ─────────────────────────────────────────
RESUME           = True                          # resume from last checkpoint if available
RESUME_CKPT_PATH = f"{CKPT_DIR}/last_checkpoint.pth"
SAVE_EVERY_EPOCH = True                          # overwrite last_checkpoint.pth each epoch
SAVE_LAST_PATH   = f"{CKPT_DIR}/last_checkpoint.pth"

# ── W&B resume settings ───────────────────────────────────────────────────
WANDB_RESUME      = True
WANDB_RUN_ID_FILE = f"{CKPT_DIR}/wandb_run_id.txt"  # stores run ID across restarts

# ── Tokeniser output path ─────────────────────────────────────────────────
TOKENIZER_DIR  = f"tokenizer_{_RUN_TAG}"
TOKENIZER_FILE = f"{TOKENIZER_DIR}/isign_tokenizer.json"
os.makedirs(TOKENIZER_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – REPRODUCIBILITY
# Fixing all random seeds ensures that two runs with identical configs produce
# identical results, and that a resumed run matches a full run exactly.
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Fix Python, NumPy, PyTorch (CPU + all CUDA), and cuDNN seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # same op order every run
    torch.backends.cudnn.benchmark     = False  # disable auto-tuner (non-deterministic)

set_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – DATA LOADING
# The CSVs provide the text labels and the uid→video_id mapping.
# Actual skeleton keypoints live in the WebDataset shards referenced by SHARD_DIR.
# ══════════════════════════════════════════════════════════════════════════════

print("Loading datasets...")
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
# Use val as a stand-in test set if no dedicated test CSV exists
test_df  = pd.read_csv(test_csv) if os.path.exists(test_csv) else val_df
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – TOKENISER TRAINING (BPE)
# A Byte-Pair Encoding tokeniser is trained from scratch on the target-language
# sentences.  This is done at runtime so the
# vocabulary perfectly matches the data split being used.
# Special tokens cover sentence boundaries, padding, and domain-specific placeholders.
# ══════════════════════════════════════════════════════════════════════════════

print("Training BPE tokenizer...")

# Pool train + val text so the vocabulary covers all seen words
all_texts = train_df['text'].tolist() + val_df['text'].tolist()

tokenizer_model = models.BPE()
tokenizer = Tokenizer(tokenizer_model)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # split on whitespace before BPE

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size_decoder,
    special_tokens=[
        "<s>",        # BOS – decoder start / beginning-of-sequence
        "<pad>",      # PAD – padding to uniform batch length
        "</s>",       # EOS – end-of-sequence marker
        "<unk>",      # UNK – any subword not in vocabulary
        "<mask>",     # MASK – reserved for possible MLM experiments
        "<PERSON>",   # anonymise named persons in transcripts
        "<UNKNOWN>",  # placeholder for signs without a gloss
    ]
)
tokenizer.train_from_iterator(all_texts, trainer=trainer)
tokenizer.save(TOKENIZER_FILE)

# Wrap in HuggingFace's PreTrainedTokenizerFast to get batch_encode, batch_decode,
# and generate() compatibility for free.
tokenizer_target = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
tokenizer_target.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "additional_special_tokens": ["<PERSON>", "<UNKNOWN>"],
})
print(f"Tokenizer trained — vocab size: {len(tokenizer_target)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – DATASET & DATALOADER CONSTRUCTION
# WebDatasetWrapper reads pre-processed shards of normalised skeleton keypoints.
# uid→video_id dicts allow the loader to route each sample to the right shard.
# ══════════════════════════════════════════════════════════════════════════════

# Build uid → video_id look-ups used inside WebDatasetWrapper for shard routing
train_uid_to_vid = dict(zip(train_df["uid"].astype(str), train_df["video_id"].astype(str)))
val_uid_to_vid   = dict(zip(val_df["uid"].astype(str),   val_df["video_id"].astype(str)))
test_uid_to_vid  = dict(zip(test_df["uid"].astype(str),  test_df["video_id"].astype(str)))


def _make_dataset(split_name: str, uid_to_videoid: dict,
                  shuffle: bool, merge_csv_path: str) -> WebDatasetWrapper:
    """
    Instantiate a WebDatasetWrapper for any split.

    Args:
        split_name     : "train" | "val" | "test"
        uid_to_videoid : dict mapping UID strings to video ID strings
        shuffle        : True for training (randomise order), False for eval
        merge_csv_path : path where the UID→shard mapping CSV is written
    Returns:
        WebDatasetWrapper ready to pass to DataLoader
    """
    return WebDatasetWrapper(
        split=split_name,
        tokenizer=tokenizer_target,
        uid_to_videoid=uid_to_videoid,
        max_frames=MAX_FRAMES,
        max_length_decoder=max_length_train,
        step_frames=STEP_FRAMES_ISIGN,   # temporal sub-sampling stride
        add_noise=ADD_NOISE_ISIGN,
        window_size=1,        # no sliding-window augmentation
        stride=1,
        shuffle=shuffle,
        random_window=False,
        cross_video_random=False,
        seed=42,
        shard_dir=SHARD_DIR,
        save_merge_csv=True,
        merge_csv_path=merge_csv_path,
    )


train_dataset = _make_dataset("train", train_uid_to_vid, shuffle=True,
                              merge_csv_path=f"{MERGE_DIR}/merge_train.csv")
val_dataset   = _make_dataset("val",   val_uid_to_vid,   shuffle=False,
                              merge_csv_path=f"{MERGE_DIR}/merge_val.csv")
test_dataset  = _make_dataset("test",  test_uid_to_vid,  shuffle=False,
                              merge_csv_path=f"{MERGE_DIR}/merge_test.csv")

# pin_memory=True speeds up CPU→GPU data transfers on CUDA systems
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train samples: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
b = next(iter(train_loader))
print(f"Batch shapes — input_ids: {b['input_ids'].shape} | "
      f"attention_mask: {b['attention_mask'].shape} | labels: {b['labels'].shape}")
print("Example sample uid:", train_dataset.sample_uids[0])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – MODEL DEFINITION
#
# End-to-end architecture:
#   Input  : (B, T, num_keypoints)   — batch of frame-keypoint sequences
#   Step 1 : FeatureProjection       → (B, T, encoder_hidden_size)
#   Step 2 : TemporalPositionalEncoding → adds sinusoidal frame-order signal
#   Step 3 : BERT encoder            → contextualised frame representations
#   Step 4 : GPT-2 decoder (+ cross-attention) → autoregressive token prediction
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── BERT encoder ──────────────────────────────────────────────────────────
encoder = BertModel(BertConfig(
    hidden_size=encoder_hidden_size,
    num_hidden_layers=num_encoder_layers,
    num_attention_heads=num_attention_heads,
    hidden_dropout_prob=dropout,
    attention_probs_dropout_prob=dropout,
))

# ── GPT-2 decoder ─────────────────────────────────────────────────────────
# add_cross_attention=True enables the cross-attention sublayer that attends
# over encoder outputs; without this it is a standard language model only.
decoder = GPT2LMHeadModel(GPT2Config(
    vocab_size=len(tokenizer_target),
    n_positions=max(max_length_train, max_length_generate),
    n_embd=decoder_hidden_size,
    n_layer=num_decoder_layers,
    n_head=num_attention_heads,
    pad_token_id=tokenizer_target.pad_token_id,
    bos_token_id=tokenizer_target.bos_token_id,
    eos_token_id=tokenizer_target.eos_token_id,
    add_cross_attention=True,
    embd_pdrop=dropout,
    attn_pdrop=dropout,
    resid_pdrop=dropout,
))


# ── [2] Feature Projection Layer ──────────────────────────────────────────
class FeatureProjection(torch.nn.Module):
    """
    Projects raw per-frame keypoint vectors into the encoder's hidden space.

    Architecture:
        Linear(num_keypoints → 512) → LayerNorm → GELU → Dropout(0.1)
        → Linear(512 → encoder_hidden_size) → LayerNorm

    Design notes:
      • Two LayerNorm layers stabilise training: one after the first linear
        (prevents activation explosion) and one before the encoder (ensures
        the encoder receives well-scaled inputs).
      • GELU is used instead of ReLU to match BERT's internal activations.
      • A small Dropout(0.1) between the two linear layers adds regularisation
        without over-suppressing the signal.

    Args:
        input_dim  : number of keypoints per frame  (num_keypoints = 152)
        output_dim : encoder hidden size            (encoder_hidden_size = 512)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, output_dim),
            torch.nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, input_dim) — batch of frame-keypoint sequences
        Returns:
            (B, T, output_dim)    — projected frame embeddings
        """
        return self.net(x)


# ── [1] Temporal Positional Encoding ──────────────────────────────────────
class TemporalPositionalEncoding(torch.nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017) for skeleton frames.

    Without positional information the BERT encoder treats all frames as an
    unordered set, discarding the temporal structure that is critical for
    understanding sign language.  Adding this fixed (non-learned) encoding
    gives the encoder a signal about each frame's position in the sequence.

    Args:
        d_model : embedding dimension — must match encoder_hidden_size
        max_len : maximum expected sequence length (MAX_FRAMES = 80)
    """
    def __init__(self, d_model: int, max_len: int = 80):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims → sine waves
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims → cosine waves
        # register_buffer: stored in state_dict but not treated as a learnable parameter
        self.register_buffer('pe', pe.unsqueeze(0))   # shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, d_model) — projected frame embeddings
        Returns:
            (B, T, d_model) — embeddings with positional signal added
        """
        return x + self.pe[:, :x.size(1)]


# ── Instantiate auxiliary modules ─────────────────────────────────────────
feature_projection = FeatureProjection(num_keypoints, encoder_hidden_size)
temporal_pe        = TemporalPositionalEncoding(encoder_hidden_size, MAX_FRAMES)

# ── Assemble encoder-decoder ──────────────────────────────────────────────
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# Register special token IDs so generate() knows where sequences start / end
model.config.decoder_start_token_id = tokenizer_target.bos_token_id
model.config.eos_token_id           = tokenizer_target.eos_token_id
model.config.pad_token_id           = tokenizer_target.pad_token_id
model.config.vocab_size             = len(tokenizer_target)
model.config.max_length             = max(max_length_train, max_length_generate)
model.config.label_smoothing_factor = LABEL_SMOOTHING  # [3]

model              = model.to(device)
feature_projection = feature_projection.to(device)
temporal_pe        = temporal_pe.to(device)

# Enable multi-GPU data parallelism when more than one GPU is visible
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model              = torch.nn.DataParallel(model)
    feature_projection = torch.nn.DataParallel(feature_projection)

print(f"Device: {device}")
print(f"Projection weight device: {next(feature_projection.parameters()).device}")
print(f"Model weight device:      {next(model.parameters()).device}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – OPTIMISER & LEARNING RATE SCHEDULE
# AdamW with cosine decay and a linear warm-up period.
# A hard LR floor (5 % of peak) prevents the schedule from collapsing to
# near-zero in the final epochs where small updates are still useful.
# ══════════════════════════════════════════════════════════════════════════════

# Jointly optimise encoder-decoder, projection, and temporal PE
optimizer = torch.optim.AdamW(
    list(model.parameters())
    + list(feature_projection.parameters())
    + list(temporal_pe.parameters()),
    lr=learning_rate,
    weight_decay=WEIGHT_DECAY,
)

total_steps  = (len(train_loader) // GRAD_ACCUM_STEPS) * num_epochs
warmup_steps = int(warmup_steps_ratio * total_steps)


def cosine_with_floor(current_step: int) -> float:
    """
    Combined linear warm-up + cosine decay schedule with a hard LR floor.

    Phase 1 — warm-up (steps 0 … warmup_steps):
        LR increases linearly from 0 to learning_rate.
        This prevents large, destabilising gradient steps at the very start.

    Phase 2 — cosine decay (steps warmup_steps … total_steps):
        LR follows a half-cosine curve from learning_rate down toward 0,
        but is clamped to at least 5 % of the peak value.

    Args:
        current_step : current global optimiser step count
    Returns:
        Multiplicative factor applied to the base learning_rate by LambdaLR
    """
    min_lr_ratio = 0.05
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    progress     = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr_ratio, cosine_decay)


scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_floor)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – WEIGHTS & BIASES INITIALISATION
# All hyperparameters are logged as run config so every experiment is fully
# reproducible from the W&B dashboard.  The run ID is persisted to disk so
# that a resumed run continues the same W&B entry (not a duplicate run).
# ══════════════════════════════════════════════════════════════════════════════

hyperparameters = {
    "dataset":              "iSign1.0",
    "learning_rate":        learning_rate,
    "batch_size":           batch_size,
    "num_epochs":           num_epochs,
    "MAX_FRAMES":           MAX_FRAMES,
    "num_beams":            num_beams,
    "dropout":              dropout,
    "max_length_train":     max_length_train,
    "max_length_generate":  max_length_generate,
    "grad_accum_steps":     GRAD_ACCUM_STEPS,
    "label_smoothing":      LABEL_SMOOTHING,
    "rdrop_alpha":          RDROP_ALPHA,
    "freeze_decoder_epochs":FREEZE_DECODER_EPOCHS,
    "weight_decay":         WEIGHT_DECAY,
    "warmup_steps_ratio":   warmup_steps_ratio,
    "gen_length_penalty":   GEN_LENGTH_PENALTY,
    "gen_no_repeat_ngram":  GEN_NO_REPEAT_NGRAM,
    "gen_min_length":       GEN_MIN_LENGTH,
    "train_samples":        len(train_df),
    "val_samples":          len(val_df),
    "test_samples":         len(test_df),
}

# Try to reuse an existing W&B run ID (enables continuation on the dashboard)
wandb_id = None
if WANDB_RESUME and os.path.exists(WANDB_RUN_ID_FILE):
    with open(WANDB_RUN_ID_FILE, "r") as f:
        wandb_id = f.read().strip()

run = wandb.init(
    project=project_name,
    name=run_name,
    config=hyperparameters,
    id=wandb_id,
    resume="allow" if WANDB_RESUME else None,
)

# Persist the new run ID so future resumes can find it
if WANDB_RESUME and wandb_id is None:
    with open(WANDB_RUN_ID_FILE, "w") as f:
        f.write(run.id)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – METRIC TRACKING STATE & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

metrics_history  = []        # list of dicts, one entry per (epoch, split)
rouge_scorer_obj = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
)

# Running best values — updated whenever a new high is reached
best_val_bleu4  = 0.0
best_val_loss   = float('inf')
best_test_bleu4 = 0.0
best_test_loss  = float('inf')


def save_best_metrics_json(val_bleu4: float, val_loss: float,
                            test_bleu4: float, test_loss: float,
                            epoch: int) -> None:
    """
    Write current best metric values to a small JSON file.

    This provides a quick human-readable snapshot of training progress that
    survives job restarts without parsing the full XLSX metrics table.
    BLEU scores are stored as percentages (multiplied by 100).

    Args:
        val_bleu4  : best BLEU-4 on the validation split (0–1 range)
        val_loss   : best cross-entropy loss on the validation split
        test_bleu4 : best BLEU-4 on the test split (0–1 range)
        test_loss  : best cross-entropy loss on the test split
        epoch      : epoch at which these bests were achieved
    """
    data = {
        "best_val_bleu4":       round(val_bleu4  * 100, 4),
        "best_val_loss":        round(val_loss,   6),
        "best_test_bleu4":      round(test_bleu4 * 100, 4),
        "best_test_loss":       round(test_loss,  6),
        "epoch_of_last_update": epoch,
    }
    with open(BEST_METRICS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f">>> Best metrics saved → {BEST_METRICS_FILE}")


def load_checkpoint(path: str) -> int:
    """
    Restore all training state from a .pth checkpoint file.

    Restores: model weights, projection weights, temporal PE weights,
    optimiser state (momentum / adaptive LR buffers), scheduler state,
    best metric trackers, full metrics history, global step counter,
    and all RNG states (Python / NumPy / PyTorch / CUDA).

    Args:
        path : filesystem path to the checkpoint file
    Returns:
        start_epoch : epoch to begin training from (checkpoint epoch + 1)
    """
    global best_val_bleu4, best_val_loss, best_test_bleu4, best_test_loss
    global metrics_history, global_step

    ckpt = torch.load(path, map_location="cpu")  # load to CPU first to avoid OOM

    # Unwrap DataParallel before loading state dicts
    model_raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    proj_raw  = feature_projection.module if isinstance(feature_projection, torch.nn.DataParallel) else feature_projection

    model_raw.load_state_dict(ckpt["model_state_dict"], strict=True)
    proj_raw.load_state_dict(ckpt["proj_state_dict"],   strict=True)
    if "temporal_pe_state_dict" in ckpt:
        temporal_pe.load_state_dict(ckpt["temporal_pe_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    best_val_bleu4  = ckpt.get("best_val_bleu4",  0.0)
    best_val_loss   = ckpt.get("best_val_loss",   float("inf"))
    best_test_bleu4 = ckpt.get("best_test_bleu4", 0.0)
    best_test_loss  = ckpt.get("best_test_loss",  float("inf"))
    metrics_history = ckpt.get("metrics_history", [])
    global_step     = ckpt.get("global_step", 0)

    # Restore RNG states for exact reproducibility — failure is non-fatal
    try:
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.set_rng_state(ckpt["rng_torch"])
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    except Exception as e:
        print(f"Warning: could not fully restore RNG states: {e}")

    start_epoch = ckpt["epoch"] + 1
    print(f">>> Resumed from {path} — next epoch: {start_epoch}, global_step: {global_step}")
    return start_epoch


def save_checkpoint(path: str, epoch: int, global_step: int,
                    best_val_bleu4: float, best_val_loss: float,
                    best_test_bleu4: float, best_test_loss: float,
                    metrics_history: list) -> None:
    """
    Persist the complete training state to a checkpoint file.

    The file is self-contained: loading it is sufficient to exactly replicate
    the training state at the end of `epoch`, including RNG seeds.
    DataParallel wrappers are stripped before serialisation so the checkpoint
    can be loaded on any GPU count (including CPU-only environments).

    Args:
        path            : destination file path
        epoch           : last completed epoch index
        global_step     : total optimiser steps taken so far
        best_val_bleu4  : best val BLEU-4 seen so far (0–1)
        best_val_loss   : best val loss seen so far
        best_test_bleu4 : best test BLEU-4 seen so far (0–1)
        best_test_loss  : best test loss seen so far
        metrics_history : full per-epoch metric log list
    """
    model_raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    proj_raw  = feature_projection.module if isinstance(feature_projection, torch.nn.DataParallel) else feature_projection

    torch.save({
        "epoch":                 epoch,
        "global_step":           global_step,
        "best_val_bleu4":        best_val_bleu4,
        "best_val_loss":         best_val_loss,
        "best_test_bleu4":       best_test_bleu4,
        "best_test_loss":        best_test_loss,
        "metrics_history":       metrics_history,
        "model_state_dict":      model_raw.state_dict(),
        "proj_state_dict":       proj_raw.state_dict(),
        "temporal_pe_state_dict":temporal_pe.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict(),
        # RNG states for bit-exact resume
        "rng_python":            random.getstate(),
        "rng_numpy":             np.random.get_state(),
        "rng_torch":             torch.get_rng_state(),
        "rng_cuda":              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – EVALUATION FUNCTION
# Runs a full teacher-forced forward pass (for loss) and a beam-search
# generation pass (for translation quality) over an entire DataLoader.
# Computes BLEU-1/2/3/4, SacreBLEU, and ROUGE-1/2/L.
# Saves per-sample predictions to CSV and logs everything to W&B.
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_save(loader, epoch: int, split: str = "val"):
    """
    Evaluate the model on one split and persist results.

    Steps:
      1. Teacher-forced forward pass → cross-entropy loss.
      2. Beam-search generation      → predicted token sequences.
      3. Decode predictions and references to text strings.
      4. Compute BLEU-1/2/3/4, SacreBLEU, ROUGE-1/2/L.
      5. Log metrics to W&B.
      6. Append to metrics_history and save to XLSX.
      7. Save (uid, reference, prediction) CSV for manual inspection.

    Args:
        loader : DataLoader for the split to evaluate
        epoch  : current epoch number (used in output filenames)
        split  : "val" or "test"

    Returns:
        avg_loss  : mean cross-entropy loss (float)
        b4        : BLEU-4 score in 0–1 range (float)
        all_preds : list of decoded prediction strings
        all_refs  : list of decoded reference strings
    """
    model.eval()
    feature_projection.eval()

    total_loss = 0.0
    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split.upper()} Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)      # (B, T, num_keypoints)
            labels    = batch['labels'].to(device)          # (B, max_length_train)

            # Step 1: project keypoints → encoder hidden space
            proj = feature_projection(input_ids).view(
                input_ids.size(0), -1, encoder_hidden_size
            )
            # Step 2: add temporal positional signal so the encoder sees frame order
            proj = temporal_pe(proj)  # (B, T, encoder_hidden_size)

            # [5] Build binary frame mask: 1 for real frames, 0 for zero-padded frames.
            # A frame is considered padding if ALL keypoint values are exactly zero.
            frame_mask = (input_ids.abs().sum(-1) != 0).long()  # (B, T)

            # Teacher-forced pass to get training loss (used for model selection)
            outputs = model(inputs_embeds=proj, attention_mask=frame_mask, labels=labels)
            total_loss += outputs.loss.item()

            # Beam-search decoding: produces the best predicted token sequence
            generated = model.generate(
                inputs_embeds=proj,
                attention_mask=frame_mask,
                max_length=max_length_generate,
                min_length=GEN_MIN_LENGTH,
                num_beams=num_beams,
                length_penalty=GEN_LENGTH_PENALTY,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
                early_stopping=GEN_EARLY_STOPPING,
            )

            # Replace -100 ignore-index in labels with pad_token_id before decoding
            labels_clean = torch.where(
                labels == -100, tokenizer_target.pad_token_id, labels
            )
            preds = tokenizer_target.batch_decode(generated,     skip_special_tokens=True)
            refs  = tokenizer_target.batch_decode(labels_clean,  skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    avg_loss = total_loss / len(loader)

    # ── BLEU 1–4 (corpus-level) ───────────────────────────────────────────
    try:
        b1, b2, b3, b4 = quick_bleu_metric(
            [[ref.split()] for ref in all_refs],
            [pred.split() for pred in all_preds],
            split=split.upper(),
        )
    except ZeroDivisionError:
        # Can happen in very early epochs when all outputs are empty strings
        print(f"WARNING: empty predictions at epoch {epoch} {split} — BLEU set to 0")
        wandb.log({f"{split}/empty_pred_warning": 1, "epoch": epoch})
        b1, b2, b3, b4 = 0.0, 0.0, 0.0, 0.0

    # SacreBLEU: tokenisation-independent BLEU; comparable across papers
    sacre_score = sacrebleu.corpus_bleu(all_preds, [all_refs]).score

    # ── ROUGE (per-sample, then averaged) ────────────────────────────────
    r1_list, r2_list, rl_list = [], [], []
    for ref, pred in zip(all_refs, all_preds):
        rs = rouge_scorer_obj.score(ref, pred)
        r1_list.append(rs['rouge1'].fmeasure)
        r2_list.append(rs['rouge2'].fmeasure)
        rl_list.append(rs['rougeL'].fmeasure)
    avg_r1 = float(np.mean(r1_list)) if r1_list else 0.0
    avg_r2 = float(np.mean(r2_list)) if r2_list else 0.0
    avg_rl = float(np.mean(rl_list)) if rl_list else 0.0

    # ── Save per-sample predictions CSV ─────────────────────────────────
    pd.DataFrame({
        "uid":        loader.dataset.sample_uids,
        "reference":  all_refs,
        "prediction": all_preds,
    }).to_csv(f"{PRED_DIR}/predictions_epoch{epoch}_{split}.csv", index=False)

    # ── Log to W&B ───────────────────────────────────────────────────────
    wandb.log({
        f"{split}/loss":      avg_loss,
        f"{split}/bleu1":     b1  * 100,
        f"{split}/bleu2":     b2  * 100,
        f"{split}/bleu3":     b3  * 100,
        f"{split}/bleu4":     b4  * 100,
        f"{split}/sacrebleu": sacre_score,
        f"{split}/rouge1":    avg_r1 * 100,
        f"{split}/rouge2":    avg_r2 * 100,
        f"{split}/rouge_l":   avg_rl * 100,
        "epoch":              epoch,
    })

    # ── Append to rolling history and persist ────────────────────────────
    metrics_history.append({
        "epoch":     epoch,
        "split":     split,
        "loss":      round(avg_loss,      4),
        "bleu1":     round(b1  * 100,     2),
        "bleu2":     round(b2  * 100,     2),
        "bleu3":     round(b3  * 100,     2),
        "bleu4":     round(b4  * 100,     2),
        "sacrebleu": round(sacre_score,   2),
        "rouge1":    round(avg_r1 * 100,  2),
        "rouge2":    round(avg_r2 * 100,  2),
        "rouge_l":   round(avg_rl * 100,  2),
    })
    pd.DataFrame(metrics_history).to_excel(METRICS_FILE, index=False)

    print(
        f"{split.upper()} Epoch {epoch} | Loss: {avg_loss:.4f} | "
        f"BLEU-4: {b4*100:.2f} | SacreBLEU: {sacre_score:.2f} | "
        f"R1: {avg_r1*100:.2f} | R2: {avg_r2*100:.2f} | RL: {avg_rl*100:.2f}"
    )

    model.train()
    feature_projection.train()
    return avg_loss, b4, all_preds, all_refs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

model.train()
feature_projection.train()
global_step = 0  # total optimiser steps across all epochs (used by the LR scheduler)

# Resume from an existing checkpoint if one is available
start_epoch = 1
if RESUME and os.path.exists(RESUME_CKPT_PATH):
    start_epoch = load_checkpoint(RESUME_CKPT_PATH)

for epoch in range(start_epoch, num_epochs + 1):

    # ── [6] Decoder freeze / unfreeze ────────────────────────────────────
    # During the first FREEZE_DECODER_EPOCHS only the encoder and projection
    # are trained.  This lets the encoder develop meaningful frame representations
    # before the decoder's randomly-initialised cross-attention layers join
    # training and potentially corrupt them.
    model_raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    if epoch <= FREEZE_DECODER_EPOCHS:
        for param in model_raw.decoder.parameters():
            param.requires_grad = False
        if epoch == start_epoch:
            print(f">>> Decoder FROZEN for epochs 1–{FREEZE_DECODER_EPOCHS}")
    else:
        for param in model_raw.decoder.parameters():
            param.requires_grad = True
        if epoch == FREEZE_DECODER_EPOCHS + 1:
            print(f">>> Decoder UNFROZEN from epoch {epoch}")

    epoch_loss = 0.0
    progress   = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    optimizer.zero_grad()

    for step_idx, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(device)  # (B, T, num_keypoints)
        labels    = batch['labels'].to(device)      # (B, max_length_train), -100 for pad

        # Project keypoints → encoder hidden space, then add temporal position signal
        proj = feature_projection(input_ids).view(
            input_ids.size(0), -1, encoder_hidden_size
        )
        proj = temporal_pe(proj)  # (B, T, encoder_hidden_size)

        # [5] Zero-padded frames masked out so the encoder ignores them
        frame_mask = (input_ids.abs().sum(-1) != 0).long()  # (B, T)

        # ── [7] R-Drop: two stochastic forward passes ─────────────────────
        # Because dropout is still active at training time, each pass produces
        # a slightly different output distribution.  Minimising the KL between
        # the two distributions forces the model to be consistent despite dropout,
        # acting as an additional regulariser beyond the CE loss alone.
        outputs1 = model(inputs_embeds=proj, attention_mask=frame_mask, labels=labels)
        outputs2 = model(inputs_embeds=proj, attention_mask=frame_mask, labels=labels)

        # Symmetrical KL divergence (average of both directions)
        p = F.log_softmax(outputs1.logits, dim=-1)
        q = F.log_softmax(outputs2.logits, dim=-1)
        kl_loss = 0.5 * (
            F.kl_div(p, q.detach().exp(), reduction='batchmean') +
            F.kl_div(q, p.detach().exp(), reduction='batchmean')
        )

        # Average CE loss across both passes to reduce gradient variance
        ce_loss = 0.5 * (outputs1.loss + outputs2.loss)

        # Combined loss divided by accumulation steps before backward
        loss = (ce_loss + RDROP_ALPHA * kl_loss) / GRAD_ACCUM_STEPS
        loss.backward()

        epoch_loss += ce_loss.item()

        # Optimiser step every GRAD_ACCUM_STEPS micro-batches (or at end of epoch)
        if (step_idx + 1) % GRAD_ACCUM_STEPS == 0 or (step_idx + 1) == len(train_loader):
            global_step += 1

            # [4] Clip gradient norm to 1.0 — prevents any single update being too large
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(feature_projection.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress.set_postfix({
                "ce":  f"{ce_loss.item():.4f}",
                "kl":  f"{kl_loss.item():.4f}",
                "lr":  f"{scheduler.get_last_lr()[0]:.2e}",
            })
            wandb.log({
                "train/ce_loss":    ce_loss.item(),
                "train/kl_loss":    kl_loss.item(),
                "train/total_loss": (ce_loss + RDROP_ALPHA * kl_loss).item(),
                "train/lr":         scheduler.get_last_lr()[0],
                "global_step":      global_step,
                "decoder_frozen":   int(epoch <= FREEZE_DECODER_EPOCHS),
            })

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f}")

    # ── Per-epoch evaluation ──────────────────────────────────────────────
    print("\n--- Validation ---")
    val_loss,  val_bleu4,  _, _ = evaluate_and_save(val_loader,  epoch, split="val")

    print("\n--- Test ---")
    test_loss, test_bleu4, _, _ = evaluate_and_save(test_loader, epoch, split="test")

    # Update W&B run summary with current best values (visible on project overview)
    wandb.run.summary["best_val_bleu4"]  = best_val_bleu4  * 100
    wandb.run.summary["best_val_loss"]   = best_val_loss
    wandb.run.summary["best_test_bleu4"] = best_test_bleu4 * 100
    wandb.run.summary["best_test_loss"]  = best_test_loss

    # ── Best val model checkpoint ─────────────────────────────────────────
    # Primary criterion: BLEU-4; tie-break: lower loss
    new_best_val = val_bleu4 > best_val_bleu4 or (
        abs(val_bleu4 - best_val_bleu4) < 1e-4 and val_loss < best_val_loss
    )
    if new_best_val:
        best_val_bleu4 = val_bleu4
        best_val_loss  = val_loss
        save_checkpoint(f"{CKPT_DIR}/best_val_model.pth",
                        epoch, global_step, best_val_bleu4, best_val_loss,
                        best_test_bleu4, best_test_loss, metrics_history)
        save_best_metrics_json(best_val_bleu4, best_val_loss,
                               best_test_bleu4, best_test_loss, epoch)
        print(">>> NEW BEST VAL MODEL saved (Val BLEU-4 improved)")

    # ── Best test model checkpoint ────────────────────────────────────────
    new_best_test = test_bleu4 > best_test_bleu4 or (
        abs(test_bleu4 - best_test_bleu4) < 1e-4 and test_loss < best_test_loss
    )
    if new_best_test:
        best_test_bleu4 = test_bleu4
        best_test_loss  = test_loss
        save_checkpoint(f"{CKPT_DIR}/best_test_model.pth",
                        epoch, global_step, best_val_bleu4, best_val_loss,
                        best_test_bleu4, best_test_loss, metrics_history)
        save_best_metrics_json(best_val_bleu4, best_val_loss,
                               best_test_bleu4, best_test_loss, epoch)
        print(">>> NEW BEST TEST MODEL saved (Test BLEU-4 improved)")

    # ── Rolling last-checkpoint (always overwrite) ────────────────────────
    if SAVE_EVERY_EPOCH:
        save_checkpoint(SAVE_LAST_PATH,
                        epoch, global_step, best_val_bleu4, best_val_loss,
                        best_test_bleu4, best_test_loss, metrics_history)
        print(f">>> Last checkpoint saved → {SAVE_LAST_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 – DONE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"  Metrics history : {METRICS_FILE}")
print(f"  Best metrics    : {BEST_METRICS_FILE}")
print(f"  Checkpoints     : {CKPT_DIR}/")
print(f"  Predictions     : {PRED_DIR}/")
print("=" * 60)

wandb.finish()