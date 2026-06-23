
"""

ISL Coreference Resolution Pipeline

Usage:
    python isl_coref_pipeline.py \
        --input  path/to/translations.csv \
        --output path/to/output_coref.csv \
        --review path/to/human_review.csv \
        --checkpoint_dir path/to/checkpoints \
        --batch_size 128
"""

import argparse
import logging
import os
import re
import pickle


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("fastcoref").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"




import pandas as pd
import torch
from tqdm import tqdm
from fastcoref import FCoref, LingMessCoref



# Checkpoint helpers

def ckpt_path(checkpoint_dir: str, name: str) -> str:
    return os.path.join(checkpoint_dir, name)

def save_ckpt(obj, checkpoint_dir: str, name: str):
    path = ckpt_path(checkpoint_dir, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[ckpt] Saved  → {path}")

def load_ckpt(checkpoint_dir: str, name: str):
    path = ckpt_path(checkpoint_dir, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[ckpt] Loaded ← {path}")
        return obj
    return None



# 1. CLI Arguments


def parse_args():
    parser = argparse.ArgumentParser(description="ISL Coreference Resolution Pipeline")

    
    parser.add_argument("--input",  default="/DATACSEShare/sanjeet/Khushi/coherence/orig_ds.csv",  help="Input CSV with ISL translations")
    parser.add_argument("--output", default="/DATACSEShare/sanjeet/Khushi/coherence/coref.csv",  help="Output CSV with coref annotations")
    parser.add_argument("--review", default="/DATACSEShare/sanjeet/Khushi/coherence/human_review.csv",  help="Output CSV for human review (low confidence)")
    parser.add_argument("--confidence_threshold", type=float, default=0.80,
                        help="Review threshold")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for fastcoref inference (default: 32)")
    parser.add_argument("--model", choices=["lingmess", "fcoref"], default="lingmess",
                        help="Coreference model to use (default: lingmess — more accurate)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--checkpoint_dir", default="/DATACSEShare/sanjeet/Khushi/coherence/checkpoints",
                        help="Directory to save/load checkpoints (default: ./checkpoints)")
    return parser.parse_args()



# 2. Load & Validate Data


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"uid", "text", "video_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    before = len(df)
    # df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"[load] Loaded {before} rows -> {len(df)} after dropping empty transcripts")
    return df



# 3. Build Pseudo-Documents Per Video


def extract_clip_order(uid: str) -> int:
    m = re.search(r"_clip_(\d+)$", uid)
    if m:
        return int(m.group(1))
    m = re.search(r"-(\d+)$", uid)
    if m:
        return int(m.group(1))
    return 0


def build_video_groups(df: pd.DataFrame) -> dict:
    groups = {}
    for yt_id, grp in df.groupby("video_id"):
        grp = grp.copy()
        grp["_order"] = grp["uid"].apply(extract_clip_order)
        grp = grp.sort_values("_order").reset_index(drop=True)

        clips = []
        doc_parts = []
        offset = 0

        for _, row in grp.iterrows():
            text = str(row["text"]).strip()
            char_start = offset
            char_end   = offset + len(text)
            clips.append({
                "uid":    row["uid"],
                "text":       text,
                "char_start": int(char_start),
                "char_end":   int(char_end),
            })
            doc_parts.append(text)
            offset = char_end + 1

        groups[yt_id] = {
            "clips":      clips,
            "pseudo_doc": " ".join(doc_parts),
        }

    print(f"[build] Built {len(groups)} video pseudo-documents")
    return groups



# 4. Load Coreference Model


def load_model(model_name: str, device: str):
    print(f"[model] Loading {model_name} on {device} ...")
    if model_name == "lingmess":
        model = LingMessCoref(device=device, enable_progress_bar=False)
    else:
        model = FCoref(device=device, enable_progress_bar=False)
    print(" Model loaded ")
    return model



# 5. Run Coreference Inference  


def run_coref_on_groups(groups: dict, model, batch_size: int, checkpoint_dir: str) -> dict:
    """
    Saves a checkpoint after every batch so progress survives crashes.
    On restart, already-completed batches are skipped automatically.

    Checkpoint: step2_coref_raw.pkl  ->  { yt_id: result_object }
    """
    CKPT_NAME = "step2_coref_raw.pkl"

    # Load previous results if available
    coref_map = load_ckpt(checkpoint_dir, CKPT_NAME) or {}
    done_ids  = set(coref_map.keys())

    yt_ids = list(groups.keys())
    

    # Skip completed videos
    ids_left  = [yt for yt in yt_ids if yt not in done_ids]
    docs_left = [groups[yt]["pseudo_doc"] for yt in ids_left]

    if not ids_left:
        print(f"[coref] All {len(yt_ids)} documents already in checkpoint — skipping inference.")
        return coref_map

    print(f"[coref] {len(done_ids)} already done, {len(ids_left)} remaining.")
    print(f"[coref] Running inference (batch_size={batch_size}) ...")

    with tqdm(total=len(docs_left), desc="[coref]", unit="doc") as pbar:
        for i in range(0, len(docs_left), batch_size):
            batch_ids  = ids_left[i : i + batch_size]
            batch_docs = docs_left[i : i + batch_size]

            batch_result = model.predict(texts=batch_docs)
            if not isinstance(batch_result, list):
                batch_result = [batch_result]

            # Merge into coref_map and save checkpoint
            for yt_id, result in zip(batch_ids, batch_result):
                coref_map[yt_id] = result

            save_ckpt(coref_map, checkpoint_dir, CKPT_NAME)
            pbar.update(len(batch_docs))

    print(f"[coref] Inference complete   ({len(coref_map)} videos)")
    return coref_map



# 6. Map Clusters to Clip-Level Annotations


PRONOUN_SET = {
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "its", "their", "theirs",
    "this", "that", "these", "those",
    "we", "us", "our", "ours",
    "who", "which",
    "himself", "herself", "itself", "themselves", "ourselves",
}


def char_offset_to_uidx(char_start: int, clips: list) -> int:
    """Returns index into clips[] that owns char_start. -1 if not found."""
    char_start = int(char_start)
    for idx, clip in enumerate(clips):
        if int(clip["char_start"]) <= char_start < int(clip["char_end"]):
            return idx
    return -1

def logit_to_confidence(logit: float) -> float:
    return float(torch.sigmoid(torch.tensor(logit)).item())


def extract_annotations(groups: dict, coref_map: dict,
                         confidence_threshold: float, checkpoint_dir: str) -> list:
    """Convert FastCoref clusters to clip-level annotations."""
    CKPT_NAME = "step3_rows.pkl"
    cached = load_ckpt(checkpoint_dir, CKPT_NAME)
    if cached is not None:
        print(f"[annotate] Loaded {len(cached):,} rows from checkpoint — skipping annotation.")
        return cached

    rows = []
    for yt_id, group_data in tqdm(groups.items(), desc="[annotate]"):
        if yt_id not in coref_map:          
            print(f"[annotate] WARNING: {yt_id} missing from coref_map, skipping.")
            continue
        clips  = group_data["clips"]
        result = coref_map[yt_id]

        clusters_spans   = result.get_clusters(as_strings=False)  # [(start_char, end_char), ...]
        clusters_strings = result.get_clusters(as_strings=True)   # ["mention text", ...]

        clip_annotations = {c["uid"]: [] for c in clips}

        for span_cluster, str_cluster in zip(clusters_spans, clusters_strings):
            # fastcoref may return None spans occasionally
            valid = [(span, text) for span, text in zip(span_cluster, str_cluster) if span is not None]
            if len(valid) < 2:          # need at least antecedent and one mention
                continue
            
            mentions = sorted(valid, key=lambda x: x[0][0])

            antecedent_span, antecedent_text = mentions[0]
            ant_uidx = char_offset_to_uidx(antecedent_span[0], clips)
            if ant_uidx == -1:
                continue

            for mention_span, mention_text in mentions[1:]:
                if mention_span[0] is None:   
                    continue
                men_uidx = char_offset_to_uidx(mention_span[0], clips)
                if men_uidx == -1 or men_uidx <= ant_uidx:
                    continue

                
                try:
                    logit = result.get_logit(
                        span_i=antecedent_span,
                        span_j=mention_span
                    )
                    confidence = logit_to_confidence(logit)
                except Exception:
                    print("Fallback to 0.7 and 0.9")
                    
                    is_pronoun = mention_text.strip().lower() in PRONOUN_SET
                    confidence = 0.90 if is_pronoun else 0.70

                men_uid = clips[men_uidx]["uid"]
                clip_annotations[men_uid].append({
                    "mention":         mention_text.strip(),
                    "resolves_to_uid": clips[ant_uidx]["uid"],
                    "resolved_entity": antecedent_text.strip(),
                    "confidence":      round(confidence, 4),
                })

                

        for clip in clips:
            cid    = clip["uid"]
            annots = clip_annotations[cid]
            if not annots:
                rows.append({
                    "uid": cid,
                    "video_id": yt_id,
                    "text": clip["text"],
                    "has_coreference": False,
                    "mention": None,
                    "resolves_to_uid": None,
                    "resolved_entity": None,
                    "confidence": None,
                    "needs_review": False,
                })
            else:
                for annot in annots:
                    rows.append({
                        "uid": cid,
                        "video_id": yt_id,
                        "text": clip["text"],
                        "has_coreference": True,
                        "mention": annot["mention"],
                        "resolves_to_uid": annot["resolves_to_uid"],
                        "resolved_entity": annot["resolved_entity"],
                        "confidence": annot["confidence"],
                        "needs_review": annot["confidence"] < confidence_threshold,
                    })

    save_ckpt(rows, checkpoint_dir, CKPT_NAME)
    return rows



# 7. Save Outputs


def save_outputs(rows: list, output_path: str, review_path: str):
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"[save] Full output   -> {output_path}  ({len(df_out):,} rows)")

    df_review = df_out[df_out["needs_review"] == True].copy()
    df_review.to_csv(review_path, index=False)
    print(f"[save] Human review  -> {review_path}  ({len(df_review):,} rows flagged)")

    n_coref = int(df_out["has_coreference"].sum())
    n_total = len(df_out)
    pct     = 100 * n_coref / n_total if n_total else 0
    print(f"\n{'─'*50}")
    print(f"  Total clip rows    : {n_total:,}")
    print(f"  Has coreference    : {n_coref:,}  ({pct:.1f}%)")
    print(f"  Flagged for review : {len(df_review):,}")
    print(f"{'─'*50}\n")



# 8. Main


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("  ISL Coreference Resolution Pipeline")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*50}\n")

    # Step 1  Build groups (checkpoint: step1_groups.pkl)
    groups = load_ckpt(args.checkpoint_dir, "step1_groups.pkl")
    if groups is None:
        df     = load_data(args.input)
        groups = build_video_groups(df)
        save_ckpt(groups, args.checkpoint_dir, "step1_groups.pkl")

   
    # Step 2  Coreference inference (checkpoint: step2_coref_raw.pkl)
    _coref_map_cached = load_ckpt(args.checkpoint_dir, "step2_coref_raw.pkl") or {}
    _missing_check    = [yt for yt in groups if yt not in _coref_map_cached]

    if _missing_check:
        model     = load_model(args.model, args.device)
        coref_map = run_coref_on_groups(groups, model, args.batch_size, args.checkpoint_dir)
        missing   = [yt for yt in groups if yt not in coref_map]
        if missing:
            print(f"[coref] Re-running {len(missing)} missing videos ...")
            for yt_id in tqdm(missing, desc="[coref-retry]"):
                result = model.predict(texts=[groups[yt_id]["pseudo_doc"]])
                coref_map[yt_id] = result[0] if isinstance(result, list) else result
            save_ckpt(coref_map, args.checkpoint_dir, "step2_coref_raw.pkl")
    else:
        print(f"[coref] All {len(_coref_map_cached)} videos in checkpoint — skipping model load.")
        coref_map = _coref_map_cached

    # Step 3 Annotate (checkpoint: step3_rows.pkl)
    rows = extract_annotations(groups, coref_map, args.confidence_threshold, args.checkpoint_dir)

    # Step 4 Write CSVs
    save_outputs(rows, args.output, args.review)

    print("Pipeline complete ")


if __name__ == "__main__":
    main()