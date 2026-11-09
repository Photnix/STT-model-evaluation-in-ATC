"""
Transcribe ATCOSIM with openai/whisper-medium (general-purpose baseline).

    python whisper_raw.py   # writes transcripts/<OUT_TAG>.csv
    python rescore_test.py   # scores every CSV in transcripts/
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import time
import random
import warnings

import whisper
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

warnings.filterwarnings("ignore", category=UserWarning)

random.seed(0)
np.random.seed(0)

import torch
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")


# =============================================================================
# CONFIG
# =============================================================================
MODEL_SIZE     = "medium"
SPLIT          = os.environ.get("ATCOSIM_SPLIT", "train")
OUT_TAG        = f"whisper_medium_raw_{SPLIT}"
OUT_DIR        = "transcripts" if SPLIT == "test" else f"transcripts_{SPLIT}"
NUM_SAMPLES    = None
PROGRESS_EVERY = 25

TARGET_SR = 16000


# =============================================================================
# MODEL
# =============================================================================
print(f"Loading Whisper {MODEL_SIZE}...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
print(f"Model loaded on: {next(model.parameters()).device}")


# =============================================================================
# DATA
# =============================================================================
def _load_dataset():
    print(f"Loading ATCOSIM {SPLIT} split...")
    ds = load_dataset("Jzuluaga/atcosim_corpus", split=SPLIT)
    ds = ds.cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()
    print(f"Loaded {len(df)} samples")
    return df


df = _load_dataset()


# =============================================================================
# TRANSCRIPTION
# =============================================================================
n = len(df) if NUM_SAMPLES is None else min(NUM_SAMPLES, len(df))
print(f"\nTranscribing {n} utterances with whisper-{MODEL_SIZE}...")

raw_rows = []
t0 = time.time()

for idx in range(n):
    sample = df.iloc[idx]

    audio_array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]),
                              dtype="float32")
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    if sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr,
                                       target_sr=TARGET_SR)
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

    result = model.transcribe(
        audio_array,
        language="en",
        task="transcribe",
        temperature=0,
        fp16=(DEVICE == "cuda"),
    )

    raw_rows.append({
        "id": sample["id"],
        "raw_ref": str(sample["text"]),
        "raw_hyp": result["text"],
    })

    if (idx + 1) % PROGRESS_EVERY == 0:
        rate = (idx + 1) / (time.time() - t0)
        print(f"  {idx + 1}/{n}   ({rate:.1f} utt/s)")

elapsed = time.time() - t0


# =============================================================================
# OUTPUT
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)
out_csv = os.path.join(OUT_DIR, f"{OUT_TAG}.csv")
if os.path.exists(out_csv):
    backup_dir = os.path.join(OUT_DIR, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(out_csv)))
    backup_csv = os.path.join(backup_dir, f"{OUT_TAG}_{stamp}.csv")
    os.replace(out_csv, backup_csv)
    print(f"Previous file moved to: {backup_csv}")
pd.DataFrame(raw_rows).to_csv(out_csv, index=False)

empty = sum(1 for r in raw_rows if not str(r["raw_hyp"]).strip())

print("\n==== TRANSCRIPTION COMPLETE ====")
print(f"Model:         whisper-{MODEL_SIZE}")
print(f"Preprocessing: raw")
print(f"Utterances:    {len(raw_rows)}")
print(f"Empty output:  {empty}")
print(f"Elapsed:       {elapsed:.1f}s  ({elapsed / max(len(raw_rows), 1):.3f}s per utterance)")
print(f"Written to:    {out_csv}")

print("\nFirst 3 pairs (raw, unnormalized):")
for r in raw_rows[:3]:
    print(f"  ref: {r['raw_ref']}")
    print(f"  hyp: {str(r['raw_hyp']).strip()}\n")

print("Next step:  python rescore_test.py")
