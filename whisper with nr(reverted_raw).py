"""
Transcribe ATCOSIM with openai/whisper-medium (general-purpose baseline).

    python whisper_transcribe.py   # writes transcripts/<OUT_TAG>.csv
    python rescore.py              # scores every CSV in transcripts/
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
from huggingface_hub import hf_hub_download

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
OUT_TAG        = "whisper_medium_raw"
NUM_SAMPLES    = 1900
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
    print("Downloading parquet file...")
    parquet_path = hf_hub_download(
        repo_id="Jzuluaga/atcosim_corpus",
        filename="data/train-00000-of-00004-c1d7fb31dcbf644a.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples")
    return df


df = _load_dataset()


# =============================================================================
# TRANSCRIPTION
# =============================================================================
n = min(NUM_SAMPLES, len(df))
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
os.makedirs("transcripts", exist_ok=True)
out_csv = os.path.join("transcripts", f"{OUT_TAG}.csv")
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

print("Next step:  python rescore.py")