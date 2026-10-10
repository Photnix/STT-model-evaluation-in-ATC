"""
Transcribe ATCOSIM with vosk-model-en-us-0.22 (general-purpose baseline).
Vosk is Kaldi-based and runs on CPU only — cannot use CUDA.
"""

import io
import os
import json
import time
import logging
import warnings
import contextlib

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from huggingface_hub import hf_hub_download
from vosk import Model, KaldiRecognizer, SetLogLevel

warnings.filterwarnings("ignore", category=UserWarning)
for _n in ("huggingface_hub", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)
SetLogLevel(-1)


@contextlib.contextmanager
def suppress_stdout_stderr():
    devnull = os.open(os.devnull, os.O_RDWR)
    old_stdout, old_stderr = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)


# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH     = r"C:\Users\bliu0\OneDrive\Desktop\vosk stt model\vosk-model-en-us-0.22"
MODEL_NAME     = "vosk-model-en-us-0.22"
OUT_TAG        = "vosk_0.22_raw"
NUM_SAMPLES    = 1900
PROGRESS_EVERY = 25
SAMPLE_EVERY   = 200

TARGET_SR  = 16000
CHUNK_BYTES = 4000 * 2          # 4000 frames * 2 bytes per PCM16 sample


# =============================================================================
# MODEL
# =============================================================================
print(f"Loading {MODEL_NAME}...")
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}")
with suppress_stdout_stderr():
    vosk_model = Model(MODEL_PATH)
print("Model loaded (CPU).")


def transcribe_vosk(audio_array: np.ndarray, sr: int) -> str:
    pcm = (np.clip(audio_array, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    rec = KaldiRecognizer(vosk_model, float(sr))
    rec.SetWords(False)

    chunks = []
    for off in range(0, len(pcm), CHUNK_BYTES):
        if rec.AcceptWaveform(pcm[off:off + CHUNK_BYTES]):
            chunks.append(json.loads(rec.Result()).get("text", ""))
    chunks.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(c for c in chunks if c).strip()


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
print(f"\nTranscribing {n} utterances with {MODEL_NAME}...")

raw_rows = []
t0 = time.time()

for idx in range(n):
    sample = df.iloc[idx]

    audio_array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]),
                              dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    if sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr,
                                       target_sr=TARGET_SR)
        sr = TARGET_SR
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

    raw_hypothesis = transcribe_vosk(audio_array, sr)

    raw_rows.append({
        "id": sample["id"],
        "raw_ref": str(sample["text"]),
        "raw_hyp": raw_hypothesis,
    })

    if idx == 0 or (idx + 1) % SAMPLE_EVERY == 0:
        print(f"    [{idx + 1}] ref: {str(sample['text'])}")
        print(f"    [{idx + 1}] hyp: {raw_hypothesis}")

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
print(f"Model:         {MODEL_NAME}")
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
