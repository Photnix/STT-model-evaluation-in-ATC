"""
Transcribe ATCOSIM with Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc.

    python w2v2_atc_uwbatcc_transcribe.py   # writes transcripts/<OUT_TAG>.csv
    python rescore.py                       # scores every CSV in transcripts/
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_TQDM"] = "1"

import io
import time
import random
import logging
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

import torch
from transformers import Wav2Vec2Processor, AutoModelForCTC

warnings.filterwarnings("ignore", category=UserWarning)
for _n in ("transformers", "huggingface_hub", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME     = "Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc"
OUT_TAG        = "w2v2_lv60_atc_uwbatcc_test"
NUM_SAMPLES    = 1901
PROGRESS_EVERY = 25
SAMPLE_EVERY   = 200

TARGET_SR = 16000


# =============================================================================
# MODEL
# =============================================================================
_PROCESSOR = None
_MODEL = None


def _ensure_model():
    global _PROCESSOR, _MODEL
    if _PROCESSOR is not None and _MODEL is not None:
        return

    print(f"Loading {MODEL_NAME}...")
    # Wav2Vec2Processor, NOT AutoProcessor. This repo ships a KenLM 4-gram;
    # AutoProcessor would detect it and silently enable LM beam search, which
    # the general baseline does not have.
    _PROCESSOR = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    _MODEL = AutoModelForCTC.from_pretrained(MODEL_NAME)
    _MODEL.eval()
    _MODEL = _MODEL.to(DEVICE)
    print(f"Model loaded on: {next(_MODEL.parameters()).device}")
    print(f"Vocab size: {_MODEL.config.vocab_size}")


def transcribe_ctc(audio_array: np.ndarray, sr: int) -> str:
    _ensure_model()

    input_values = _PROCESSOR(
        audio_array, sampling_rate=sr, return_tensors="pt", padding="longest"
    ).input_values.to(DEVICE)

    with torch.no_grad():
        logits = _MODEL(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1).cpu()
    return _PROCESSOR.batch_decode(predicted_ids)[0].strip()


# =============================================================================
# DATA
# =============================================================================
def _load_dataset():
    print("Loading ATCOSIM test split...")
    ds = load_dataset("Jzuluaga/atcosim_corpus", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()
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
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

    raw_hypothesis = transcribe_ctc(audio_array, TARGET_SR)

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
