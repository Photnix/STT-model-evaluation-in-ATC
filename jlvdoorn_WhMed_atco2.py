"""
Transcribe ATCOSIM with jlvdoorn/whisper-medium-atco2-asr.

    python whisper_atc_atco2_transcribe.py   # writes transcripts/<OUT_TAG>.csv
    python rescore.py                        # scores every CSV in transcripts/
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
from huggingface_hub import hf_hub_download

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

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
MODEL_NAME     = "jlvdoorn/whisper-medium-atco2-asr"
OUT_TAG        = "whisper_medium_atc_atco2"
ENGLISH_ONLY   = False      # True for any ".en" checkpoint
FP16           = False

NUM_SAMPLES    = 1900
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
    _PROCESSOR = WhisperProcessor.from_pretrained(MODEL_NAME)
    _MODEL = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    _MODEL.eval()

    if FP16 and DEVICE == "cuda":
        _MODEL = _MODEL.half()
    _MODEL = _MODEL.to(DEVICE)

    if getattr(_MODEL.generation_config, "forced_decoder_ids", None) is not None:
        _MODEL.generation_config.forced_decoder_ids = None

    print(f"Model loaded on: {next(_MODEL.parameters()).device}")
    print(f"dtype: {next(_MODEL.parameters()).dtype}")
    print(f"English-only checkpoint: {ENGLISH_ONLY}")


def transcribe_whisper(audio_array: np.ndarray, sr: int) -> str:
    _ensure_model()

    features = _PROCESSOR(
        audio_array, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(DEVICE)

    if FP16 and DEVICE == "cuda":
        features = features.half()

    gen_kwargs = dict(max_new_tokens=200, num_beams=1, do_sample=False)
    if not ENGLISH_ONLY:
        gen_kwargs.update(language="en", task="transcribe")

    with torch.no_grad():
        ids = _MODEL.generate(features, **gen_kwargs)

    return _PROCESSOR.batch_decode(ids, skip_special_tokens=True)[0].strip()


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
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

    raw_hypothesis = transcribe_whisper(audio_array, TARGET_SR)

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