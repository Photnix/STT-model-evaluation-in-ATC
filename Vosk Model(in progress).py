from vosk import Model, KaldiRecognizer
import wave
import json
import os
import io
import pandas as pd
import tempfile
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download
from jiwer import wer, cer

import warnings
import re
warnings.filterwarnings("ignore", category=UserWarning)
import contextlib
import sys
import logging


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr (including C-level prints) within the context."""
    devnull = os.open(os.devnull, os.O_RDWR)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)

NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
TENS_WORDS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def int_to_words(number: int) -> str:
    if number < 0:
        return "minus " + int_to_words(-number)
    if number < 20:
        return NUM_WORDS[number]
    if number < 100:
        if number in TENS_WORDS:
            return TENS_WORDS[number]
        return TENS_WORDS[number - number % 10] + " " + NUM_WORDS[number % 10]
    if number < 1000:
        hundreds = number // 100
        remainder = number % 100
        result = NUM_WORDS[hundreds] + " hundred"
        if remainder:
            result += " " + int_to_words(remainder)
        return result
    return str(number)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", lambda m: int_to_words(int(m.group(0))), text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ICAO ATC digit normalization
ATC_DIGITS = {
    "zero": "zero", "one": "one", "two": "two", "three": "three", "tree": "three",
    "four": "four", "five": "five", "fife": "five", "six": "six", "seven": "seven",
    "eight": "eight", "nine": "nine", "niner": "nine"
}

CARDINALS = {
    "ten": ["one", "zero"],
    "eleven": ["one", "one"],
    "twelve": ["one", "two"],
    "thirteen": ["one", "three"],
    "fourteen": ["one", "four"],
    "fifteen": ["one", "five"],
    "sixteen": ["one", "six"],
    "seventeen": ["one", "seven"],
    "eighteen": ["one", "eight"],
    "nineteen": ["one", "nine"],
    "twenty": ["two"],
    "thirty": ["three"],
    "forty": ["four"],
    "fifty": ["five"],
    "sixty": ["six"],
    "seventy": ["seven"],
    "eighty": ["eight"],
    "ninety": ["nine"]
}


def normalize_number_phrase(words):
    tokens = words.split()

    if "hundred" in tokens:
        idx = tokens.index("hundred")
        hundreds_digit = tokens[idx - 1]

        if hundreds_digit in ATC_DIGITS:
            hundreds_digit = ATC_DIGITS[hundreds_digit]

        remainder = tokens[idx + 1:]
        converted = []
        for token in remainder:
            if token in CARDINALS:
                converted.extend(CARDINALS[token])
            elif token in ATC_DIGITS:
                converted.append(ATC_DIGITS[token])

        return " ".join([hundreds_digit] + converted)

    converted = []
    for token in tokens:
        if token in CARDINALS:
            converted.extend(CARDINALS[token])
        elif token in ATC_DIGITS:
            converted.append(ATC_DIGITS[token])
        else:
            converted.append(token)

    return " ".join(converted)


NUMBER_WORDS = set(ATC_DIGITS.keys()) | set(CARDINALS.keys()) | {"hundred"}


def normalize_digits(text):
    tokens = text.split()
    output = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token in NUMBER_WORDS:
            phrase = [token]
            j = i + 1

            while j < len(tokens) and tokens[j] in NUMBER_WORDS:
                phrase.append(tokens[j])
                j += 1

            normalized = normalize_number_phrase(" ".join(phrase))
            output.append(normalized)
            i = j
        else:
            output.append(token)
            i += 1

    return " ".join(output)


SEMANTIC_KEYWORDS = {
    "left", "right",
    "turn", "climb", "descend", "maintain", "contact", "report", "continue",
    "proceed", "hold", "cleared", "expedite", "intercept",
    "heading", "course", "radar", "identified", "approach", "departure",
    "runway", "landing", "takeoff", "tower", "ground",
    "frequency", "decimal", "radio",
    "flight", "level", "altitude",
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
    "hundred", "thousand",
    "north", "south", "east", "west",
    "lufthansa", "sabena", "transwede", "transavia", "speedbird", "swissair",
    "india", "oscar", "kilo",
    "zurich", "rhein", "trasadingen", "dinkelsbuhl"
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def semantic_similarity(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    ref_sem = {token for token in ref_tokens if token in SEMANTIC_KEYWORDS}
    hyp_sem = {token for token in hyp_tokens if token in SEMANTIC_KEYWORDS}

    if not ref_sem:
        return 1.0 if not hyp_sem else 0.0

    overlap = len(ref_sem & hyp_sem)
    return overlap / len(ref_sem)


def semantic_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    ref_sem = [token for token in ref_tokens if token in SEMANTIC_KEYWORDS]
    hyp_sem = [token for token in hyp_tokens if token in SEMANTIC_KEYWORDS]

    errors = 0
    for token in ref_sem:
        if token not in hyp_sem:
            errors += 1

    if len(ref_sem) == 0:
        return 0.0

    return errors / len(ref_sem)


def compute_sts(reference: str, hypothesis: str) -> float:
    return semantic_similarity(reference, hypothesis)


def compute_ser(reference: str, hypothesis: str) -> float:
    return semantic_error_rate(reference, hypothesis)


def _make_fallback_audio_bytes(text: str, sr: int = 16000, duration: float = 0.2) -> bytes:
    frequency = 220 + len(text) % 7 * 55
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format="WAV")
    return buffer.getvalue()


def _load_dataset():
    try:
        print("Downloading parquet file...")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        parquet_path = hf_hub_download(
            repo_id="Jzuluaga/atcosim_corpus",
            filename="data/train-00000-of-00004-c1d7fb31dcbf644a.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} samples")
        return df
    except Exception as exc:
        print(f"Falling back to built-in sample data because the parquet download failed: {exc}")
        fallback_texts = [
            "zero one two three",
            "the quick brown fox jumps over the lazy dog",
            "speech recognition evaluation",
        ]
        rows = []
        for text in fallback_texts:
            rows.append({"text": text, "audio": {"bytes": _make_fallback_audio_bytes(text)}})
        return pd.DataFrame(rows)


# -----------------------------
# Download dataset parquet
# -----------------------------
df = _load_dataset()

# -----------------------------
# Load Vosk model
# -----------------------------
print("Loading Vosk model...")
VOSK_MODEL_PATH = r"C:\Users\bliu0\OneDrive\Desktop\vosk stt model\vosk-model-en-us-0.22"
print("VOSK_MODEL_PATH:", VOSK_MODEL_PATH)
print("Path exists:", os.path.exists(VOSK_MODEL_PATH))
try:
    print("Listing model folder contents:")
    if os.path.exists(VOSK_MODEL_PATH):
        print(os.listdir(VOSK_MODEL_PATH))
except Exception as _e:
    print("Could not list folder contents:", _e)
try:
    with suppress_stdout_stderr():
        vosk_model = Model(VOSK_MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load Vosk model from '{VOSK_MODEL_PATH}': {e}\n"
        "Download an English Vosk model and set VOSK_MODEL_PATH to its folder."
    ) from e

# -----------------------------
# Evaluation
# -----------------------------
NUM_SAMPLES = 50  # increase later (e.g. 100+)

wer_scores = []
cer_scores_all = []
sts_scores = []
ser_scores = []

for idx in range(NUM_SAMPLES):
    print(f"\n--- Sample {idx + 1} ---")

    sample = df.iloc[idx]

    reference = sample["text"].lower().strip()
    reference = normalize_digits(reference)
    reference = normalize_text(reference)
    audio_bytes = sample["audio"]["bytes"]

    # Write audio bytes to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    # Transcribe with Vosk
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result_text += " " + res.get("text", "")
    final_res = json.loads(rec.FinalResult())
    result_text += " " + final_res.get("text", "")

    hypothesis = result_text.lower().strip()
    hypothesis = normalize_digits(hypothesis)
    hypothesis = normalize_text(hypothesis)

    print("Reference:  ", reference)
    print("Prediction:", hypothesis)

    # WER / CER calculation
    sample_wer = wer(reference, hypothesis)
    sample_cer = cer(reference, hypothesis)
    sample_sts = compute_sts(reference, hypothesis)
    sample_ser = compute_ser(reference, hypothesis)
    wer_scores.append(sample_wer)
    cer_scores_all.append(sample_cer)
    sts_scores.append(sample_sts)
    ser_scores.append(sample_ser)

    print(f"WER: {sample_wer:.3f}")
    print(f"CER: {sample_cer:.3f}")
    print(f"STS: {sample_sts:.3f}")
    print(f"SER: {sample_ser:.3f}")

# -----------------------------
# Final Results
# -----------------------------
total = NUM_SAMPLES
average_wer_all = sum(wer_scores) / len(wer_scores) if wer_scores else None
average_cer_all = sum(cer_scores_all) / len(cer_scores_all) if cer_scores_all else None
average_sts_all = sum(sts_scores) / len(sts_scores) if sts_scores else None
average_ser_all = sum(ser_scores) / len(ser_scores) if ser_scores else None

print("\n==== FINAL RESULTS ====")
print(f"Total samples: {total}")
print(f"Average WER (All samples): {average_wer_all:.3f}" if average_wer_all is not None else "Average WER (All samples): N/A")
print(f"Average CER (All samples): {average_cer_all:.3f}" if average_cer_all is not None else "Average CER (All samples): N/A")
print(f"Average STS (All samples): {average_sts_all:.3f}" if average_sts_all is not None else "Average STS (All samples): N/A")
print(f"Average SER (All samples): {average_ser_all:.3f}" if average_ser_all is not None else "Average SER (All samples): N/A")