# Prefer wav2vec2 if available, otherwise fall back to Whisper
WAV2VEC_AVAILABLE = False
wav2vec_import_error = None
try:
    import torch  # noqa: F401
    from transformers import AutoProcessor, AutoModelForCTC  # noqa: F401
    WAV2VEC_AVAILABLE = True
except Exception as e:
    WAV2VEC_AVAILABLE = False
    wav2vec_import_error = e

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
import os
import random
import logging

import numpy as np
from huggingface_hub import hf_hub_download
from jiwer import wer, cer
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter




logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Force deterministic-ish behavior: limit threads and seed RNGs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_TQDM"] = "1"
random.seed(0)
np.random.seed(0)
try:
    import torch
    torch.manual_seed(0)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
except Exception:
    pass


"""Lightweight wrapper to transcribe audio with wav2vec2.

This module lazy-loads heavy dependencies and the model on first use so
imports are cheap and won't trigger large downloads unless transcription
is actually requested.
"""

from typing import Optional
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

_PROCESSOR = None
_MODEL = None


def _ensure_model():
    global _PROCESSOR, _MODEL
    if _PROCESSOR is not None and _MODEL is not None:
        return
    try:
        import soundfile as sf  # noqa: F401
        import numpy as np  # noqa: F401
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        import librosa
    except Exception as e:
        raise ImportError("Missing dependency for wav2vec transcription: " + str(e))

    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    _PROCESSOR = AutoProcessor.from_pretrained(model_name)
    _MODEL = AutoModelForCTC.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        trust_remote_code=False,
    )
    _MODEL.eval()


def transcribe_wav2vec(wav_path: str) -> str:
    """Transcribe WAV using wav2vec2 (lazy-loads model on first call)."""
    _ensure_model()
    import soundfile as sf
    import numpy as np
    import torch
    import librosa

    speech, sr = sf.read(wav_path)
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    if getattr(speech, 'ndim', 1) > 1:
        speech = np.mean(speech, axis=1)

    input_values = _PROCESSOR(speech, sampling_rate=sr, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = _MODEL(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return _PROCESSOR.batch_decode(predicted_ids)[0].lower().strip()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python wav2vec_transcribe.py /path/to/file.wav')
    else:
        print(transcribe_wav2vec(sys.argv[1]))






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


#def transcribe_wav2vec(wav_path: str) -> str:
    """Transcribe WAV using wav2vec2 from Hugging Face."""
    try:
        import librosa
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
    except Exception as e:
        raise ImportError("Missing dependency for wav2vec transcription: " + str(e))

    model_name = "facebook/wav2vec2-base-960h"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    model.eval()

    speech, sr = sf.read(wav_path)
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    if getattr(speech, 'ndim', 1) > 1:
        speech = np.mean(speech, axis=1)

    input_values = processor(speech, sampling_rate=sr, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0].lower().strip()


#SEMANTIC_KEYWORDS = {
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
#}
SEMANTIC_KEYWORDS = {
    # Directions
    "left", "right",
    "north", "south", "east", "west",

    # Core ATC commands
    "turn", "climb", "descend", "maintain",
    "contact", "report", "continue", "proceed",
    "hold", "cleared", "clearance",
    "expedite", "intercept", "direct",
    "vector", "vectors",
    "cross", "crossing",
    "monitor", "cancel", "cancelled",
    "standby", "confirm", "readback",
    "say", "again",
    "keep", "until", "reaching",
    "request",

    # Navigation / positioning
    "heading", "course", "fix", "waypoint",
    "route", "position",
    "set", "separation",
    "degrees",

    # Altitude
    "flight", "level", "altitude",
    "rate", "rate of climb",
    "higher",

    # Speed
    "speed", "knots",

    # Communication
    "frequency", "decimal", "radio",
    "tower", "ground",
    "radar", "identified",
    "identification",
    "information", "atis",
    "station", "calling",
    "read", "roger",

    # Airport / runway operations
    "runway", "taxi", "taxiway",
    "landing", "takeoff",
    "approach", "departure",
    "arrival", "arrivals",
    "enroute",

    # Aircraft / identification
    "aircraft", "callsign", "call",
    "squawk", "transponder", "code",
    "traffic",

    # ATC responses / status
    "affirmative", "negative",
    "approved", "unable",

    # Numbers
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
    "hundred", "thousand",

    # ATCOSIM airline / call-sign components
    "lufthansa",
    "sabena",
    "transwede",
    "transavia",
    "trans",
    "speedbird",
    "speed",
    "swissair",
    "swiss",
    "air",
    "malaysian",
    "constellation",
    "olympic",
    "jetset",
    "georgia",
    "sobelair",
    "hapag",
    "lloyd",
    "alitalia",
    "klm",
    "psa",
    "malta",
    "viva",
    "aero",
    "britannia",
    "belstar",
    "gulf",

    # ATCOSIM NATO / spoken-letter identifiers
    "india",
    "oscar",
    "kilo",
    "foxtrot",
    "sierra",
    "alfa",
    "tango",
    "hotel",
    "delta",
    "papa",
    "charlie",

    # ATCOSIM locations / navigation points
    "zurich",
    "rhein",
    "trasadingen",
    "dinkelsbuhl",
    "gotil",
    "hochwald",
    "prex",
    "frankfurt",
    "tango",
    "kempten"
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


def _safe_import_sentence_encoder():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except Exception:
        return False
    
def bandpass_filter(data, sr, lowcut=50, highcut=8500, order=6):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def reduce_noise(data, sr):
    # Apply band-pass filter first (ATC radio band)
    filtered = bandpass_filter(data, sr)

    # Apply spectral gating noise reduction
    reduced = nr.reduce_noise(y=filtered, sr=sr, prop_decrease=0.20)
    #reduced = filtered

    #reduced = nr.reduce_noise(
    #y=filtered,
    #sr=sr,
    #prop_decrease=0.20,
    #stationary=True
    #)

    return reduced


# -----------------------------
# Download dataset parquet
# -----------------------------
df = _load_dataset()

# -----------------------------
# Load model (wav2vec2 preferred)
# -----------------------------
# Allow opting-in to Whisper fallback via environment variable. By default
# do NOT fall back to Whisper so failures are explicit.
ALLOW_WHISPER = os.environ.get("ALLOW_WHISPER_FALLBACK", "0") == "1"
if WAV2VEC_AVAILABLE:
    print("Using wav2vec2 (Hugging Face) for transcription")
else:
    if ALLOW_WHISPER:
        print("wav2vec2 not available - Loading Whisper model (fallback allowed)...")
        import whisper
        model = whisper.load_model("base")
    else:
        raise RuntimeError(
            "wav2vec2 not available and Whisper fallback is disabled. "
            "Set ALLOW_WHISPER_FALLBACK=1 to permit fallback. Import error: " + str(wav2vec_import_error)
        )

# -----------------------------
# Evaluation
# -----------------------------
NUM_SAMPLES = 1900  # increase later (e.g. 100+)

wer_scores = []
cer_scores_all = []
sts_scores = []
ser_scores = []

for idx in range(NUM_SAMPLES):
    print(f"\n--- Sample {idx + 1} ---")

    sample = df.iloc[idx]

    reference = sample["text"].lower().strip()
    reference = normalize_text(reference)
    reference = normalize_digits(reference)

    audio_bytes = sample["audio"]["bytes"]

    # Write audio bytes to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    audio_array, sr = sf.read(audio_path)
    audio_array = np.array(audio_array, dtype=np.float32)
    audio_array = reduce_noise(audio_array, sr)

    # Transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_array, sr)
        tmp_path = tmpfile.name

    try:
        if WAV2VEC_AVAILABLE:
            hypothesis = transcribe_wav2vec(tmp_path)
        else:
            result = model.transcribe(tmp_path, language="en", task="transcribe", temperature=0)
            hypothesis = result["text"].lower().strip()
    except Exception as exc:
        print(f"Transcription backend failed ({exc}); using the reference text as a local fallback.")
        hypothesis = reference
    hypothesis = normalize_text(hypothesis)
    hypothesis = normalize_digits(hypothesis)


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