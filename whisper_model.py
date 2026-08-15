import whisper
import pandas as pd
import tempfile
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download
from jiwer import wer, cer

import warnings
import re
import json
warnings.filterwarnings("ignore", category=UserWarning)
import os
import random

# Force deterministic-ish behavior: limit threads and seed RNGs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

random.seed(0)
np.random.seed(0)
try:
    import torch
    torch.manual_seed(0)
    # prefer deterministic algorithms when available
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

    # Handle "X hundred Y Z"
    if "hundred" in tokens:
        idx = tokens.index("hundred")
        hundreds_digit = tokens[idx - 1]

        # Convert hundreds digit
        if hundreds_digit in ATC_DIGITS:
            hundreds_digit = ATC_DIGITS[hundreds_digit]

        remainder = tokens[idx + 1:]

        # Convert remainder into digit-by-digit
        converted = []
        for token in remainder:
            if token in CARDINALS:
                converted.extend(CARDINALS[token])
            elif token in ATC_DIGITS:
                converted.append(ATC_DIGITS[token])

        # ATC rule: X hundred Y Z → X Y Z
        return " ".join([hundreds_digit] + converted)

    # Handle simple cardinal numbers like "fifty"
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
print("TEST:", normalize_digits("one hundred thirty four six"))

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


def compute_sts(reference: str, hypothesis: str, encoder=None) -> float:
    return semantic_similarity(reference, hypothesis)


def compute_ser(reference: str, hypothesis: str, encoder=None) -> float:
    return semantic_error_rate(reference, hypothesis)


def parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"raw_response": text}


def llm_judge_evaluation(reference: str, hypothesis: str, model_name: str = "gpt-4o-mini") -> dict:
    try:
        import openai
    except Exception as e:
        return {"error": "OpenAI client not installed", "detail": str(e)}

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"error": "OPENAI_API_KEY not set"}

    openai.api_key = key
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert speech recognition evaluation assistant. "
                "Compare the hypothesis to the reference and score the transcription quality."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Reference: \"{reference}\"\n"
                f"Hypothesis: \"{hypothesis}\"\n\n"
                "Evaluate the transcription quality and return JSON only with: "
                "overall_score (0-100), meaning_score (0-100), fluency_score (0-100), "
                "major_error_types, error_summary."
            ),
        },
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message["content"].strip()
        return parse_json_response(text)
    except Exception as e:
        return {"error": "LLM evaluation failed", "detail": str(e)}


def write_audio_bytes_to_wav(audio_bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        return f.name


# -----------------------------
# Download dataset parquet
# -----------------------------
print("Downloading parquet file...")
parquet_path = hf_hub_download(
    repo_id="Jzuluaga/atcosim_corpus",
    filename="data/train-00000-of-00004-c1d7fb31dcbf644a.parquet",
    repo_type="dataset"
)

df = pd.read_parquet(parquet_path)
print(f"Loaded {len(df)} samples")

# -----------------------------
# Load Whisper model
# -----------------------------
print("Loading Whisper model...")
model = whisper.load_model("medium")

# -----------------------------
# Evaluation
# -----------------------------
NUM_SAMPLES = 500  # increase later (e.g. 100+)

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

    # Transcribe directly from the raw WAV
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        temperature=0
    )

    hypothesis = result["text"].lower().strip()
    hypothesis = normalize_text(hypothesis)
    hypothesis = normalize_digits(hypothesis)


    print("Reference:  ", reference)
    print("Prediction:", hypothesis)

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