#!/usr/bin/env python3
"""
rescore.py — recompute evaluation metrics from saved transcripts.

Reads a CSV written by the Whisper / wav2vec2 / Vosk scripts (columns:
id, raw_ref, raw_hyp) and recomputes all metrics with a corrected text
normalizer. Because the raw model output is stored, changing the metrics
never requires re-running a model.

Usage
-----
  python rescore.py transcripts/whisper_medium_bandpass.csv
  python rescore.py transcripts/*.csv                 # all models at once
  python rescore.py file.csv --compare                # old vs new normalizer
  python rescore.py file.csv --no-icao                # strict digit scoring

Every run reports TWO scores per model:

  STANDARD        the headline numbers. Compound operator names count as
                  errors ("trans avia" vs "transavia" = 1 substitution +
                  1 insertion). This is standard word-based WER and it is
                  what should be cited.

  NAME-NORMALISED a secondary score with the documented compound-name rules
                  in the COMPOUNDS table applied identically to references
                  AND predictions, for every model. It separates "the model
                  misheard the operator" from "the model wrote the operator
                  as one word". No output is ever hand-corrected.

Reporting both, rather than choosing, is deliberate: the gap between them
is itself a measurement of how much of the error rate is an orthographic
convention rather than a recognition failure.
"""

import re
import sys
import glob
import argparse
from typing import List, Optional

import pandas as pd
import jiwer


# =============================================================================
# NUMBER HANDLING
# =============================================================================
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "tree": 3, "four": 4,
         "five": 5, "fife": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "niner": 9}
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}
MAGNITUDES = {"hundred", "thousand"}
NUMBER_WORDS = set(UNITS) | set(TEENS) | set(TENS) | MAGNITUDES
DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six",
               "seven", "eight", "nine"]


def int_to_words(n: int) -> str:
    """FIX 1: the original returned str(n) for n >= 1000, and the
    [^a-z\\s] filter then deleted it. 'descend at 2000 feet' lost its
    number on both sides of the comparison."""
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return DIGIT_NAMES[n] if n < 10 else \
            {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
             14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
             18: "eighteen", 19: "nineteen"}[n]
    if n < 100:
        t = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
             60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
        return t[n] if n % 10 == 0 else t[n - n % 10] + " " + DIGIT_NAMES[n % 10]
    if n < 1000:
        head = DIGIT_NAMES[n // 100] + " hundred"
        r = n % 100
        return head + (" " + int_to_words(r) if r else "")
    if n < 1_000_000:
        head = int_to_words(n // 1000) + " thousand"
        r = n % 1000
        return head + (" " + int_to_words(r) if r else "")
    head = int_to_words(n // 1_000_000) + " million"
    r = n % 1_000_000
    return head + (" " + int_to_words(r) if r else "")


def _parse_number_run(tokens: List[str]) -> Optional[int]:
    """Parse a run of number words into one integer.

    ATC uses two conventions at once, so both must work:
      bare digit strings concatenate  — 'three four zero'     -> 340
      English numbers accumulate      — 'three hundred fifty' -> 350
    A unit CONCATENATES after another unit but ADDS after a tens word or a
    magnitude. Magnitudes close a group, so 'seven thousand five hundred'
    resolves to 7500 rather than 700500.
    """
    total = current = 0
    prev = None
    for tok in tokens:
        if tok in UNITS:
            v = UNITS[tok]
            current = current * 10 + v if prev in ("unit", None) else current + v
            prev = "unit"
        elif tok in TEENS:
            current += TEENS[tok]
            prev = "teen"
        elif tok in TENS:
            current += TENS[tok]
            prev = "tens"
        elif tok == "hundred":
            current = (current if current else 1) * 100
            prev = "mag"
        elif tok == "thousand":
            total += (current if current else 1) * 1000
            current = 0
            prev = "mag"
        else:
            return None
    return total + current


def _collapse(tokens: List[str]) -> List[str]:
    """FIX 2/3/4: resolve a number run to its value, then re-emit it
    digit-by-digit.

    The original used tokens.index('hundred'), finding only the FIRST
    magnitude word; 'thousand' was absent from NUMBER_WORDS entirely; and
    magnitude words were DELETED, so 'twelve thousand' and 'twelve' both
    became 'one two' (FL120 and 12000 ft scored as identical). It also
    turned 'three hundred fifty' into 'three five' while leaving
    'three five zero' alone, so the two spoken forms of 350 did not match.
    """
    v = _parse_number_run(tokens)
    if v is None:
        return list(tokens)
    return [DIGIT_NAMES[int(c)] for c in str(v)]


def normalize_digits(text: str) -> str:
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        if toks[i] in NUMBER_WORDS:
            j = i
            while j < len(toks) and toks[j] in NUMBER_WORDS:
                j += 1
            out.extend(_collapse(toks[i:j]))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

# FIX 5: ICAO spells 'alfa' and 'juliett'; every model writes 'alpha' and
# 'juliet'. Same word, guaranteed substitution on every occurrence.
SPELLING = {
    "alpha": "alfa", "juliet": "juliett", "xray": "x-ray",
    "aerodrome": "airdrome",
}

# -----------------------------------------------------------------------------
# COMPOUND OPERATOR NAMES
#
# ATCOSIM transcribes these operator names as two tokens where recognisers
# emit one. Under standard word-based WER that is 1 substitution plus 1
# insertion per occurrence, even when the operator was identified correctly
# — a spacing convention, not a recognition failure.
#
# These rules are applied to BOTH the reference and the prediction, for
# EVERY model, with no per-model exceptions and no hand-editing of any
# output. They drive the secondary NAME-NORMALISED score only; the headline
# STANDARD score always leaves them as errors.
#
# Each entry is bidirectional in effect: the two-token reference form and
# the one-token predicted form both collapse to the same string, so the
# rule cannot favour either side.
#
# Adding to this table changes reported numbers, so additions should be
# justified against the ATCOSIM transcription guidelines, not against
# observed model output — picking rules by looking at which ones help is
# how a normaliser quietly becomes a correction.
# -----------------------------------------------------------------------------
COMPOUNDS = [
    (r"\btrans avia\b", "transavia"),
    (r"\bspeed bird\b", "speedbird"),
    (r"\bswiss air\b", "swissair"),
    (r"\begypt air\b", "egyptair"),
    (r"\btun air\b", "tunair"),
    (r"\bgulf air\b", "gulfair"),
    (r"\baero lloyd\b", "aerolloyd"),
    (r"\bhapag lloyd\b", "hapaglloyd"),
    (r"\bbee med\b", "beemed"),
    (r"\bair tours\b", "airtours"),
    (r"\bsobel air\b", "sobelair"),
    (r"\bbel star\b", "belstar"),
]
# -----------------------------------------------------------------------------

ABBREV = [
    (r"\bfl\b", "flight level"),
]


def normalize_text(text: str) -> str:
    """FIX 6: decimals and letter-digit boundaries handled BEFORE the
    [^a-z\\s] filter.

    Whisper writes frequencies as '125.75'. Stripping the dot fused the
    halves into one number run that parsed as 200 — so every frequency in
    the corpus was scored against nonsense. It also writes 'FL270', which
    became 'fltwo seventy'.
    """
    t = text.lower()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)              # 12,000 -> 12000
    t = re.sub(r"(\d)\s*\.\s*(\d)", r"\1 decimal \2", t)   # 125.75
    t = re.sub(r"(?<=[a-z])(?=\d)", " ", t)                # FL270 -> FL 270
    t = re.sub(r"\d+", lambda m: int_to_words(int(m.group(0))), t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize(text: str, icao: bool = True, compounds: bool = False) -> str:
    t = normalize_text(text)
    for pat, rep in ABBREV:
        t = re.sub(pat, rep, t)
    t = " ".join(SPELLING.get(w, w) for w in t.split())
    if compounds:
        for pat, rep in COMPOUNDS:
            t = re.sub(pat, rep, t)
    if icao:
        t = normalize_digits(t)
    return t


# --- the ORIGINAL normalizer, kept for --compare -----------------------------
_OLD_ATC = {"zero": "zero", "one": "one", "two": "two", "three": "three",
            "tree": "three", "four": "four", "five": "five", "fife": "five",
            "six": "six", "seven": "seven", "eight": "eight", "nine": "nine",
            "niner": "nine"}
_OLD_CARD = {"ten": ["one", "zero"], "eleven": ["one", "one"],
             "twelve": ["one", "two"], "thirteen": ["one", "three"],
             "fourteen": ["one", "four"], "fifteen": ["one", "five"],
             "sixteen": ["one", "six"], "seventeen": ["one", "seven"],
             "eighteen": ["one", "eight"], "nineteen": ["one", "nine"],
             "twenty": ["two"], "thirty": ["three"], "forty": ["four"],
             "fifty": ["five"], "sixty": ["six"], "seventy": ["seven"],
             "eighty": ["eight"], "ninety": ["nine"]}
_OLD_NUM = set(_OLD_ATC) | set(_OLD_CARD) | {"hundred"}


def _old_int_to_words(n):
    W = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
         6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
         11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
         15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
         19: "nineteen"}
    T = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
         70: "seventy", 80: "eighty", 90: "ninety"}
    if n < 20:
        return W[n]
    if n < 100:
        return T[n] if n in T else T[n - n % 10] + " " + W[n % 10]
    if n < 1000:
        r = n % 100
        return W[n // 100] + " hundred" + (" " + _old_int_to_words(r) if r else "")
    return str(n)                      # the >= 1000 bug


def old_normalize(text):
    t = text.lower()
    t = re.sub(r"\d+", lambda m: _old_int_to_words(int(m.group(0))), t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks, out, i = t.split(), [], 0
    while i < len(toks):
        if toks[i] in _OLD_NUM:
            j = i + 1
            while j < len(toks) and toks[j] in _OLD_NUM:
                j += 1
            ph = toks[i:j]
            if "hundred" in ph:
                k = ph.index("hundred")
                hd = _OLD_ATC.get(ph[k - 1], ph[k - 1])
                conv = []
                for tok in ph[k + 1:]:
                    if tok in _OLD_CARD:
                        conv.extend(_OLD_CARD[tok])
                    elif tok in _OLD_ATC:
                        conv.append(_OLD_ATC[tok])
                out.append(" ".join([hd] + conv))
            else:
                conv = []
                for tok in ph:
                    if tok in _OLD_CARD:
                        conv.extend(_OLD_CARD[tok])
                    elif tok in _OLD_ATC:
                        conv.append(_OLD_ATC[tok])
                    else:
                        conv.append(tok)
                out.append(" ".join(conv))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


# =============================================================================
# METRICS
# =============================================================================
AIRLINES = {
    "lufthansa", "sabena", "transwede", "transavia", "speedbird", "swissair",
    "swiss", "malaysian", "constellation", "olympic", "jetset", "georgia",
    "sobelair", "hapag", "lloyd", "alitalia", "klm", "psa", "malta", "viva",
    "aero", "britannia", "belstar", "gulf", "adria", "meridiana", "sunwing",
    "tarom", "egyptair", "scandinavian", "crossair", "saudia", "merair",
    "nato", "tunair", "beemed", "airtours", "trans", "speed", "bird", "air",
    "egypt", "tun", "bee", "med", "sun", "wing",
}

# -----------------------------------------------------------------------------
# Keep AIRLINES in sync with COMPOUNDS automatically.
#
# BUG THIS FIXES: extract_callsign_digits() finds an AIRLINES token and then
# reads the digit run after it, skipping any further AIRLINES tokens in
# between. If one half of a two-word operator name was missing from AIRLINES,
# the scan stopped before reaching the digits and the function returned None
# — so the utterance was dropped from the callsign-accuracy DENOMINATOR
# rather than counted as an error. Silently narrowing the sample is worse
# than scoring it wrong, because nothing in the output reveals it.
#
# Two gaps existed. "avia" was absent, so every "trans avia NNN" callsign was
# skipped under standard scoring. "aerolloyd", "hapaglloyd" and "gulfair"
# were absent, so those broke under name-normalised scoring, where the two
# tokens have been joined.
#
# Deriving both forms from the COMPOUNDS table means adding a compound rule
# can no longer silently shrink the callsign sample.
# -----------------------------------------------------------------------------
for _pat, _rep in COMPOUNDS:
    AIRLINES.add(_rep)                                   # joined form
    AIRLINES.update(_pat.replace(r"\b", "").split())     # both halves

SEMANTIC_KEYWORDS = set(DIGIT_NAMES) | {
    "left", "right", "north", "south", "east", "west",
    "turn", "climb", "descend", "maintain", "contact", "report", "continue",
    "proceed", "hold", "cleared", "clearance", "expedite", "intercept",
    "direct", "vector", "vectors", "cross", "crossing", "monitor", "cancel",
    "cancelled", "standby", "confirm", "readback", "say", "again", "keep",
    "until", "reaching", "request", "heading", "course", "fix", "waypoint",
    "route", "position", "separation", "degrees", "flight", "level",
    "altitude", "rate", "higher", "speed", "knots", "frequency", "decimal",
    "radio", "tower", "ground", "radar", "identified", "identification",
    "information", "atis", "station", "calling", "roger", "runway", "taxi",
    "taxiway", "landing", "takeoff", "approach", "departure", "arrival",
    "arrivals", "enroute", "aircraft", "callsign", "squawk", "transponder",
    "code", "traffic", "affirmative", "negative", "approved", "unable",
    # full NATO alphabet — the original had only 11 letters, so errors on
    # the rest were invisible to the semantic metrics
    "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliett", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "x-ray", "yankee", "zulu",
    "zurich", "rhein", "trasadingen", "dinkelsbuhl", "gotil", "hochwald",
    "prex", "frankfurt", "kempten", "neckar",
} | AIRLINES


def keyword_error_rate(ref: str, hyp: str, keywords=SEMANTIC_KEYWORDS) -> float:
    """Alignment-based SER.

    The original compared SETS of keywords, which discarded repetition and
    ignored order: a reference frequency 'one three two decimal four' has
    'one' twice, but the set keeps one copy, so a dropped digit was free.
    Order was ignored entirely, so 'one two nine' and 'nine two one'
    scored identically — in ATC that is two different aircraft.
    """
    out = jiwer.process_words(ref, hyp)
    R, H = out.references[0], out.hypotheses[0]
    err = tot = 0
    for c in out.alignments[0]:
        if c.type in ("equal", "substitute", "delete"):
            for i in range(c.ref_start_idx, c.ref_end_idx):
                if R[i] in keywords:
                    tot += 1
                    if c.type != "equal":
                        err += 1
        elif c.type == "insert":
            for i in range(c.hyp_start_idx, c.hyp_end_idx):
                if H[i] in keywords:
                    err += 1
    return err / tot if tot else 0.0


def extract_callsign_digits(ref_tokens: List[str]) -> Optional[List[str]]:
    """Return the digit run that identifies the aircraft.

    ATCOSIM utterances lead with the callsign, so this is the first run of
    digit words that follows an airline name or a NATO letter. Returns None
    when the utterance has no callsign (e.g. 'roger', 'thank you').
    """
    for i, t in enumerate(ref_tokens):
        if t in AIRLINES or t in {"alfa", "bravo", "charlie", "delta", "echo",
                                  "foxtrot", "golf", "hotel", "india",
                                  "juliett", "kilo", "lima", "mike",
                                  "november", "oscar", "papa", "quebec",
                                  "romeo", "sierra", "tango", "uniform",
                                  "victor", "whiskey", "x-ray", "yankee",
                                  "zulu"}:
            j = i + 1
            while j < len(ref_tokens) and ref_tokens[j] in AIRLINES:
                j += 1
            k = j
            while k < len(ref_tokens) and ref_tokens[k] in DIGIT_NAMES:
                k += 1
            if k > j:
                return ref_tokens[j:k]
    return None


def callsign_correct(ref: str, hyp: str) -> Optional[bool]:
    """True/False if the reference has a callsign digit sequence and the
    hypothesis contains it contiguously; None if there is no callsign."""
    digits = extract_callsign_digits(ref.split())
    if not digits:
        return None
    h = hyp.split()
    n = len(digits)
    return any(h[i:i + n] == digits for i in range(len(h) - n + 1))


# =============================================================================
# SCORING
# =============================================================================
def score(df: pd.DataFrame, icao=True, compounds=False, use_old=False) -> dict:
    if use_old:
        refs = [old_normalize(str(r)) for r in df.raw_ref]
        hyps = [old_normalize(str(h)) for h in df.raw_hyp]
    else:
        refs = [normalize(str(r), icao, compounds) for r in df.raw_ref]
        hyps = [normalize(str(h), icao, compounds) for h in df.raw_hyp]

    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    refs = [r for r, _ in pairs]
    hyps = [h if h.strip() else "<empty>" for _, h in pairs]

    m = jiwer.process_words(refs, hyps)
    sers = [keyword_error_rate(r, h) for r, h in zip(refs, hyps)]

    cs = [callsign_correct(r, h) for r, h in zip(refs, hyps)]
    cs_scored = [c for c in cs if c is not None]

    return {
        "n": len(refs),
        "ref_words": sum(len(r.split()) for r in refs),
        "wer": m.wer,
        "cer": jiwer.cer(refs, hyps),
        "ser": sum(sers) / len(sers) if sers else 0.0,
        "sub": m.substitutions, "dele": m.deletions,
        "ins": m.insertions, "hits": m.hits,
        "macro_wer": sum(jiwer.wer(r, h) for r, h in zip(refs, hyps)) / len(refs),
        "callsign_acc": (sum(cs_scored) / len(cs_scored)) if cs_scored else None,
        "callsign_n": len(cs_scored),
    }


def show(name: str, s: dict):
    print(f"\n{name}")
    print(f"  utterances        {s['n']}")
    print(f"  reference words   {s['ref_words']}")
    print(f"  WER               {s['wer']:.4f}")
    print(f"  CER               {s['cer']:.4f}")
    print(f"  SER (aligned)     {s['ser']:.4f}")
    if s["callsign_acc"] is not None:
        print(f"  callsign accuracy {s['callsign_acc']:.4f}  "
              f"({s['callsign_n']} utterances with a callsign)")
    print(f"  sub/del/ins       {s['sub']}/{s['dele']}/{s['ins']}")


def show_normalized(std: dict, nrm: dict):
    """The secondary NAME-NORMALISED score, printed beneath the standard one.

    The standard figures are printed directly above by show(), so they are
    not repeated inline here.
    """
    print("\nNAME-NORMALISED (compound operator names joined on BOTH sides)")
    print(f"  WER               {nrm['wer']:.4f}")
    print(f"  CER               {nrm['cer']:.4f}")
    print(f"  SER (aligned)     {nrm['ser']:.4f}")
    rel = (100.0 * (std["wer"] - nrm["wer"]) / std["wer"]) if std["wer"] else 0.0
    print(f"  share of standard WER attributable to name spacing: {rel:.1f}%")


def main():
    p = argparse.ArgumentParser()
    # nargs="*" with a default means running this file with NO arguments
    # (VS Code Run button, double-click) scores every CSV in transcripts/.
    p.add_argument("csvs", nargs="*", default=["transcripts/*.csv"])
    p.add_argument("--no-icao", action="store_true",
                   help="disable ICAO digit collapsing (stricter)")
    p.add_argument("--standard-only", action="store_true",
                   help="suppress the secondary name-normalised score")
    p.add_argument("--compare", action="store_true",
                   help="also score with the original normalizer")
    a = p.parse_args()

    paths = []
    for c in a.csvs:
        paths.extend(sorted(glob.glob(c)) or [c])

    for path in paths:
        df = pd.read_csv(path).fillna("")
        if not {"raw_ref", "raw_hyp"} <= set(df.columns):
            print(f"skipping {path}: needs raw_ref and raw_hyp columns")
            continue

        print("\n" + "=" * 60)
        print(path)
        print("=" * 60)

        if a.compare:
            show("ORIGINAL normalizer", score(df, use_old=True))

        # STANDARD is always the headline score.
        std = score(df, icao=not a.no_icao, compounds=False)
        show("STANDARD (compound operator names count as errors)", std)

        if not a.standard_only:
            nrm = score(df, icao=not a.no_icao, compounds=True)
            show_normalized(std, nrm)
    return 0


if __name__ == "__main__":
    sys.exit(main())