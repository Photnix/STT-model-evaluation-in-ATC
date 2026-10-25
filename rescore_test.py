#!/usr/bin/env python3
"""
Recompute evaluation metrics from saved transcripts.

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
REPEATS = {"double": 2, "triple": 3}
NUMBER_WORDS = set(UNITS) | set(TEENS) | set(TENS) | MAGNITUDES
DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six",
               "seven", "eight", "nine"]


def int_to_words(n: int) -> str:
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
    v = _parse_number_run(tokens)
    if v is None:
        return list(tokens)
    return [DIGIT_NAMES[int(c)] for c in str(v)]


def expand_repeats(text: str) -> str:
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        if toks[i] in REPEATS and i + 1 < len(toks) and toks[i + 1] in UNITS:
            out.extend([toks[i + 1]] * REPEATS[toks[i]])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def normalize_digits(text: str) -> str:
    toks = expand_repeats(text).split()
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
SPELLING = {
    "alpha": "alfa", "juliet": "juliett", "xray": "x-ray",
    "aerodrome": "airdrome",
}

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

PHRASES = [
    (r"\bgood bye\b", "goodbye"),
]

ABBREV = [
    (r"\bfl\b", "flight level"),
]


def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"(\d)\s*\.\s*(\d)", r"\1 decimal \2", t)
    t = re.sub(r"(?<=[a-z])(?=\d)", " ", t)
    t = re.sub(r"\d+", lambda m: int_to_words(int(m.group(0))), t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize(text: str, icao: bool = True, compounds: bool = False) -> str:
    t = normalize_text(text)
    for pat, rep in ABBREV:
        t = re.sub(pat, rep, t)
    t = " ".join(SPELLING.get(w, w) for w in t.split())
    if compounds:
        for pat, rep in COMPOUNDS + PHRASES:
            t = re.sub(pat, rep, t)
    if icao:
        t = normalize_digits(t)
    return t


# =============================================================================
# ORIGINAL NORMALIZER  (--compare; bugs are intentional, do not fix)
# =============================================================================
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
    return str(n)


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

OPERATOR_WORDS = {
    "ltu", "europa", "aerovic", "condor", "algerie", "iberia", "united",
    "turkish", "lauda", "netherlands", "shamrock", "israeli", "austrian",
    "bama", "turkair", "emirates", "leisure", "portugalia", "cotam", "twa",
    "japanair", "spar", "airlink", "swiss", "sata", "midland", "balkan",
    "futura", "jat", "ascot", "monarch", "germania", "berlin", "blue",
    "singa", "centennial",
}

OPERATOR_PAIRS = [
    "air france", "air force", "air berlin", "air inter", "air algerie",
    "air portugal", "belgian airforce", "hamburg air", "jet aviation",
    "french line", "onur air", "lux air", "tunis air", "top swiss",
    "georgia air", "speed way", "british midland", "lauda air", "israeli air",
]

NATO = {"alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliett", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "x-ray", "yankee", "zulu"}

_FRAGMENTS = set()
for _pat, _rep in COMPOUNDS:
    _FRAGMENTS.update(_pat.replace(r"\b", "").split())

OPERATORS_2 = ([tuple(_pat.replace(r"\b", "").split()) for _pat, _ in COMPOUNDS]
               + [tuple(_p.split()) for _p in OPERATOR_PAIRS])
OPERATORS_1 = ((AIRLINES - _FRAGMENTS) | {_rep for _, _rep in COMPOUNDS}
               | NATO | OPERATOR_WORDS)

for _pat, _rep in COMPOUNDS:
    AIRLINES.add(_rep)
    AIRLINES.update(_pat.replace(r"\b", "").split())
AIRLINES |= OPERATOR_WORDS
for _p in OPERATOR_PAIRS:
    AIRLINES.update(_p.split())

SEMANTIC_KEYWORDS = set(DIGIT_NAMES) | NATO | AIRLINES | {
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
    "zurich", "rhein", "trasadingen", "dinkelsbuhl", "gotil", "hochwald",
    "prex", "frankfurt", "kempten", "neckar", "geneva", "milan", "milano",
    "reims", "marseille",
}


def keyword_error_rate(ref: str, hyp: str, keywords=SEMANTIC_KEYWORDS) -> float:
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


def extract_callsign(ref_tokens: List[str]):
    n = len(ref_tokens)
    for i in range(n):
        j = None
        for two in OPERATORS_2:
            if tuple(ref_tokens[i:i + 2]) == two:
                j = i + 2
                break
        if j is None and ref_tokens[i] in OPERATORS_1:
            j = i + 1
        if j is None:
            continue
        k = j
        while k < n and ref_tokens[k] in DIGIT_NAMES:
            k += 1
        if k > j:
            return ref_tokens[i:j], ref_tokens[j:k]
    return None


def extract_callsign_digits(ref_tokens: List[str]) -> Optional[List[str]]:
    cs = extract_callsign(ref_tokens)
    return cs[1] if cs else None


def _contains(hay: List[str], needle: List[str]) -> bool:
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def callsign_correct(ref: str, hyp: str) -> Optional[bool]:
    digits = extract_callsign_digits(ref.split())
    if not digits:
        return None
    return _contains(hyp.split(), digits)


def callsign_full_correct(ref: str, hyp: str) -> Optional[bool]:
    cs = extract_callsign(ref.split())
    if cs is None:
        return None
    op, digits = cs
    return _contains(hyp.split(), list(op) + list(digits))


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

    fu = [callsign_full_correct(r, h) for r, h in zip(refs, hyps)]
    fu_scored = [c for c in fu if c is not None]

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
        "callsign_full": (sum(fu_scored) / len(fu_scored)) if fu_scored else None,
    }


# =============================================================================
# OUTPUT
# =============================================================================
def show(name: str, s: dict):
    print(f"\n{name}")
    print(f"  utterances        {s['n']}")
    print(f"  reference words   {s['ref_words']}")
    print(f"  WER               {s['wer']:.4f}")
    print(f"  CER               {s['cer']:.4f}")
    print(f"  SER (aligned)     {s['ser']:.4f}")
    if s["callsign_acc"] is not None:
        print(f"  callsign digits   {s['callsign_acc']:.4f}  "
              f"({s['callsign_n']} utterances with a callsign)")
        print(f"  callsign full     {s['callsign_full']:.4f}  "
              f"(operator name + digits)")
    print(f"  sub/del/ins       {s['sub']}/{s['dele']}/{s['ins']}")


def show_normalized(std: dict, nrm: dict):
    print("\nNAME-NORMALISED (compound operator names and \"good bye\" joined on BOTH sides)")
    print(f"  reference words   {nrm['ref_words']}")
    print(f"  WER               {nrm['wer']:.4f}")
    print(f"  CER               {nrm['cer']:.4f}")
    print(f"  SER (aligned)     {nrm['ser']:.4f}")
    print(f"  sub/del/ins       {nrm['sub']}/{nrm['dele']}/{nrm['ins']}")


def main():
    p = argparse.ArgumentParser()
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

        std = score(df, icao=not a.no_icao, compounds=False)
        show("STANDARD (compound operator names count as errors)", std)

        if not a.standard_only:
            nrm = score(df, icao=not a.no_icao, compounds=True)
            show_normalized(std, nrm)
    return 0


if __name__ == "__main__":
    sys.exit(main())