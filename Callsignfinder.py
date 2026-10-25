#!/usr/bin/env python3
"""
Words right before a number, in ATCOSIM references where no callsign was found.

counts total call signs in each utterance
    python find_operators.py
"""
from collections import Counter
from itertools import zip_longest
from datasets import load_dataset
from rescore_test import normalize, extract_callsign, DIGIT_NAMES

# =============================================================================
# CONFIG
# =============================================================================
SPLITS = ["test", "train"]
TOP    = 60
SKIP   = {"level", "heading", "squawk", "decimal", "runway", "knots", "speed",
          "miles", "feet", "altitude", "frequency", "point", "flight", "and",
          "to", "or", "of", "on", "at", "by", "than", "below", "above", "is",
          "the"}

# =============================================================================
# DATA
# =============================================================================
refs = []
for split in SPLITS:
    ds = load_dataset("Jzuluaga/atcosim_corpus", split=split).select_columns(["text"])
    texts = list(ds["text"])
    print(f"{split:<6}{len(texts)} utterances")
    refs += [normalize(str(t), icao=True, compounds=False) for t in texts]

# =============================================================================
# CANDIDATES
# =============================================================================
digits = set(DIGIT_NAMES)
one, two = Counter(), Counter()
missing = 0
for r in refs:
    t = r.split()
    if not t or extract_callsign(t) is not None:
        continue
    missing += 1
    for i in range(1, len(t)):
        if t[i] in digits and t[i - 1] not in digits and t[i - 1] not in SKIP:
            one[t[i - 1]] += 1
            if i >= 2 and t[i - 2] not in digits:
                two[f"{t[i - 2]} {t[i - 1]}"] += 1

# =============================================================================
# OUTPUT
# =============================================================================
print(f"\nno callsign found in {missing} of {len(refs)} utterances\n")
print(f"  {'word before a number':<26}{'two words before a number'}")
for a, b in zip_longest(one.most_common(TOP), two.most_common(TOP)):
    left = f"{a[0]:<18}{a[1]:>5}" if a else " " * 23
    right = f"{b[0]:<26}{b[1]:>5}" if b else ""
    print(f"  {left}   {right}")