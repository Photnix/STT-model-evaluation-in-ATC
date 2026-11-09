#!/usr/bin/env python3
"""
Speaker coverage and per-speaker WER.

    python speakers.py
    python speakers.py transcripts/whisper_medium_raw.csv
"""

import sys
import glob
import argparse
from collections import Counter

import pandas as pd
import jiwer

from rescore_test import normalize


def speaker_of(uid):
    parts = str(uid).split("_")
    return parts[1] if len(parts) > 1 else "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="*", default=["transcripts/*.csv"])
    a = p.parse_args()

    paths = []
    for pat in a.csvs:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    if not paths:
        sys.exit("No CSVs found. Run from the folder containing transcripts/")

    first = pd.read_csv(paths[0]).fillna("")
    counts = Counter(speaker_of(i) for i in first["id"])

    print(f"ATCOSIM has 10 speakers: sm1-sm4, gm1, gm2 (male); gf1, zf1-zf3 (female)\n")
    print(f"speakers present in your {len(first)} utterances: {len(counts)}\n")
    print(f"  {'speaker':<10}{'utterances':>12}{'share':>9}")
    print("  " + "-" * 31)
    for s, n in sorted(counts.items()):
        print(f"  {s:<10}{n:>12}{100.0 * n / len(first):>8.1f}%")

    print(f"\n\nper-speaker WER\n")
    spk = sorted(counts)
    head = "model".ljust(32) + "".join(f"{s:>9}" for s in spk)
    print(head)
    print("-" * len(head))

    for path in paths:
        name = path.split("/")[-1].split("\\")[-1].replace(".csv", "")
        df = pd.read_csv(path).fillna("")
        row = name[:31].ljust(32)
        for s in spk:
            sub = df[[speaker_of(i) == s for i in df["id"]]]
            refs, hyps = [], []
            for r, h in zip(sub["raw_ref"], sub["raw_hyp"]):
                R = normalize(str(r), icao=True, compounds=False)
                H = normalize(str(h), icao=True, compounds=False)
                if R.strip():
                    refs.append(R)
                    hyps.append(H if H.strip() else "<empty>")
            row += f"{jiwer.wer(refs, hyps):>9.4f}" if refs else f"{'—':>9}"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())