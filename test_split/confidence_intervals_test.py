#!/usr/bin/env python3
"""
95% bootstrap confidence intervals for WER, and for the WER difference between models.

    python confidence_intervals.py
    python confidence_intervals.py "transcripts_train/*.csv"
"""
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import jiwer
from rescore_test import normalize

# =============================================================================
# CONFIG
# =============================================================================
N_BOOT = 5000
SEED = 0
PAIRS = [
    ("wav2vec2_lv60", "w2v2_lv60_atc_uwbatcc"),
    ("whisper_medium_hf", "whisper_medium_atc_atco2"),
    ("whisper_medium_raw", "whisper_medium_atc_atco2"),
    ("whisper_medium_en_hf", "whisper_medium_en_atc_jacktol"),
]


# =============================================================================
# DATA
# =============================================================================
def model_name(path):
    stem = Path(path).stem
    for suffix in ("_test", "_train"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load(path):
    df = pd.read_csv(path).fillna("")
    rows = {}
    for uid, r, h in zip(df["id"], df["raw_ref"], df["raw_hyp"]):
        ref = normalize(str(r))
        if not ref.strip():
            continue
        hyp = normalize(str(h))
        out = jiwer.process_words(ref, hyp if hyp.strip() else "<empty>")
        errors = out.substitutions + out.deletions + out.insertions
        rows[str(uid)] = (errors, len(ref.split()), not str(h).strip())
    return rows


def session_of(uid):
    parts = uid.split("_")
    return "_".join(parts[1:3]) if len(parts) > 3 else uid


# =============================================================================
# BOOTSTRAP
# =============================================================================
def resample(E, W, groups, rng):
    keys, inv = np.unique(groups, return_inverse=True)
    Eg = np.zeros((len(keys), E.shape[1]))
    Wg = np.zeros(len(keys))
    np.add.at(Eg, inv, E)
    np.add.at(Wg, inv, W)
    out = np.empty((N_BOOT, E.shape[1]))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(keys), len(keys))
        out[b] = Eg[idx].sum(axis=0) / Wg[idx].sum()
    return out, len(keys)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return lo, hi


def verdict(d_utt, d_ses):
    if ci(d_ses)[0] > 0 or ci(d_ses)[1] < 0:
        return "yes"
    if ci(d_utt)[0] > 0 or ci(d_utt)[1] < 0:
        return "unclear"
    return "no"


# =============================================================================
# OUTPUT
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="*", default=["transcripts/*.csv"])
    a = p.parse_args()
    paths = sorted(set(x for pat in a.csvs for x in glob.glob(pat)))
    if not paths:
        sys.exit("No CSVs found. Run from the folder containing transcripts/")

    data = {model_name(pth): load(pth) for pth in paths}
    names = list(data)
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    if any(len(d) != len(ids) for d in data.values()):
        print(f"note: using the {len(ids)} utterances present in every file")

    E = np.array([[data[m][u][0] for m in names] for u in ids], dtype=float)
    W = np.array([data[names[0]][u][1] for u in ids], dtype=float)
    wer = E.sum(axis=0) / W.sum()

    rng = np.random.default_rng(SEED)
    utt, _ = resample(E, W, np.arange(len(ids)), rng)
    ses, n_ses = resample(E, W, np.array([session_of(u) for u in ids]), rng)
    n_spk = len({u.split("_")[1] for u in ids if "_" in u})

    print(f"{len(ids)} utterances, {n_ses} speaker-sessions, {n_spk} speakers, {N_BOOT} bootstrap samples")
    if n_ses < 10:
        print("warning: fewer than 10 sessions, so the session-level intervals are rough")

    order = np.argsort(wer)
    w = max(len(n) for n in names) + 2
    print(f"\nWER WITH 95% CONFIDENCE INTERVALS")
    print(f"  {'model':<{w}}{'WER':>8}{'by utterance':>20}{'by session':>20}{'empty':>8}")
    for i in order:
        m = names[i]
        lu, hu = ci(utt[:, i])
        ls, hs = ci(ses[:, i])
        empty = sum(data[m][u][2] for u in ids)
        print(f"  {m:<{w}}{wer[i]:>8.4f}{f'{lu:.4f}-{hu:.4f}':>20}{f'{ls:.4f}-{hs:.4f}':>20}{empty:>8}")

    print(f"\nNEIGHBOURS IN THE RANKING  (is the better model really better?)")
    print(f"  {'better':<{w}}{'worse':<{w}}{'gap':>8}{'by utterance':>20}{'by session':>20}  different?")
    for i, j in zip(order[:-1], order[1:]):
        du, ds = utt[:, j] - utt[:, i], ses[:, j] - ses[:, i]
        lu, hu = ci(du)
        ls, hs = ci(ds)
        print(f"  {names[i]:<{w}}{names[j]:<{w}}{wer[j] - wer[i]:>8.4f}"
              f"{f'{lu:+.4f} to {hu:+.4f}':>20}{f'{ls:+.4f} to {hs:+.4f}':>20}  {verdict(du, ds)}")

    found = [(g, f) for g, f in PAIRS if g in names and f in names]
    if found:
        print(f"\nMATCHED PAIRS  (relative WER reduction from fine-tuning)")
        print(f"  {'general':<{w}}{'fine-tuned':<{w}}{'reduction':>10}{'by utterance':>20}{'by session':>20}")
        for g, f in found:
            gi, fi = names.index(g), names.index(f)
            ru = (utt[:, gi] - utt[:, fi]) / utt[:, gi]
            rs = (ses[:, gi] - ses[:, fi]) / ses[:, gi]
            lu, hu = ci(ru)
            ls, hs = ci(rs)
            r = (wer[gi] - wer[fi]) / wer[gi]
            print(f"  {g:<{w}}{f:<{w}}{r:>10.1%}{f'{lu:.1%} to {hu:.1%}':>20}{f'{ls:.1%} to {hs:.1%}':>20}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
