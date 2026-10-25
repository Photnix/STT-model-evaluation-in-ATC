#!/usr/bin/env python3
"""
Speakers in the ATCOSIM train and test splits, against the official corpus.

    python split_speakers.py
"""
from collections import Counter
from datasets import load_dataset

# =============================================================================
# CONFIG
# =============================================================================
OFFICIAL = {
    "gf1": 238, "gm1": 384, "gm2": 378,
    "sm1": 1167, "sm2": 1848, "sm3": 808, "sm4": 1162,
    "zf1": 1716, "zf2": 1739, "zf3": 638,
}

# =============================================================================
# DATA
# =============================================================================
counts = {}
for split in ["train", "test"]:
    ds = load_dataset("Jzuluaga/atcosim_corpus", split=split).select_columns(["id"])
    counts[split] = Counter(str(i).split("_")[1] for i in ds["id"])

# =============================================================================
# OUTPUT
# =============================================================================
speakers = sorted(set(OFFICIAL) | set(counts["train"]) | set(counts["test"]))
print(f"{'speaker':<10}{'train':>8}{'test':>8}{'official':>10}{'missing':>9}")
print("-" * 45)
for s in speakers:
    tr, te, of = counts["train"][s], counts["test"][s], OFFICIAL.get(s, 0)
    print(f"{s:<10}{tr:>8}{te:>8}{of:>10}{of - tr - te:>9}")
print("-" * 45)
tr, te = sum(counts["train"].values()), sum(counts["test"].values())
of = sum(OFFICIAL.values())
print(f"{'total':<10}{tr:>8}{te:>8}{of:>10}{of - tr - te:>9}")
print(f"\nspeakers in train: {len(counts['train'])}   in test: {len(counts['test'])}"
      f"   in either: {len(set(counts['train']) | set(counts['test']))} of 10")