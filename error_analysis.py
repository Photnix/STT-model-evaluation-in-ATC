#!/usr/bin/env python3
"""
Per-field error rates for ATC transcripts.
"""

import sys
import glob
import argparse
from collections import Counter, defaultdict

import pandas as pd
import jiwer

try:
    from rescore import normalize, DIGIT_NAMES, AIRLINES, extract_callsign_digits
except ImportError:
    sys.exit(
        "ERROR: cannot import from rescore.py\n"
        "Put this script in the same folder as rescore.py, named exactly\n"
        "'rescore.py' (no spaces in the filename)."
    )

DIGITS = set(DIGIT_NAMES)
NORM = dict(icao=True, compounds=False)
MIN_N = 50


# =============================================================================
# FIELDS
# =============================================================================
ANCHORED = [
    ("flight level",    ["flight", "level"], "after"),
    ("altitude (feet)", ["feet"],            "before"),
    ("heading",         ["heading"],         "after"),
    ("speed (knots)",   ["knots"],           "before"),
    ("runway",          ["runway"],          "after"),
    ("squawk",          ["squawk"],          "after"),
]

ORDER = ["callsign", "frequency", "flight level", "heading",
         "altitude (feet)", "runway", "speed (knots)", "squawk"]


def _contains(hay, needle):
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _after(toks, anchor):
    n, out = len(anchor), []
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == anchor:
            j = k = i + n
            while k < len(toks) and toks[k] in DIGITS:
                k += 1
            if k > j:
                out.append(toks[j:k])
    return out


def _before(toks, anchor):
    n, out = len(anchor), []
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == anchor:
            k = j = i
            while k > 0 and toks[k - 1] in DIGITS:
                k -= 1
            if j > k:
                out.append(toks[k:j])
    return out


def _freqs(toks):
    out, i = [], 0
    while i < len(toks):
        if toks[i] in DIGITS or toks[i] == "decimal":
            j = i
            while j < len(toks) and (toks[j] in DIGITS or toks[j] == "decimal"):
                j += 1
            if "decimal" in toks[i:j]:
                out.append(toks[i:j])
            i = j
        else:
            i += 1
    return out


def ref_fields(rt):
    for label, anchor, side in ANCHORED:
        fn = _after if side == "after" else _before
        for v in fn(rt, anchor):
            yield label, anchor, side, v
    for v in _freqs(rt):
        yield "frequency", None, None, v


def value_in(ht, value):
    if _contains(ht, value):
        return True
    if "decimal" not in value:
        return False
    k = value.index("decimal")
    left, right = value[:k], value[k + 1:]
    for gap in (1, 0):
        n = len(left) + gap + len(right)
        for i in range(len(ht) - n + 1):
            if (ht[i:i + len(left)] == left
                    and ht[i + len(left) + gap:i + n] == right):
                return True
    return False


def labeled_in(ht, anchor, side, value):
    if anchor is None:
        return _contains(ht, value)
    return (_contains(ht, anchor + value) if side == "after"
            else _contains(ht, value + anchor))


# =============================================================================
# LEFT / RIGHT
# =============================================================================
def left_right(ref, hyp):
    out = jiwer.process_words(ref, hyp)
    R, H = out.references[0], out.hypotheses[0]
    opp = {"left": "right", "right": "left"}
    n = correct = swapped = other = deleted = 0

    for c in out.alignments[0]:
        if c.type == "equal":
            for i in range(c.ref_start_idx, c.ref_end_idx):
                if R[i] in opp:
                    n += 1
                    correct += 1
        elif c.type == "substitute":
            for off in range(c.ref_end_idx - c.ref_start_idx):
                r = R[c.ref_start_idx + off]
                if r not in opp:
                    continue
                n += 1
                h = (H[c.hyp_start_idx + off]
                     if c.hyp_start_idx + off < c.hyp_end_idx else None)
                if h == opp[r]:
                    swapped += 1
                else:
                    other += 1
        elif c.type == "delete":
            for i in range(c.ref_start_idx, c.ref_end_idx):
                if R[i] in opp:
                    n += 1
                    deleted += 1
    return n, correct, swapped, other, deleted


# =============================================================================
# SUBSTITUTIONS
# =============================================================================
def classify(w):
    if w in DIGITS or w == "decimal":
        return "number"
    if w in AIRLINES:
        return "operator"
    return "place/station"


def substitutions(pairs):
    subs, classes, dels = Counter(), Counter(), Counter()
    for ref, hyp in pairs:
        out = jiwer.process_words(ref, hyp)
        R, H = out.references[0], out.hypotheses[0]
        for c in out.alignments[0]:
            if c.type == "substitute":
                for off in range(c.ref_end_idx - c.ref_start_idx):
                    r = R[c.ref_start_idx + off]
                    h = (H[c.hyp_start_idx + off]
                         if c.hyp_start_idx + off < c.hyp_end_idx else "")
                    subs[(r, h)] += 1
                    classes[classify(r)] += 1
            elif c.type == "delete":
                for i in range(c.ref_start_idx, c.ref_end_idx):
                    dels[R[i]] += 1
                    classes[classify(R[i])] += 1
    return subs, classes, dels


# =============================================================================
# ANALYSIS
# =============================================================================
def analyse(path, dump_rows, model):
    df = pd.read_csv(path).fillna("")
    if not {"raw_ref", "raw_hyp"} <= set(df.columns):
        print("  SKIPPED — needs raw_ref and raw_hyp columns")
        return None

    ids = df["id"] if "id" in df.columns else range(len(df))
    pairs, keep = [], []
    for i, r, h in zip(ids, df["raw_ref"], df["raw_hyp"]):
        R, H = normalize(str(r), **NORM), normalize(str(h), **NORM)
        if R.strip():
            pairs.append((R, H if H.strip() else "<empty>"))
            keep.append(i)

    tally = defaultdict(lambda: [0, 0, 0])

    for uid, (R, H) in zip(keep, pairs):
        rt, ht = R.split(), H.split()

        for label, anchor, side, v in ref_fields(rt):
            ok = value_in(ht, v)
            t = tally[label]
            t[0] += 1
            t[1] += ok
            t[2] += labeled_in(ht, anchor, side, v)
            if not ok:
                dump_rows.append({
                    "model": model, "id": uid, "field": label,
                    "ref_value": " ".join(v), "hyp": H[:70],
                })

        cd = extract_callsign_digits(rt)
        if cd:
            ok = _contains(ht, list(cd))
            t = tally["callsign"]
            t[0] += 1
            t[1] += ok
            t[2] += ok
            if not ok:
                dump_rows.append({
                    "model": model, "id": uid, "field": "callsign",
                    "ref_value": " ".join(cd), "hyp": H[:70],
                })

    lr = [0] * 5
    for R, H in pairs:
        for i, v in enumerate(left_right(R, H)):
            lr[i] += v

    subs, classes, dels = substitutions(pairs)
    return {"n_utt": len(pairs), "tally": dict(tally), "lr": lr,
            "subs": subs, "classes": classes, "dels": dels}


# =============================================================================
# OUTPUT
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="*", default=["transcripts/*.csv"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--dump", metavar="FILE")
    p.add_argument("--name-normalized", action="store_true")
    a = p.parse_args()

    if a.name_normalized:
        NORM["compounds"] = True

    paths = []
    for pat in a.csvs:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    if not paths:
        sys.exit("No CSVs found. Run from the folder containing transcripts/")

    dump_rows, results = [], {}

    for path in paths:
        name = path.split("/")[-1].split("\\")[-1].replace(".csv", "")
        res = analyse(path, dump_rows, name)
        if not res:
            continue
        results[name] = res

        tot_err = sum(res["classes"].values())
        print(f"\n{'=' * 72}")
        print(f"{name}     {res['n_utt']} utterances   {tot_err} errors")
        print("=" * 72)
        print(f"  {'field':<17}{'n':>6}{'correct':>9}{'wrong':>7}"
              f"{'accuracy':>10}{'labeled':>10}")
        print("  " + "-" * 57)
        for f in ORDER:
            if f not in res["tally"]:
                continue
            n, c, lab = res["tally"][f]
            if n < MIN_N and not a.all:
                continue
            labs = "—" if f == "callsign" else f"{100.0 * lab / n:.1f}%"
            print(f"  {f:<17}{n:>6}{c:>9}{n - c:>7}"
                  f"{100.0 * c / n:>9.1f}%{labs:>10}")

        n, c, sw, ot, de = res["lr"]
        if n:
            print(f"  {'left/right':<17}{n:>6}{c:>9}{n - c:>7}"
                  f"{100.0 * c / n:>9.1f}%{'':>10}   swapped: {sw}")

        if tot_err:
            cl = "   ".join(f"{k} {100.0 * v / tot_err:.1f}%"
                            for k, v in res["classes"].most_common())
            print(f"\n  errors: {cl}")
        top = "  ".join(f"{r}→{h or '(drop)'} {ct}"
                        for (r, h), ct in res["subs"].most_common(a.examples))
        print(f"  top:    {top}")

        if a.verbose:
            print("\n  substitutions:")
            for (r, h), ct in res["subs"].most_common(12):
                print(f"    {r:<22} -> {h if h else '(dropped)':<22} {ct:>4}")
            print("  dropped:")
            for w, ct in res["dels"].most_common(6):
                print(f"    {w:<22} {ct:>4}")

    if len(results) > 1:
        fields = [f for f in ORDER
                  if any(f in r["tally"] and
                         (r["tally"][f][0] >= MIN_N or a.all)
                         for r in results.values())]
        print(f"\n\n{'=' * 72}\nFIELD ACCURACY (%)   value recognised\n{'=' * 72}")
        head = "model".ljust(32) + "".join(f"{f[:12]:>13}" for f in fields) \
               + f"{'left/right':>13}"
        print(head)
        print("-" * len(head))
        for name, r in results.items():
            row = name[:31].ljust(32)
            for f in fields:
                if f in r["tally"]:
                    n, c, _ = r["tally"][f]
                    row += f"{100.0 * c / n:>12.1f} "
                else:
                    row += f"{'—':>12} "
            n, c = r["lr"][0], r["lr"][1]
            row += f"{100.0 * c / n:>12.1f} " if n else f"{'—':>12} "
            print(row)
        print(f"\nn per field: " + "  ".join(
            f"{f}={next(r['tally'][f][0] for r in results.values() if f in r['tally'])}"
            for f in fields) + f"  left/right={results[list(results)[0]]['lr'][0]}")

    if a.dump and dump_rows:
        pd.DataFrame(dump_rows).to_csv(a.dump, index=False)
        print(f"\n{len(dump_rows)} field errors written to {a.dump}")

    return 0


if __name__ == "__main__":
    sys.exit(main())