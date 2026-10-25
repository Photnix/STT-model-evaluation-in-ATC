#!/usr/bin/env python3
"""
Exact versions of every model, dataset and package used.

    python versions.py
"""
import platform
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from huggingface_hub import scan_cache_dir

# =============================================================================
# CONFIG
# =============================================================================
HF_REPOS = [
    "facebook/wav2vec2-large-960h-lv60-self",
    "Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc",
    "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc",
    "jlvdoorn/whisper-medium-atco2-asr",
    "jacktol/whisper-medium.en-fine-tuned-for-ATC",
    "Jzuluaga/atcosim_corpus",
]
VOSK_PATH = r"C:\Users\bliu0\OneDrive\Desktop\vosk stt model\vosk-model-en-us-0.22"
PACKAGES = ["openai-whisper", "transformers", "torch", "vosk", "datasets",
            "huggingface_hub", "jiwer", "pandas"]

# =============================================================================
# HUGGING FACE
# =============================================================================
print("HUGGING FACE  (commit ID, which one is 'main', date last downloaded)")
try:
    cached = {r.repo_id: r for r in scan_cache_dir().repos}
except Exception as e:
    cached = {}
    print(f"  could not read the download cache ({e})")
for repo_id in HF_REPOS:
    r = cached.get(repo_id)
    if r is None:
        print(f"  {repo_id:<58}not found")
        continue
    for rev in sorted(r.revisions, key=lambda x: x.last_modified):
        refs = ", ".join(sorted(rev.refs)) or "-"
        day = datetime.fromtimestamp(rev.last_modified).strftime("%Y-%m-%d")
        print(f"  {repo_id:<58}{rev.commit_hash}  {refs:<6}{day}")

# =============================================================================
# OPENAI WHISPER
# =============================================================================
print("\nOPENAI WHISPER")
try:
    import whisper
    print(f"  medium  {whisper._MODELS['medium']}")
except Exception as e:
    print(f"  not available ({e})")

# =============================================================================
# VOSK
# =============================================================================
print("\nVOSK")
print(f"  {VOSK_PATH}")

# =============================================================================
# PACKAGES
# =============================================================================
print("\nPACKAGES")
print(f"  {'python':<18}{platform.python_version()}")
for p in PACKAGES:
    try:
        v = version(p)
    except PackageNotFoundError:
        v = "not installed"
    print(f"  {p:<18}{v}")