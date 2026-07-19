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

    model_name = "facebook/wav2vec2-base-960h"
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
