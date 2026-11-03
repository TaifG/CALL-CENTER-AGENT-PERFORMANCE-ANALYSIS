"""Whole-channel ASR with VAD. Human words/timing are never decoding inputs."""

import math, time, hashlib, json, re
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from jiwer import process_words
from .data import read, save, sha

REVISION = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def audio_array(path):
    # Decode WAV directly to avoid the PyAV file-opening interface.
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Expected nonempty, finite, mono audio")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
    return np.ascontiguousarray(waveform, dtype=np.float32)


class Transcriber:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.settings = {
            "language": "en",
            "beam_size": 3,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "word_timestamps": False,
            "temperature": 0.0,
        }

    def transcribe(self, path):
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "audio": sha(path),
                    "revision": REVISION,
                    "settings": self.settings,
                    "loader": "soundfile_resample16k_v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        target = self.cache / (fingerprint + ".json")
        if target.exists():
            return read(target)
        # Load the CPU model only when a transcript is not cached.
        if self.model is None:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                "base.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                revision=REVISION,
            )
        started = time.perf_counter()
        wave = audio_array(path)
        segments, info = self.model.transcribe(wave, **self.settings)
        rows = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "avg_logprob": float(s.avg_logprob),
                "no_speech_prob": float(s.no_speech_prob),
            }
            for s in segments
        ]
        result = {
            "segments": rows,
            "seconds": time.perf_counter() - started,
            "audio_sha256": sha(path),
            "cache_key": fingerprint,
            "model": "base.en",
            "revision": REVISION,
            "settings": self.settings,
            "speech_seconds": info.duration_after_vad,
            "audio_seconds": len(wave) / 16000,
        }
        save(target, result)
        return result

    def call(self, dataset, item):
        cid = item["call_id"]
        turns = []
        measurements = []
        warnings = []
        channel_info = {}
        for role, speaker, prefix in [
            ("agent", "Agent", "A"),
            ("caller", "Customer", "C"),
        ]:
            print("ASR", cid, speaker, flush=True)
            decoded = self.transcribe(dataset.audio(cid, role))
            channel_info[speaker] = decoded
            for i, s in enumerate(decoded["segments"], 1):
                if s["text"]:
                    turns.append(
                        {"turn_id": f"{prefix}{i:03}", "speaker": speaker, **s}
                    )
            if not decoded["segments"]:
                warnings.append(speaker + ":no_speech_decoded")
            if any(s["avg_logprob"] < -1 for s in decoded["segments"]):
                warnings.append(speaker + ":low_asr_log_probability")
            # Reference text is used for evaluation only.
            reference = dataset.references(cid, role)
            if reference["human_complete"] and normalize(reference["human"]):
                reference_text = normalize(reference["human"])
                for method, text in [
                    ("fresh_asr", " ".join(s["text"] for s in decoded["segments"])),
                    ("provided_asr", reference["provided_asr"]),
                ]:
                    p = process_words(reference_text, normalize(text))
                    measurements.append(
                        {
                            "call_id": cid,
                            "speaker": speaker,
                            "split": item["split"],
                            "method": method,
                            "wer": p.wer,
                            "reference_words": p.hits + p.substitutions + p.deletions,
                            "substitutions": p.substitutions,
                            "deletions": p.deletions,
                            "insertions": p.insertions,
                            "reference": reference_text,
                            "hypothesis": normalize(text),
                        }
                    )
            else:
                warnings.append(
                    speaker + ":human_reference_incomplete_excluded_from_WER"
                )
        return {
            "call_id": cid,
            "source": "fresh_whisper_whole_channel",
            "chronology": "channel_order_only; cross-speaker ordering unverified",
            "turns": turns,
            "quality_flags": warnings,
            "audio_info": channel_info,
        }, measurements


def metric_summary(rows):
    result = {}
    for method in ["fresh_asr", "provided_asr"]:
        group = [r for r in rows if r["method"] == method]
        n = sum(r["reference_words"] for r in group)
        result[method] = {
            "channels": len(group),
            "reference_words": n,
            "corpus_wer": (
                sum(
                    r["substitutions"] + r["deletions"] + r["insertions"] for r in group
                )
                / n
                if n
                else None
            ),
        }
    return result
