#%%
# %%
from pyannote.audio import Pipeline
import os
import librosa
import soundfile as sf
import numpy as np
from collections import defaultdict

#Testing one autuio 
audio_files = "Data\amicorpus\ES2008a\audio\ES2008a.Mix-Headset.wav"  # replace with your files


# inference on the whole file

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Community-1 open-source speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=key2)

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    output = pipeline(audio_files, hook=hook)  # runs locally
# %%

# === 1. Audio Preprocessing ===
def preprocess_audio(file_path, sr=16000):
    y, orig_sr = librosa.load(file_path, sr=None, mono=True)
    y = librosa.resample(y, orig_sr, sr)
    y = y / np.max(np.abs(y))  # normalize
    out_path = f"processed_{os.path.basename(file_path)}"
    sf.write(out_path, y, sr)
    return out_path

# === 2. Pyannote.audio diarization ===
from pyannote.audio import Pipeline
pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def diarize_pyannote(file_path):
    diarization = pyannote_pipeline(file_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})
    return segments

# === 3. WhisperX diarization (requires installation) ===
# pip install git+https://github.com/m-bain/whisperX.git
# Uncomment below if WhisperX installed
"""
from whisperx import load_model, align
whisper_model, _ = load_model("large-v2", device="cpu")

def diarize_whisperx(file_path):
    result = whisper_model.transcribe(file_path)
    aligned_result = align(result["segments"], file_path, whisper_model, device="cpu")
    segments = []
    for seg in aligned_result:
        segments.append({"speaker": seg['speaker'], "start": seg['start'], "end": seg['end']})
    return segments
"""

# === 4. NeMo diarization (optional) ===
# pip install nemo_toolkit[all]
# See NeMo examples: VAD + embedding + clustering pipeline
# segments = nemo_diarize(file_path)

# === 5. Overlap calculation ===
def compute_overlaps(segments):
    overlaps = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            a, b = segments[i], segments[j]
            if a["speaker"] != b["speaker"] and a["start"] < b["end"] and a["end"] > b["start"]:
                overlaps.append((a, b))
    return overlaps

# === 6. Run pipeline on multiple audio files ===
audio_files = ["meeting1.wav", "meeting2.wav"]  # replace with your files
all_results = defaultdict(dict)

for f in audio_files:
    processed_file = preprocess_audio(f)
    
    # Pyannote
    py_segments = diarize_pyannote(processed_file)
    py_overlaps = compute_overlaps(py_segments)
    all_results[f]["pyannote"] = {"segments": py_segments, "overlaps": py_overlaps}
    
    # WhisperX (uncomment if installed)
    # wx_segments = diarize_whisperx(processed_file)
    # wx_overlaps = compute_overlaps(wx_segments)
    # all_results[f]["whisperx"] = {"segments": wx_segments, "overlaps": wx_overlaps}
    
    # NeMo (add your NeMo function)
    # nm_segments = nemo_diarize(processed_file)
    # nm_overlaps = compute_overlaps(nm_segments)
    # all_results[f]["nemo"] = {"segments": nm_segments, "overlaps": nm_overlaps}

# === 7. Example: print results ===
for f, methods in all_results.items():
    print(f"\nAudio file: {f}")
    for method, data in methods.items():
        print(f"  Method: {method}")
        print(f"    Number of segments: {len(data['segments'])}")
        print(f"    Number of overlaps: {len(data['overlaps'])}")

