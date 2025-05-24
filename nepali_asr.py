import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import ffmpeg
import numpy as np
import os
import tempfile

# Load pretrained Nepali Wav2Vec2 model
model_name = "prajin/wav2vec2-large-xlsr-300m-nepali"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()

# Settings
target_sample_rate = 16000
chunk_duration_sec = 30  # split into 30s chunks to avoid memory crashes

def load_audio_ffmpeg(filepath, sample_rate=16000):
    """Load audio and return numpy array using ffmpeg"""
    out, _ = (
        ffmpeg
        .input(filepath, threads=0)
        .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)
        .run(capture_stdout=True, capture_stderr=True)
    )
    audio = np.frombuffer(out, np.float32)
    return audio

def split_audio(audio, chunk_size):
    """Split audio array into multiple chunks"""
    chunks = []
    total_len = len(audio)
    for i in range(0, total_len, chunk_size):
        chunks.append(audio[i:i + chunk_size])
    return chunks

def transcribe(audio_chunk, sample_rate=16000):
    """Transcribe a single audio chunk"""
    inputs = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def transcribe_large_audio(filepath):
    audio = load_audio_ffmpeg(filepath, sample_rate=target_sample_rate)
    chunk_size = target_sample_rate * chunk_duration_sec  # e.g. 480000 samples for 30s

    chunks = split_audio(audio, chunk_size)
    print(f"Total chunks: {len(chunks)}")

    final_transcript = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        text = transcribe(chunk)
        final_transcript.append(text.strip())

    return " ".join(final_transcript)


if __name__ == "__main__":
    # Replace this with your path
    audio_path = "sample.mp3"  # or "sample.wav"
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Transcribing: {audio_path}")
    transcript = transcribe_large_audio(audio_path)
    print("\n📝 Final Transcription:\n")
    print(transcript)
    print("\nTranscription completed.")