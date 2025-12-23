import os
import io
import time
import wave
import asyncio
import re
from contextlib import asynccontextmanager
from typing import Optional, List
# import cyrtranslit
from LotinKrillYangiLotin import Almashtirish
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

from models import TTSRequest
from infer_utils import process_text

import logging

logger = logging.getLogger()

almashtir = Almashtirish()

print("cuda" if torch.cuda.is_available() else "cpu")

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Configuration
MODEL_NAME = "OvozifyLabs/matcha-tts-uz-v1"
MODEL_FILENAME = "model.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 22050  #  graph in this repo uses 22050 Hz
DEFAULT_TEMPERATURE = float(os.getenv("MATCHA_TEMPERATURE", "0.667"))

# Global model/session
ort_sess: Optional[ort.InferenceSession] = None
is_multi_speaker: bool = False


def _load_session() -> None:
    global ort_sess, is_multi_speaker

    model_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename=MODEL_FILENAME,
        repo_type="model",
    )

    providers: List[str] = (
        ["CUDAExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
    )

    ort_sess = ort.InferenceSession(model_path, providers=providers)
    is_multi_speaker = len(ort_sess.get_inputs()) == 4


# Create app via lifespan to avoid deprecated on_event usage and ensure proper startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_session()
    yield

app = FastAPI(title="TTS FastAPI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)


def _synthesize_wav(
    text: str,
    speed: float,
    temperature: float,
) -> np.ndarray:
    
    assert ort_sess is not None, "ONNX sessions not initialized"

    # Preprocess text -> phoneme/id tensor(s)
    processed = process_text(0, " " + text, DEVICE)  # leading space to mimic CLI behavior
    x = processed["x"].squeeze(0)  # [T]
    x_len = processed["x_lengths"].item()

    # Prepare batch of size 1
    x = x.unsqueeze(0)  # [1, T]
    x_np = x.detach().cpu().numpy()
    x_lengths = np.array([x_len], dtype=np.int64)
    scales = np.array([float(temperature), float(speed)], dtype=np.float32)

    inputs = {
        "x": x_np,
        "x_lengths": x_lengths,
        "scales": scales,
    }

    if is_multi_speaker:
        spk_id = 0
        inputs["spks"] = np.array([spk_id], dtype=np.int64)

    
    wavs, wav_lengths = ort_sess.run(None, inputs)  # wavs: [B, T], wav_lengths: [B]
    wav = wavs[0]
    wav_len = int(wav_lengths[0])
    audio = wav[:wav_len]  # trim to actual length

    return audio.astype(np.float32)

@app.post("/tts_wav")
async def tts_wav(request: TTSRequest):
    """
    TTS endpoint using ONNX Runtime inference with chunking for large texts.
    Input: JSON body
    Output: Auto-downloaded WAV file
    """
    text = request.text
    speed = request.speed
    max_chars = request.max_chars
    cross_fade_duration = request.cross_fade_duration

    print(f"Received text: {text}")

    if not text:
        audio_samples = np.zeros(10000, dtype=np.float32)
    else:
        start_time = time.time()
        text = almashtir.Lotinga(text)

        # Split into chunks
        text_chunks = chunk_text(text, max_chars=max_chars)
        print(f"Text split into {len(text_chunks)} chunks")

        generated_waves = []
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
            chunk_audio = await asyncio.get_running_loop().run_in_executor(
                None, _synthesize_wav, chunk, speed, DEFAULT_TEMPERATURE
            )
            generated_waves.append(chunk_audio)

        # Combine waves with cross-fade
        if cross_fade_duration <= 0 or len(generated_waves) <= 1:
            audio_samples = np.concatenate(generated_waves)
        else:
            audio_samples = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave = audio_samples
                next_wave = generated_waves[i]

                cross_fade_samples = int(cross_fade_duration * SAMPLE_RATE)
                cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                if cross_fade_samples <= 0:
                    audio_samples = np.concatenate([prev_wave, next_wave])
                    continue

                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]

                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)

                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                audio_samples = np.concatenate([
                    prev_wave[:-cross_fade_samples],
                    cross_faded_overlap,
                    next_wave[cross_fade_samples:]
                ])

        print(f"TTS inference time: {time.time() - start_time:.3f}s")

    # Resample to 24kHz
    wav = torch.Tensor(audio_samples).unsqueeze(0)
    wav = torchaudio.functional.resample(wav, SAMPLE_RATE, 24000)
    audio_samples = wav.squeeze(0).numpy()

    # float32 -> int16 -> WAV bytes
    audio_samples = np.clip(audio_samples, -1.0, 1.0)
    int_samples = (audio_samples * 32767.0).astype(np.int16)

    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(int_samples.tobytes())
    wav_bytes.seek(0)

    return StreamingResponse(
        wav_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="tts_output.wav"'},
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        factory=False,
        reload=False,
    )