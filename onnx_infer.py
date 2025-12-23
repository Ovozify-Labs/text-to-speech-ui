import argparse
import os
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from infer_utils import process_text

def write_wavs(model, inputs, output_dir, external_vocoder=None):
    if external_vocoder is None:
        print("The provided model has the vocoder embedded in the graph.\nGenerating waveform directly")
        t0 = perf_counter()
        wavs, wav_lengths = model.run(None, inputs)
        infer_secs = perf_counter() - t0
        mel_infer_secs = vocoder_infer_secs = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (wav, wav_length) in enumerate(zip(wavs, wav_lengths)):
        output_filename = output_dir.joinpath(f"output_{i + 1}.wav")
        audio = wav[:wav_length]
        print(f"Writing audio to {output_filename}")
        sf.write(output_filename, audio, 22050, "PCM_24")

    wav_secs = wav_lengths.sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    if mel_infer_secs is not None:
        mel_rtf = mel_infer_secs / wav_secs
        print(f"Matcha RTF: {mel_rtf}")
    if vocoder_infer_secs is not None:
        vocoder_rtf = vocoder_infer_secs / wav_secs
        print(f"Vocoder RTF: {vocoder_rtf}")
    print(f"Overall RTF: {rtf}")


def main():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "model",
        type=str,
        help="ONNX model to use",
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )

    args = parser.parse_args()

    if args.gpu:
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    
    model = ort.InferenceSession(args.model, providers=providers)

    model_inputs = model.get_inputs()

    if args.text:
        text_lines = args.text.splitlines()
    else:
        with open(args.file, encoding="utf-8") as file:
            text_lines = file.read().splitlines()
    
    processed_lines = [process_text(0, line, "cpu") for line in text_lines]

    for i in range(50):
        start_time = perf_counter()

        x = [line["x"].squeeze() for line in processed_lines]
        # Pad
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        x = x.detach().cpu().numpy()
        x_lengths = np.array([line["x_lengths"].item() for line in processed_lines], dtype=np.int64)
        inputs = {
            "x": x,
            "x_lengths": x_lengths,
            "scales": np.array([args.temperature, args.speaking_rate], dtype=np.float32),
        }
        is_multi_speaker = len(model_inputs) == 4
        if is_multi_speaker:
            if args.spk is None:
                args.spk = 0
                warn = "[!] Speaker ID not provided! Using speaker ID 0"
                warnings.warn(warn, UserWarning)
            inputs["spks"] = np.repeat(args.spk, x.shape[0]).astype(np.int64)

        write_wavs(model, inputs, args.output_dir)

        # total time
        total_time = perf_counter() - start_time
        print(f"Total time: {total_time}")


if __name__ == "__main__":
    main()
