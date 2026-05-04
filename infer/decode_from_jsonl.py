import argparse
import json
import os
import re

import torch
import torchaudio

try:
    from decode_audio import MuCodec
except ModuleNotFoundError as exc:
    if exc.name == "model":
        raise ModuleNotFoundError(
            "MuCodec runtime files are missing. `decode_audio.py` requires modules like "
            "`model.py` from the MuCodec repository. Clone/copy MuCodec into `infer/` "
            "as documented, then run this script again."
        ) from exc
    raise


AUDIO_TOKEN_PATTERN = re.compile(r"<AUDIO_(\d+)>")


def extract_codes(text: str) -> torch.Tensor:
    ids = [int(x) for x in AUDIO_TOKEN_PATTERN.findall(text)]
    if not ids:
        raise ValueError("No <AUDIO_...> tokens found in assistant content.")
    return torch.tensor(ids, dtype=torch.long).view(1, 1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Muse JSONL outputs (with <AUDIO_...> tokens) into WAV using MuCodec."
    )
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to generated JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output WAV/PT files.")
    parser.add_argument(
        "--mucodec_ckpt",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "mucodec.pt"),
        help="Path to MuCodec checkpoint (default: infer/ckpt/mucodec.pt).",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Line index in JSONL to decode.")
    parser.add_argument(
        "--assistant_idx",
        type=int,
        default=None,
        help="If set, decode only this assistant-turn index (0-based among assistant messages).",
    )
    parser.add_argument("--steps", type=int, default=50, help="MuCodec denoising steps.")
    parser.add_argument("--duration", type=float, default=40.96, help="Chunk duration used by MuCodec.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for decoding (default cuda:0; use with CUDA_VISIBLE_DEVICES for UUID pinning).",
    )
    parser.add_argument(
        "--save_pt",
        action="store_true",
        help="Also save extracted code tensors as .pt files for reuse with decode_audio.py.",
    )
    parser.add_argument(
        "--concat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically concatenate decoded assistant WAVs when decoding multiple turns (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if args.sample_idx < 0 or args.sample_idx >= len(lines):
        raise IndexError(f"sample_idx {args.sample_idx} out of range [0, {len(lines)-1}]")

    sample = json.loads(lines[args.sample_idx])
    assistants = [m["content"] for m in sample.get("messages", []) if m.get("role") == "assistant"]
    if not assistants:
        raise ValueError("No assistant messages found in selected sample.")

    if args.assistant_idx is not None:
        if args.assistant_idx < 0 or args.assistant_idx >= len(assistants):
            raise IndexError(f"assistant_idx {args.assistant_idx} out of range [0, {len(assistants)-1}]")
        decode_indices = [args.assistant_idx]
    else:
        decode_indices = list(range(len(assistants)))

    mucodec = MuCodec(model_path=args.mucodec_ckpt, layer_num=7, load_main_model=True, device=args.device)
    base = os.path.splitext(os.path.basename(args.input_jsonl))[0]

    decoded_waves = []
    sample_rate = 48000
    for i in decode_indices:
        codes = extract_codes(assistants[i])
        if args.save_pt:
            pt_path = os.path.join(args.output_dir, f"{base}_sample{args.sample_idx}_assistant{i}.pt")
            torch.save(codes, pt_path)
            print(f"saved pt: {pt_path}")

        wave = mucodec.code2sound(
            codes,
            prompt=None,
            duration=args.duration,
            guidance_scale=1.5,
            num_steps=args.steps,
            disable_progress=False,
        )
        decoded_waves.append(wave.detach().cpu())
        wav_path = os.path.join(args.output_dir, f"{base}_sample{args.sample_idx}_assistant{i}.wav")
        torchaudio.save(wav_path, decoded_waves[-1], sample_rate)
        print(f"saved wav: {wav_path}")

    if args.concat and len(decoded_waves) > 1:
        concat_wave = torch.cat(decoded_waves, dim=1)
        concat_path = os.path.join(args.output_dir, f"{base}_sample{args.sample_idx}_concat.wav")
        torchaudio.save(concat_path, concat_wave, sample_rate)
        print(f"saved concat wav: {concat_path}")


if __name__ == "__main__":
    main()
