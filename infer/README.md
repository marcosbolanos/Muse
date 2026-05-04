# Inference Guide

## 1) Generate token JSONL

```bash
python infer/batch_multi_generate.py \
  --input_path infer/test.jsonl \
  --output_dir ./outputs \
  --ckpt_dir ./models/Muse-0.6b \
  --repetition_penalty 1.1 \
  --batch_size 8
```

This writes `generate_multi_*.jsonl` with `<AUDIO_...>` tokens.

## 2) Decode JSONL to WAV

Prereq: MuCodec repo + checkpoints (`mucodec.pt`, `muq.pt`, `audioldm_48k.pth`) available.

```bash
cd infer
CUDA_VISIBLE_DEVICES=<GPU_UUID> /path/to/MuCodec/.venv/bin/python decode_from_jsonl.py \
  --input_jsonl ../outputs/<your_generate_file>.jsonl \
  --output_dir ../outputs/wavs \
  --sample_idx 0 \
  --device cuda:0
```

Default behavior:
- decodes all assistant turns
- saves per-turn `.wav` and `.pt`
- saves one concatenated file: `*_concat.wav`
