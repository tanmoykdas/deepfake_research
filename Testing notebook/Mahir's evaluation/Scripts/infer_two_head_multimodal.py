import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from train_fusion_two_head_multimodal import FOUR_CLASS_NAMES, TwoHeadMultimodalFusion


cv2.setNumThreads(0)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAIR_TO_FOUR_CLASS = {(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def sample_frame_indices(total_frames: int, fps: float, num_frames: int, clip_seconds: int) -> torch.Tensor:
    if total_frames <= 0:
        return torch.zeros(num_frames, dtype=torch.long)

    usable_frames = total_frames
    if fps > 0:
        max_frames = int(math.floor(fps * clip_seconds))
        if max_frames > 0:
            usable_frames = min(total_frames, max_frames)

    if usable_frames <= 1:
        return torch.zeros(num_frames, dtype=torch.long)

    return torch.linspace(0, usable_frames - 1, steps=num_frames).long()


def extract_video_embedding(
    video_path: Path,
    video_backbone: torch.nn.Module,
    device: torch.device,
    num_frames: int,
    image_size: int,
    clip_seconds: int,
) -> torch.Tensor:
    transform = build_transform(image_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, fps, num_frames, clip_seconds).tolist()
    indices_set = set(indices)

    frames: Dict[int, torch.Tensor] = {}
    frame_idx = 0
    max_read = total_frames if total_frames > 0 else int(max(1, fps) * clip_seconds)

    while frame_idx < max_read:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in indices_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames[frame_idx] = transform(frame_rgb)
        frame_idx += 1

    cap.release()

    if not frames:
        blank = torch.zeros(3, image_size, image_size, dtype=torch.float32)
        clip = blank.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    else:
        fallback = next(iter(frames.values()))
        selected = [frames.get(i, fallback) for i in indices]
        if len(selected) < num_frames:
            selected.extend([selected[-1]] * (num_frames - len(selected)))
        elif len(selected) > num_frames:
            selected = selected[:num_frames]
        clip = torch.stack(selected, dim=0)

    flat = clip.to(device)

    with torch.no_grad():
        features = video_backbone.features(flat)
        pooled = video_backbone.avgpool(features)
        per_frame = torch.flatten(pooled, 1)

    clip_embed = per_frame.mean(dim=0, keepdim=True)  # (1, 768)
    return clip_embed


def extract_audio_embedding(
    video_path: Path,
    wav2vec2: Wav2Vec2Model,
    processor: Wav2Vec2Processor,
    device: torch.device,
    target_sr: int,
    duration_sec: int,
    ffmpeg_timeout_sec: int,
) -> torch.Tensor:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-t",
        str(duration_sec),
        "-f",
        "s16le",
        "pipe:1",
    ]

    try:
        res = subprocess.run(cmd, check=True, capture_output=True, timeout=ffmpeg_timeout_sec)
        if not res.stdout:
            waveform = np.zeros(target_sr * duration_sec, dtype=np.float32)
        else:
            waveform = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        waveform = np.zeros(target_sr * duration_sec, dtype=np.float32)

    target_samples = target_sr * duration_sec
    if len(waveform) >= target_samples:
        waveform = waveform[:target_samples]
    else:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)), mode="constant")

    input_values = processor(waveform, sampling_rate=target_sr, return_tensors="pt")["input_values"].to(device)

    with torch.no_grad():
        out = wav2vec2(input_values)

    clip_embed = out.last_hidden_state.mean(dim=1)  # (1, 768)
    return clip_embed


def build_fusion_model_from_ckpt(ckpt_path: Path, video_dim: int, audio_dim: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})

    model = TwoHeadMultimodalFusion(
        video_dim=video_dim,
        audio_dim=audio_dim,
        d_model=int(args.get("d_model", 256)),
        num_heads=int(args.get("num_attn_heads", 8)),
        dropout=float(args.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-video inference for two-head multimodal deepfake model")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--fusion-checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")

    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-seconds", type=int, default=5)
    parser.add_argument("--audio-duration-sec", type=int, default=10)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--ffmpeg-timeout", type=int, default=45)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ckpt_path = Path(args.fusion_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt_path}")

    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading pretrained backbones...")

    video_backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).to(device).eval()
    wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    print("Extracting video embedding...")
    v_emb = extract_video_embedding(
        video_path=video_path,
        video_backbone=video_backbone,
        device=device,
        num_frames=args.num_frames,
        image_size=args.image_size,
        clip_seconds=args.clip_seconds,
    )

    print("Extracting audio embedding...")
    a_emb = extract_audio_embedding(
        video_path=video_path,
        wav2vec2=wav2vec2,
        processor=processor,
        device=device,
        target_sr=args.audio_sr,
        duration_sec=args.audio_duration_sec,
        ffmpeg_timeout_sec=args.ffmpeg_timeout,
    )

    print("Loading fusion checkpoint and running prediction...")
    fusion_model = build_fusion_model_from_ckpt(
        ckpt_path=ckpt_path,
        video_dim=v_emb.shape[1],
        audio_dim=a_emb.shape[1],
        device=device,
    )

    with torch.no_grad():
        video_logits, audio_logits, four_logits, _, _ = fusion_model(v_emb, a_emb)

    video_probs = F.softmax(video_logits, dim=1).squeeze(0).cpu().numpy()
    audio_probs = F.softmax(audio_logits, dim=1).squeeze(0).cpu().numpy()
    four_probs = F.softmax(four_logits, dim=1).squeeze(0).cpu().numpy()

    video_pred = int(video_probs.argmax())  # 1=fake video
    audio_pred = int(audio_probs.argmax())  # 1=fake audio
    derived_idx = PAIR_TO_FOUR_CLASS[(video_pred, audio_pred)]

    four_idx = int(four_probs.argmax())
    result = {
        "video_path": str(video_path),
        "video_head": {
            "pred_label": "fake_video" if video_pred == 1 else "real_video",
            "probs": {
                "real_video": float(video_probs[0]),
                "fake_video": float(video_probs[1]),
            },
        },
        "audio_head": {
            "pred_label": "fake_audio" if audio_pred == 1 else "real_audio",
            "probs": {
                "real_audio": float(audio_probs[0]),
                "fake_audio": float(audio_probs[1]),
            },
        },
        "four_class": {
            "pred_index": four_idx,
            "pred_label": FOUR_CLASS_NAMES[four_idx],
            "probs": {name: float(four_probs[i]) for i, name in enumerate(FOUR_CLASS_NAMES)},
            "derived_from_two_heads": FOUR_CLASS_NAMES[derived_idx],
        },
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved result JSON to: {out_path}")


if __name__ == "__main__":
    main()
