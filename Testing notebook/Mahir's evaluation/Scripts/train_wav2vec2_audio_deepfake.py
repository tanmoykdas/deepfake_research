import argparse
import math
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


@dataclass
class AudioSample:
    path: str
    label: int
    av_type: str
    language: str
    gender: str
    subject_id: str


class AudioDeepfakeDataset(Dataset):
    """Custom dataset that reads mp4 files, extracts audio, and returns (waveform_tensor, label)."""

    def __init__(
        self,
        samples: List[AudioSample],
        processor: Wav2Vec2Processor,
        target_sr: int = 16000,
        duration_sec: int = 10,
        ffmpeg_timeout_sec: int = 45,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.target_sr = target_sr
        self.duration_sec = duration_sec
        self.target_samples = target_sr * duration_sec
        self.ffmpeg_timeout_sec = ffmpeg_timeout_sec

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_audio_waveform(self, video_path: str) -> Optional[np.ndarray]:
        """Extract audio from mp4 using ffmpeg pipe and convert to 16kHz mono."""
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.target_sr),
                "-t",
                str(self.duration_sec),
                "-f",
                "s16le",
                "pipe:1",
            ]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=self.ffmpeg_timeout_sec,
            )
            if not result.stdout:
                return None
            waveform = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return waveform
        except subprocess.TimeoutExpired:
            print(f"ffmpeg timeout ({self.ffmpeg_timeout_sec}s) for: {video_path}")
            return None
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return None

    def _pad_or_truncate(self, waveform: np.ndarray) -> np.ndarray:
        """Pad or truncate waveform to exactly target_samples."""
        if len(waveform) >= self.target_samples:
            return waveform[: self.target_samples]
        else:
            return np.pad(waveform, (0, self.target_samples - len(waveform)), mode="constant")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        waveform = self._extract_audio_waveform(sample.path)

        if waveform is None:
            # Fallback: return silence
            waveform = np.zeros(self.target_samples, dtype=np.float32)

        waveform = self._pad_or_truncate(waveform)

        # Process with Wav2Vec2Processor to get input_values
        inputs = self.processor(waveform, sampling_rate=self.target_sr, return_tensors="pt")
        input_values = inputs["input_values"].squeeze(0)  # Remove batch dim added by processor

        label = torch.tensor(sample.label, dtype=torch.long)
        return input_values, label


class Wav2Vec2AudioClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_values: (batch, 160000) or variable length
        with torch.no_grad():
            outputs = self.backbone(input_values)

        last_hidden_state = outputs.last_hidden_state  # (batch, time_steps, 768)
        clip_embed = last_hidden_state.mean(dim=1)  # (batch, 768)
        logits = self.classifier(clip_embed)  # (batch, 2)
        return logits, clip_embed


def parse_filename_metadata(file_path: Path) -> AudioSample:
    """
    Expected filename format example: FF1B00_00.mp4
      chars 0-1: FF/FR/RF/RR
      char 2: 0=English, 1=Bangla
      char 3: B=Boy, G=Girl
      chars 4-5: subject ID (00-15)
    """
    stem = file_path.stem
    if len(stem) < 6:
        raise ValueError(f"Filename too short to parse metadata: {file_path.name}")

    av_type = stem[0:2].upper()
    lang_code = stem[2]
    gender_code = stem[3].upper()
    subject_id = stem[4:6]

    if av_type not in {"FF", "FR", "RF", "RR"}:
        raise ValueError(f"Unknown AV type in filename: {file_path.name}")

    language = "english" if lang_code == "0" else "bangla" if lang_code == "1" else "unknown"
    gender = "boy" if gender_code == "B" else "girl" if gender_code == "G" else "unknown"

    # Audio authenticity target: fake audio=1 for FF/RF, real audio=0 for FR/RR.
    label = 1 if av_type in {"FF", "RF"} else 0

    return AudioSample(
        path=str(file_path),
        label=label,
        av_type=av_type,
        language=language,
        gender=gender,
        subject_id=subject_id,
    )


def discover_samples(dataset_root: Path) -> List[AudioSample]:
    samples: List[AudioSample] = []
    mp4_files = list(dataset_root.rglob("*.mp4"))
    mp4_files.extend(dataset_root.rglob("*.MP4"))
    if len(mp4_files) == 0:
        raise RuntimeError(f"No mp4 files found in {dataset_root}")

    skipped = 0
    for path in mp4_files:
        try:
            samples.append(parse_filename_metadata(path))
        except ValueError:
            skipped += 1

    if len(samples) == 0:
        raise RuntimeError("No valid samples found after filename parsing.")

    if skipped > 0:
        print(f"Warning: skipped {skipped} files due to unexpected filename format.")

    return samples


def split_samples(
    samples: List[AudioSample],
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    seed: int = 42,
) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
    rng = random.Random(seed)

    by_group: Dict[str, List[AudioSample]] = {}
    for s in samples:
        by_group.setdefault(s.av_type, []).append(s)

    train_samples: List[AudioSample] = []
    val_samples: List[AudioSample] = []
    test_samples: List[AudioSample] = []

    for group in by_group.values():
        rng.shuffle(group)
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_samples.extend(group[:train_end])
        val_samples.extend(group[train_end:val_end])
        test_samples.extend(group[val_end:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def cap_samples(samples: List[AudioSample], max_count: int, seed: int) -> List[AudioSample]:
    if max_count <= 0 or len(samples) <= max_count:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    keep = set(indices[:max_count])
    return [s for i, s in enumerate(samples) if i in keep]


def one_hot_labels(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    total_steps = len(loader)
    first_batch_t0 = time.perf_counter()
    for step_idx, (input_values, labels) in enumerate(pbar, start=1):
        if total == 0:
            first_batch_dt = time.perf_counter() - first_batch_t0
            print(f"First train batch prepared in {first_batch_dt:.2f}s")

        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits, _ = model(input_values)
            targets = one_hot_labels(labels)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * input_values.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / max(total, 1))
        if step_idx == 1 or step_idx % 10 == 0 or step_idx == total_steps:
            print(
                f"Train step {step_idx}/{total_steps} | "
                f"loss={loss.item():.4f} | acc={correct / max(total, 1):.4f}",
                flush=True,
            )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Val", leave=False)
    total_steps = len(loader)
    for step_idx, (input_values, labels) in enumerate(pbar, start=1):
        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits, _ = model(input_values)
            targets = one_hot_labels(labels)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        running_loss += loss.item() * input_values.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step_idx == 1 or step_idx % 10 == 0 or step_idx == total_steps:
            print(
                f"Val step {step_idx}/{total_steps} | "
                f"loss={loss.item():.4f} | acc={correct / max(total, 1):.4f}",
                flush=True,
            )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def export_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
) -> None:
    model.eval()

    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_paths: List[str] = []
    all_av_types: List[str] = []
    all_languages: List[str] = []
    all_genders: List[str] = []
    all_subject_ids: List[str] = []
    sample_ptr = 0
    ordered_samples = loader.dataset.samples

    for input_values, labels in tqdm(loader, desc=f"Export {out_path.stem}", leave=False):
        input_values = input_values.to(device, non_blocking=True)
        _, clip_embeddings = model(input_values)

        all_embeddings.append(clip_embeddings.detach().cpu())
        all_labels.append(labels.clone().cpu())

        batch_size = labels.size(0)
        batch_samples = ordered_samples[sample_ptr : sample_ptr + batch_size]
        all_paths.extend([s.path for s in batch_samples])
        all_av_types.extend([s.av_type for s in batch_samples])
        all_languages.extend([s.language for s in batch_samples])
        all_genders.extend([s.gender for s in batch_samples])
        all_subject_ids.extend([s.subject_id for s in batch_samples])
        sample_ptr += batch_size

    payload = {
        "embeddings": torch.cat(all_embeddings, dim=0),  # (N, 768)
        "labels": torch.cat(all_labels, dim=0),
        "paths": all_paths,
        "av_type": all_av_types,
        "language": all_languages,
        "gender": all_genders,
        "subject_id": all_subject_ids,
    }
    torch.save(payload, out_path)


def make_loader(
    samples: List[AudioSample],
    processor: Wav2Vec2Processor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    ffmpeg_timeout_sec: int,
) -> DataLoader:
    dataset = AudioDeepfakeDataset(
        samples=samples,
        processor=processor,
        target_sr=16000,
        duration_sec=10,
        ffmpeg_timeout_sec=ffmpeg_timeout_sec,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7:
            gpu_name = torch.cuda.get_device_name(0)
            print(
                f"CUDA GPU detected ({gpu_name}, sm_{major}{minor}) but unsupported by this runtime. "
                "Falling back to CPU."
            )
            return torch.device("cpu")
        _ = torch.tensor([0.0], device="cuda")
        return torch.device("cuda")
    except Exception as e:
        print(f"CUDA unavailable at runtime ({e}). Falling back to CPU.")
        return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wav2Vec2-Base audio deepfake detection trainer")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/kaggle/input/fakebdteen",
        help="Root folder containing mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/outputs_audio",
        help="Directory to save outputs",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ffmpeg-timeout", type=int, default=45)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Cap train split size for faster debug runs")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Cap val split size for faster debug runs")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Cap test split size for faster debug runs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_dataset_root(dataset_root: Path) -> Path:
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")

    if any(dataset_root.rglob("*.mp4")) or any(dataset_root.rglob("*.MP4")):
        return dataset_root

    for child in dataset_root.iterdir():
        if child.is_dir() and (any(child.rglob("*.mp4")) or any(child.rglob("*.MP4"))):
            return child

    raise RuntimeError(f"Could not find mp4 files under: {dataset_root}")


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = select_device(force_cpu=args.force_cpu)
    print(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root(Path(args.dataset_root))
    print(f"Using dataset root: {dataset_root}")

    all_samples = discover_samples(dataset_root)
    train_samples, val_samples, test_samples = split_samples(
        all_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_samples = cap_samples(train_samples, args.max_train_samples, args.seed)
    val_samples = cap_samples(val_samples, args.max_val_samples, args.seed + 1)
    test_samples = cap_samples(test_samples, args.max_test_samples, args.seed + 2)

    print(f"Total videos: {len(all_samples)}")
    print(f"Split sizes -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")
    print(
        f"Label counts (0=real audio [FR/RR], 1=fake audio [FF/RF]): "
        f"train={[sum(s.label == i for s in train_samples) for i in [0, 1]]}, "
        f"val={[sum(s.label == i for s in val_samples) for i in [0, 1]]}, "
        f"test={[sum(s.label == i for s in test_samples) for i in [0, 1]]}"
    )

    model = Wav2Vec2AudioClassifier(dropout=0.3).to(device)
    processor = model.processor

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_samples,
        processor,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        ffmpeg_timeout_sec=args.ffmpeg_timeout,
    )
    val_loader = make_loader(
        val_samples,
        processor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        ffmpeg_timeout_sec=args.ffmpeg_timeout,
    )
    test_loader = make_loader(
        test_samples,
        processor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        ffmpeg_timeout_sec=args.ffmpeg_timeout,
    )
    train_export_loader = make_loader(
        train_samples,
        processor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        ffmpeg_timeout_sec=args.ffmpeg_timeout,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_acc = -1.0
    best_ckpt_path = output_dir / "wav2vec2_base_deepfake_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }
            torch.save(checkpoint, best_ckpt_path)
            print(f"Saved new best checkpoint to: {best_ckpt_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    export_embeddings(model, train_export_loader, device, output_dir / "train_embeddings.pt")
    export_embeddings(model, val_loader, device, output_dir / "val_embeddings.pt")
    export_embeddings(model, test_loader, device, output_dir / "test_embeddings.pt")

    metrics = {
        "best_val_acc": best_val_acc,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
    }
    torch.save(metrics, output_dir / "metrics.pt")
    print(f"Saved metrics to: {output_dir / 'metrics.pt'}")


if __name__ == "__main__":
    main()
