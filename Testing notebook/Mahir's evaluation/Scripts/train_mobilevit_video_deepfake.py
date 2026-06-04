import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from tqdm import tqdm

cv2.setNumThreads(0)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class VideoSample:
    path: str
    label: int
    av_type: str
    language: str
    gender: str
    subject_id: str


class VideoDeepfakeDataset(Dataset):
    """Custom dataset that reads mp4 files and returns (frames_tensor, label)."""

    def __init__(
        self,
        samples: List[VideoSample],
        num_frames: int = 16,
        image_size: int = 256,
        clip_seconds: int = 10,
    ) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.clip_seconds = clip_seconds
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, total_frames: int, fps: float) -> torch.Tensor:
        if total_frames <= 0:
            return torch.zeros(self.num_frames, dtype=torch.long)

        usable_frames = total_frames
        if fps > 0:
            max_frames_10s = int(math.floor(fps * self.clip_seconds))
            if max_frames_10s > 0:
                usable_frames = min(total_frames, max_frames_10s)

        if usable_frames <= 1:
            return torch.zeros(self.num_frames, dtype=torch.long)

        return torch.linspace(0, usable_frames - 1, steps=self.num_frames).long()

    def _read_video_frames(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = self._sample_indices(total_frames, fps).tolist()
        unique_sample_indices = set(sample_indices)

        frames: Dict[int, torch.Tensor] = {}
        frame_idx = 0
        max_read = total_frames if total_frames > 0 else int(max(1, fps) * self.clip_seconds)

        while frame_idx < max_read:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx in unique_sample_indices:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames[frame_idx] = self.transform(frame_rgb)
            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            # Fallback to a black frame tensor if video decoding fails unexpectedly.
            blank = torch.zeros(3, 256, 256, dtype=torch.float32)
            return blank.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        fallback_frame = next(iter(frames.values()))
        selected = [frames.get(i, fallback_frame) for i in sample_indices]

        if len(selected) < self.num_frames:
            selected.extend([selected[-1]] * (self.num_frames - len(selected)))
        elif len(selected) > self.num_frames:
            selected = selected[: self.num_frames]

        return torch.stack(selected, dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        frames = self._read_video_frames(sample.path)
        label = torch.tensor(sample.label, dtype=torch.long)
        return frames, label


def parse_filename_metadata(file_path: Path) -> VideoSample:
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

    # Video authenticity target: fake video=1 for FF/FR, real video=0 for RF/RR.
    label = 1 if av_type in {"FF", "FR"} else 0

    return VideoSample(
        path=str(file_path),
        label=label,
        av_type=av_type,
        language=language,
        gender=gender,
        subject_id=subject_id,
    )


class MobileViTVideoClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for module in self.backbone.features[5:]:
            for param in module.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 2),
        )

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # frames: (batch, T, 3, H, W)
        bsz, num_frames, c, h, w = frames.shape
        flat_frames = frames.view(bsz * num_frames, c, h, w)

        features = self.backbone.features(flat_frames)
        pooled = self.backbone.avgpool(features)
        per_frame_embed = torch.flatten(pooled, 1)

        per_frame_embed = per_frame_embed.view(bsz, num_frames, -1)
        clip_embed = per_frame_embed.mean(dim=1)
        logits = self.classifier(clip_embed)  # (batch, 2)
        return logits, clip_embed


def discover_samples(dataset_root: Path) -> List[VideoSample]:
    samples: List[VideoSample] = []
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


def resolve_dataset_root(dataset_root: Path) -> Path:
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")

    if any(dataset_root.rglob("*.mp4")) or any(dataset_root.rglob("*.MP4")):
        return dataset_root

    # Kaggle datasets are often nested by one extra folder after extraction/upload.
    for child in dataset_root.iterdir():
        if child.is_dir() and (any(child.rglob("*.mp4")) or any(child.rglob("*.MP4"))):
            return child

    raise RuntimeError(f"Could not find mp4 files under: {dataset_root}")


def split_samples(
    samples: List[VideoSample],
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    seed: int = 42,
) -> Tuple[List[VideoSample], List[VideoSample], List[VideoSample]]:
    rng = random.Random(seed)

    # Split by subject_id to avoid leakage between train/val/test.
    # This keeps all clips from the same subject in the same split.
    by_group: Dict[str, List[VideoSample]] = {}
    samples = sorted(samples, key=lambda s: (s.subject_id, s.path))
    for s in samples:
        by_group.setdefault(s.subject_id, []).append(s)

    train_samples: List[VideoSample] = []
    val_samples: List[VideoSample] = []
    test_samples: List[VideoSample] = []

    for group_key in sorted(by_group.keys()):
        group = by_group[group_key]
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


def cap_samples(samples: List[VideoSample], max_count: int, seed: int) -> List[VideoSample]:
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
    for frames, labels in pbar:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits, _ = model(frames)
            targets = one_hot_labels(labels)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

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
    for frames, labels in pbar:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits, _ = model(frames)
            targets = one_hot_labels(labels)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        running_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

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

    for frames, labels in tqdm(loader, desc=f"Export {out_path.stem}", leave=False):
        frames = frames.to(device, non_blocking=True)
        _, clip_embeddings = model(frames)

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
        "embeddings": torch.cat(all_embeddings, dim=0),  # (N, 640)
        "labels": torch.cat(all_labels, dim=0),
        "paths": all_paths,
        "av_type": all_av_types,
        "language": all_languages,
        "gender": all_genders,
        "subject_id": all_subject_ids,
        "split_strategy": "subject_id",
    }
    torch.save(payload, out_path)


def make_loader(
    samples: List[VideoSample],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_frames: int,
    image_size: int,
    clip_seconds: int,
) -> DataLoader:
    dataset = VideoDeepfakeDataset(
        samples=samples,
        num_frames=num_frames,
        image_size=image_size,
        clip_seconds=clip_seconds,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny video deepfake detection trainer")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/kaggle/input/fakebdteen",
        help="Root folder containing mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/outputs",
        help="Directory to save outputs",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-seconds", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Cap train split size for faster debug runs")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Cap val split size for faster debug runs")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Cap test split size for faster debug runs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        f"Label counts (0=real video [RF/RR], 1=fake video [FF/FR]): "
        f"train={[sum(s.label == i for s in train_samples) for i in [0, 1]]}, "
        f"val={[sum(s.label == i for s in val_samples) for i in [0, 1]]}, "
        f"test={[sum(s.label == i for s in test_samples) for i in [0, 1]]}"
    )

    train_loader = make_loader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
        clip_seconds=args.clip_seconds,
    )
    val_loader = make_loader(
        val_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
        clip_seconds=args.clip_seconds,
    )
    test_loader = make_loader(
        test_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
        clip_seconds=args.clip_seconds,
    )
    train_export_loader = make_loader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
        clip_seconds=args.clip_seconds,
    )

    model = MobileViTVideoClassifier(dropout=0.3).to(device)

    # Freeze all backbone weights. Only the head is trainable.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_acc = -1.0
    best_ckpt_path = output_dir / "convnext_tiny_deepfake_best.pt"

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

    # Load best checkpoint before exporting embeddings.
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
