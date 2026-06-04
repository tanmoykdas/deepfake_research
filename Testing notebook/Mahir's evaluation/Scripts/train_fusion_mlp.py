import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FusionEmbeddingDataset(Dataset):
    """Dataset that combines precomputed video and audio embeddings."""

    def __init__(
        self,
        video_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        assert (
            video_embeddings.shape[0] == audio_embeddings.shape[0] == labels.shape[0]
        ), "Mismatch in sample count between video, audio, and labels"

        self.video_embeddings = video_embeddings
        self.audio_embeddings = audio_embeddings
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_emb = self.video_embeddings[idx]
        audio_emb = self.audio_embeddings[idx]
        label = self.labels[idx]
        return (video_emb, audio_emb), label


class FusionMLP(nn.Module):
    def __init__(self, dropout1: float = 0.4, dropout2: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1408, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(128, 2),
        )

    def forward(self, video_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([video_emb, audio_emb], dim=1)  # (batch, 1408)
        logits = self.mlp(fused)  # (batch, 2)
        return logits


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
    for (video_emb, audio_emb), labels in pbar:
        video_emb = video_emb.to(device, non_blocking=True)
        audio_emb = audio_emb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(video_emb, audio_emb)
            targets = one_hot_labels(labels)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
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
    for (video_emb, audio_emb), labels in pbar:
        video_emb = video_emb.to(device, non_blocking=True)
        audio_emb = audio_emb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(video_emb, audio_emb)
            targets = one_hot_labels(labels)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_test_with_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate on test set with full metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    pbar = tqdm(loader, desc="Test Eval", leave=False)
    for (video_emb, audio_emb), labels in pbar:
        video_emb = video_emb.to(device, non_blocking=True)
        audio_emb = audio_emb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(video_emb, audio_emb)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(torch.softmax(logits, dim=1).cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    logits = np.concatenate(all_logits, axis=0)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    auc_roc = roc_auc_score(labels, logits[:, 1])
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "confusion_matrix": cm.tolist(),
    }


def load_embeddings_and_labels(
    video_embed_path: Path, audio_embed_path: Path, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and align video/audio embeddings and labels."""
    video_data = torch.load(video_embed_path, map_location=device)
    audio_data = torch.load(audio_embed_path, map_location=device)

    video_embeds = video_data["embeddings"]  # (N, 640)
    audio_embeds = audio_data["embeddings"]  # (N, 768)
    video_labels = video_data["labels"]  # (N,)
    audio_labels = audio_data["labels"]  # (N,)
    video_paths = video_data.get("paths", [])
    audio_paths = audio_data.get("paths", [])

    if len(video_paths) > 0 and len(audio_paths) > 0:
        # Match by path
        path_to_video_idx = {p: i for i, p in enumerate(video_paths)}
        matched_audio_indices = []
        for audio_path in audio_paths:
            if audio_path in path_to_video_idx:
                matched_audio_indices.append(path_to_video_idx[audio_path])
            else:
                raise ValueError(f"Audio path not found in video paths: {audio_path}")

        video_embeds = video_embeds[matched_audio_indices]
        video_labels = video_labels[matched_audio_indices]

    assert (
        video_embeds.shape[0] == audio_embeds.shape[0]
    ), "Mismatch after alignment"
    assert (
        video_labels == audio_labels
    ).all(), "Label mismatch after alignment"

    return video_embeds, audio_embeds, video_labels, audio_labels


def make_loader(
    video_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = FusionEmbeddingDataset(video_embeds, audio_embeds, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal late fusion MLP trainer")
    parser.add_argument(
        "--video-train-embed",
        type=str,
        required=True,
        help="Path to video train embeddings",
    )
    parser.add_argument(
        "--audio-train-embed",
        type=str,
        required=True,
        help="Path to audio train embeddings",
    )
    parser.add_argument(
        "--video-val-embed",
        type=str,
        required=True,
        help="Path to video val embeddings",
    )
    parser.add_argument(
        "--audio-val-embed",
        type=str,
        required=True,
        help="Path to audio val embeddings",
    )
    parser.add_argument(
        "--video-test-embed",
        type=str,
        required=True,
        help="Path to video test embeddings",
    )
    parser.add_argument(
        "--audio-test-embed",
        type=str,
        required=True,
        help="Path to audio test embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/outputs_fusion",
        help="Directory to save outputs",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    train_video_embeds, train_audio_embeds, train_labels, _ = load_embeddings_and_labels(
        Path(args.video_train_embed), Path(args.audio_train_embed), device
    )
    val_video_embeds, val_audio_embeds, val_labels, _ = load_embeddings_and_labels(
        Path(args.video_val_embed), Path(args.audio_val_embed), device
    )
    test_video_embeds, test_audio_embeds, test_labels, _ = load_embeddings_and_labels(
        Path(args.video_test_embed), Path(args.audio_test_embed), device
    )

    print(
        f"Train: {train_labels.shape[0]}, Val: {val_labels.shape[0]}, Test: {test_labels.shape[0]}"
    )

    # Create loaders
    train_loader = make_loader(
        train_video_embeds,
        train_audio_embeds,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        val_video_embeds, val_audio_embeds, val_labels, batch_size=args.batch_size, shuffle=False
    )
    test_loader = make_loader(
        test_video_embeds,
        test_audio_embeds,
        test_labels,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Build model
    model = FusionMLP(dropout1=0.4, dropout2=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_acc = -1.0
    best_ckpt_path = output_dir / "fusion_mlp_best.pt"

    # Train
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
            print(f"Saved best checkpoint to: {best_ckpt_path}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Load best model and evaluate on test set
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    print("\nEvaluating on test set...")
    test_metrics = evaluate_test_with_metrics(model, test_loader, device)

    print("\n=== Test Set Results ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print("Confusion Matrix:")
    cm = np.array(test_metrics["confusion_matrix"])
    print(cm)

    # Save metrics
    metrics_path = output_dir / "test_metrics.pt"
    torch.save(test_metrics, metrics_path)
    print(f"\nSaved test metrics to: {metrics_path}")

    # Save JSON for easy readability
    json_metrics = test_metrics.copy()
    json_metrics["confusion_matrix"] = json_metrics["confusion_matrix"]
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Saved JSON metrics to: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
