import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


AV_TO_FOUR_CLASS = {"FF": 0, "FR": 1, "RF": 2, "RR": 3}
FOUR_CLASS_NAMES = ["FF", "FR", "RF", "RR"]
PAIR_TO_FOUR_CLASS = {(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}


def av_to_video_label(av_type: str) -> int:
    # 1 means fake video.
    return 1 if av_type in {"FF", "FR"} else 0


def av_to_audio_label(av_type: str) -> int:
    # 1 means fake audio.
    return 1 if av_type in {"FF", "RF"} else 0


class FusionEmbeddingDataset(Dataset):
    def __init__(
        self,
        video_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        video_labels: torch.Tensor,
        audio_labels: torch.Tensor,
        four_class_labels: torch.Tensor,
    ) -> None:
        n = video_embeddings.shape[0]
        assert audio_embeddings.shape[0] == n
        assert video_labels.shape[0] == n
        assert audio_labels.shape[0] == n
        assert four_class_labels.shape[0] == n

        self.video_embeddings = video_embeddings.float()
        self.audio_embeddings = audio_embeddings.float()
        self.video_labels = video_labels.long()
        self.audio_labels = audio_labels.long()
        self.four_class_labels = four_class_labels.long()

    def __len__(self) -> int:
        return self.video_embeddings.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.video_embeddings[idx],
            self.audio_embeddings[idx],
            self.video_labels[idx],
            self.audio_labels[idx],
            self.four_class_labels[idx],
        )


class LearnableFrequencyFilter(nn.Module):
    """
    Lightweight learnable filtering in frequency space.
    It performs rFFT -> element-wise complex filtering -> irFFT.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        freq_bins = d_model // 2 + 1
        self.scale_real = nn.Parameter(torch.ones(freq_bins))
        self.scale_imag = nn.Parameter(torch.zeros(freq_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model)
        x_freq = torch.fft.rfft(x, dim=-1)
        filter_complex = torch.complex(self.scale_real, self.scale_imag)
        x_freq = x_freq * filter_complex
        x_filtered = torch.fft.irfft(x_freq, n=x.shape[-1], dim=-1)
        return x_filtered


class TwoHeadMultimodalFusion(nn.Module):
    def __init__(
        self,
        video_dim: int = 640,
        audio_dim: int = 768,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.video_freq = LearnableFrequencyFilter(d_model)
        self.audio_freq = LearnableFrequencyFilter(d_model)

        # Cross-modality attention-style fusion.
        self.cross_attn_v_to_a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_a_to_v = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fuse_norm = nn.LayerNorm(d_model * 2)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.video_head = nn.Linear(d_model, 2)
        self.audio_head = nn.Linear(d_model, 2)
        self.four_class_head = nn.Linear(d_model, 4)

    def forward(self, video_emb: torch.Tensor, audio_emb: torch.Tensor):
        v = self.video_proj(video_emb)
        a = self.audio_proj(audio_emb)

        v = self.video_freq(v)
        a = self.audio_freq(a)

        # Attention operates on sequence length 1 tokens from each modality.
        v_tok = v.unsqueeze(1)
        a_tok = a.unsqueeze(1)

        v_from_a, _ = self.cross_attn_v_to_a(query=v_tok, key=a_tok, value=a_tok)
        a_from_v, _ = self.cross_attn_a_to_v(query=a_tok, key=v_tok, value=v_tok)

        v_fused = (v_tok + v_from_a).squeeze(1)
        a_fused = (a_tok + a_from_v).squeeze(1)

        fused = torch.cat([v_fused, a_fused], dim=1)
        fused = self.fuse_norm(fused)
        fused = self.fuse_mlp(fused)

        video_logits = self.video_head(v_fused)
        audio_logits = self.audio_head(a_fused)
        four_class_logits = self.four_class_head(fused)
        return video_logits, audio_logits, four_class_logits, v_fused, a_fused


@torch.no_grad()
def _build_labels_from_av_types(av_types: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    video_labels = []
    audio_labels = []
    four_class_labels = []
    for av in av_types:
        if av not in AV_TO_FOUR_CLASS:
            raise ValueError(f"Unexpected av_type '{av}'. Expected one of {list(AV_TO_FOUR_CLASS.keys())}")
        video_labels.append(av_to_video_label(av))
        audio_labels.append(av_to_audio_label(av))
        four_class_labels.append(AV_TO_FOUR_CLASS[av])

    return (
        torch.tensor(video_labels, dtype=torch.long),
        torch.tensor(audio_labels, dtype=torch.long),
        torch.tensor(four_class_labels, dtype=torch.long),
    )


@torch.no_grad()
def _build_four_class_from_two_heads(video_labels: torch.Tensor, audio_labels: torch.Tensor) -> torch.Tensor:
    if video_labels.shape[0] != audio_labels.shape[0]:
        raise ValueError("video_labels and audio_labels length mismatch.")

    out = []
    for v, a in zip(video_labels.tolist(), audio_labels.tolist()):
        key = (int(v), int(a))
        if key not in PAIR_TO_FOUR_CLASS:
            raise ValueError(f"Unexpected (video,audio) label pair: {key}")
        out.append(PAIR_TO_FOUR_CLASS[key])
    return torch.tensor(out, dtype=torch.long)


def _align_by_paths(video_data: Dict, audio_data: Dict):
    video_paths = video_data.get("paths", [])
    audio_paths = audio_data.get("paths", [])

    if not video_paths or not audio_paths:
        raise ValueError("Both embedding files must contain 'paths' for robust alignment.")

    path_to_video_idx = {p: i for i, p in enumerate(video_paths)}
    video_indices = []
    audio_indices = []

    for a_idx, p in enumerate(audio_paths):
        v_idx = path_to_video_idx.get(p)
        if v_idx is not None:
            video_indices.append(v_idx)
            audio_indices.append(a_idx)

    if len(video_indices) == 0:
        raise ValueError("No matched paths found between video and audio embedding files.")

    v_emb = video_data["embeddings"][video_indices]
    a_emb = audio_data["embeddings"][audio_indices]

    video_av = [video_data["av_type"][i] for i in video_indices]
    audio_av = [audio_data["av_type"][i] for i in audio_indices]
    if video_av != audio_av:
        raise ValueError("av_type mismatch between aligned video/audio embedding entries.")

    return v_emb, a_emb, video_av


def load_split(video_embed_path: Path, audio_embed_path: Path, device: torch.device):
    video_data = torch.load(video_embed_path, map_location=device)
    audio_data = torch.load(audio_embed_path, map_location=device)

    v_emb, a_emb, av_types = _align_by_paths(video_data, audio_data)

    video_labels = video_data.get("labels")
    audio_labels = audio_data.get("labels")
    if video_labels is None or audio_labels is None:
        raise ValueError("Both embedding files must contain 'labels'.")

    video_paths = video_data.get("paths", [])
    audio_paths = audio_data.get("paths", [])
    path_to_video_idx = {p: i for i, p in enumerate(video_paths)}
    matched_video_indices = []
    matched_audio_indices = []
    for a_idx, p in enumerate(audio_paths):
        v_idx = path_to_video_idx.get(p)
        if v_idx is not None:
            matched_video_indices.append(v_idx)
            matched_audio_indices.append(a_idx)

    v_lbl = video_labels[matched_video_indices].long().detach().cpu()
    a_lbl = audio_labels[matched_audio_indices].long().detach().cpu()
    c4_lbl = _build_four_class_from_two_heads(v_lbl, a_lbl)

    # Optional sanity check against av_type metadata when available.
    if av_types:
        _, _, c4_from_av = _build_labels_from_av_types(av_types)
        if not torch.equal(c4_lbl, c4_from_av):
            print(
                "Warning: two-head-derived 4-class labels differ from av_type-derived labels. "
                "Using two-head-derived labels from branch embeddings.",
                flush=True,
            )

    return (
        v_emb.detach().cpu(),
        a_emb.detach().cpu(),
        v_lbl,
        a_lbl,
        c4_lbl,
    )


def make_loader(
    video_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    video_labels: torch.Tensor,
    audio_labels: torch.Tensor,
    four_class_labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = FusionEmbeddingDataset(
        video_embeddings=video_embeds,
        audio_embeddings=audio_embeds,
        video_labels=video_labels,
        audio_labels=audio_labels,
        four_class_labels=four_class_labels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def info_nce_cross_modal(video_repr: torch.Tensor, audio_repr: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    video_norm = F.normalize(video_repr, dim=1)
    audio_norm = F.normalize(audio_repr, dim=1)

    logits = (video_norm @ audio_norm.T) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)

    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_t)


def compute_losses(
    video_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    four_logits: torch.Tensor,
    video_labels: torch.Tensor,
    audio_labels: torch.Tensor,
    four_labels: torch.Tensor,
    video_repr: torch.Tensor,
    audio_repr: torch.Tensor,
    contrastive_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_video = F.cross_entropy(video_logits, video_labels)
    loss_audio = F.cross_entropy(audio_logits, audio_labels)
    loss_four = F.cross_entropy(four_logits, four_labels)

    if contrastive_weight > 0:
        loss_contrast = info_nce_cross_modal(video_repr, audio_repr)
    else:
        loss_contrast = torch.zeros((), device=video_logits.device)

    total = loss_video + loss_audio + loss_four + contrastive_weight * loss_contrast

    return total, {
        "loss_video": float(loss_video.detach().cpu()),
        "loss_audio": float(loss_audio.detach().cpu()),
        "loss_four": float(loss_four.detach().cpu()),
        "loss_contrast": float(loss_contrast.detach().cpu()),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    contrastive_weight: float,
    grad_clip_norm: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    n_samples = 0

    c4_preds = []
    c4_labels = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for v, a, v_lbl, a_lbl, c4_lbl in pbar:
        v = v.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        v_lbl = v_lbl.to(device, non_blocking=True)
        a_lbl = a_lbl.to(device, non_blocking=True)
        c4_lbl = c4_lbl.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            v_logits, a_logits, c4_logits, v_repr, a_repr = model(v, a)
            loss, loss_parts = compute_losses(
                v_logits,
                a_logits,
                c4_logits,
                v_lbl,
                a_lbl,
                c4_lbl,
                v_repr,
                a_repr,
                contrastive_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = c4_lbl.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        n_samples += batch_size

        c4_pred = c4_logits.argmax(dim=1)
        c4_preds.append(c4_pred.detach().cpu().numpy())
        c4_labels.append(c4_lbl.detach().cpu().numpy())

        if scheduler is not None:
            scheduler.step()

        running_acc = accuracy_score(
            np.concatenate(c4_labels, axis=0),
            np.concatenate(c4_preds, axis=0),
        )
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            c4_acc=f"{running_acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    c4_pred = np.concatenate(c4_preds, axis=0)
    c4_true = np.concatenate(c4_labels, axis=0)

    return {
        "loss": total_loss / max(n_samples, 1),
        "c4_acc": accuracy_score(c4_true, c4_pred),
        "c4_f1_macro": f1_score(c4_true, c4_pred, average="macro"),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, split_name: str) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    n_samples = 0

    v_preds = []
    a_preds = []
    c4_preds = []

    v_true = []
    a_true = []
    c4_true = []

    pbar = tqdm(loader, desc=f"{split_name}", leave=False)
    for v, a, v_lbl, a_lbl, c4_lbl in pbar:
        v = v.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        v_lbl = v_lbl.to(device, non_blocking=True)
        a_lbl = a_lbl.to(device, non_blocking=True)
        c4_lbl = c4_lbl.to(device, non_blocking=True)

        v_logits, a_logits, c4_logits, v_repr, a_repr = model(v, a)
        loss, _ = compute_losses(
            v_logits,
            a_logits,
            c4_logits,
            v_lbl,
            a_lbl,
            c4_lbl,
            v_repr,
            a_repr,
            contrastive_weight=0.0,
        )

        batch_size = c4_lbl.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        n_samples += batch_size

        v_preds.append(v_logits.argmax(dim=1).detach().cpu().numpy())
        a_preds.append(a_logits.argmax(dim=1).detach().cpu().numpy())
        c4_preds.append(c4_logits.argmax(dim=1).detach().cpu().numpy())

        v_true.append(v_lbl.detach().cpu().numpy())
        a_true.append(a_lbl.detach().cpu().numpy())
        c4_true.append(c4_lbl.detach().cpu().numpy())

    v_pred = np.concatenate(v_preds, axis=0)
    a_pred = np.concatenate(a_preds, axis=0)
    c4_pred = np.concatenate(c4_preds, axis=0)

    v_lbl_all = np.concatenate(v_true, axis=0)
    a_lbl_all = np.concatenate(a_true, axis=0)
    c4_lbl_all = np.concatenate(c4_true, axis=0)

    metrics = {
        "loss": total_loss / max(n_samples, 1),
        "video_head_accuracy": accuracy_score(v_lbl_all, v_pred),
        "audio_head_accuracy": accuracy_score(a_lbl_all, a_pred),
        "four_class_accuracy": accuracy_score(c4_lbl_all, c4_pred),
        "four_class_f1_macro": f1_score(c4_lbl_all, c4_pred, average="macro"),
        "four_class_confusion_matrix": confusion_matrix(c4_lbl_all, c4_pred, labels=[0, 1, 2, 3]).tolist(),
    }
    return metrics


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_val_f1: float,
    best_epoch: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "args": vars(args),
    }
    torch.save(payload, path)


def maybe_resume(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> Tuple[int, float, int]:
    if not checkpoint_path.exists():
        return 1, -1.0, -1

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler_state = ckpt.get("scaler_state_dict")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    sched_state = ckpt.get("scheduler_state_dict")
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)

    start_epoch = int(ckpt["epoch"]) + 1
    best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
    best_epoch = int(ckpt.get("best_epoch", -1))

    return start_epoch, best_val_f1, best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-head + 4-class multimodal fusion trainer on embeddings")
    parser.add_argument("--video-train-embed", type=str, required=True)
    parser.add_argument("--audio-train-embed", type=str, required=True)
    parser.add_argument("--video-val-embed", type=str, required=True)
    parser.add_argument("--audio-val-embed", type=str, required=True)
    parser.add_argument("--video-test-embed", type=str, required=True)
    parser.add_argument("--audio-test-embed", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/outputs_fusion_two_head")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-attn-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading and aligning embeddings...")

    train_split = load_split(Path(args.video_train_embed), Path(args.audio_train_embed), device)
    val_split = load_split(Path(args.video_val_embed), Path(args.audio_val_embed), device)
    test_split = load_split(Path(args.video_test_embed), Path(args.audio_test_embed), device)

    (tr_v, tr_a, tr_v_lbl, tr_a_lbl, tr_c4_lbl) = train_split
    (va_v, va_a, va_v_lbl, va_a_lbl, va_c4_lbl) = val_split
    (te_v, te_a, te_v_lbl, te_a_lbl, te_c4_lbl) = test_split

    print(f"Train: {len(tr_c4_lbl)} | Val: {len(va_c4_lbl)} | Test: {len(te_c4_lbl)}")

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        tr_v,
        tr_a,
        tr_v_lbl,
        tr_a_lbl,
        tr_c4_lbl,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        va_v,
        va_a,
        va_v_lbl,
        va_a_lbl,
        va_c4_lbl,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        te_v,
        te_a,
        te_v_lbl,
        te_a_lbl,
        te_c4_lbl,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = TwoHeadMultimodalFusion(
        video_dim=tr_v.shape[1],
        audio_dim=tr_a.shape[1],
        d_model=args.d_model,
        num_heads=args.num_attn_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    # Step per batch for smoother training on embeddings.
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    latest_ckpt = output_dir / "last_checkpoint.pt"
    best_ckpt = output_dir / "best_checkpoint.pt"

    if args.resume:
        start_epoch, best_val_f1, best_epoch = maybe_resume(
            latest_ckpt,
            model,
            optimizer,
            scaler,
            scheduler,
            device,
        )
        if latest_ckpt.exists():
            print(f"Resumed from {latest_ckpt} at epoch {start_epoch}")
    else:
        start_epoch, best_val_f1, best_epoch = 1, -1.0, -1

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            contrastive_weight=args.contrastive_weight,
            grad_clip_norm=args.grad_clip_norm,
        )

        val_metrics = evaluate(model, val_loader, device, split_name="Val")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_c4_acc={train_metrics['c4_acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_c4_acc={val_metrics['four_class_accuracy']:.4f} | "
            f"val_c4_f1={val_metrics['four_class_f1_macro']:.4f}"
        )

        save_checkpoint(
            path=latest_ckpt,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            best_val_f1=best_val_f1,
            best_epoch=best_epoch,
            args=args,
        )

        if val_metrics["four_class_f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["four_class_f1_macro"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                path=best_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                args=args,
            )
            print(f"Saved new best checkpoint at epoch {epoch} to {best_ckpt}")
        else:
            patience_counter += 1
            print(f"No val F1 improvement for {patience_counter} epoch(s)")

        if patience_counter >= args.early_stop_patience:
            print("Early stopping triggered.")
            break

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    print("Evaluating on test split...")
    test_metrics = evaluate(model, test_loader, device, split_name="Test")

    test_metrics["best_val_f1_macro"] = float(best_val_f1)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["class_names"] = FOUR_CLASS_NAMES

    print("=== Test Metrics ===")
    print(json.dumps(test_metrics, indent=2))

    torch.save(test_metrics, output_dir / "test_metrics.pt")
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved: {output_dir / 'test_metrics.pt'}")
    print(f"Saved: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
