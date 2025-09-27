import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
from torchvision import transforms


# =============================
# Utility: Seeding
# =============================
def set_seed(seed: int) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# Dataset
# =============================
class AffectDataset(Dataset):
    def __init__(self, root: Path, split_indices: Optional[List[int]] = None, is_train: bool = True) -> None:
        self.root = root
        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        assert self.images_dir.exists() and self.annotations_dir.exists(), "images/ and annotations/ must exist"

        # Build id list by image stems that have all 4 annotation files
        image_paths = sorted(self.images_dir.glob("*.jpg"))
        ids: List[int] = []
        for p in image_paths:
            stem = p.stem
            try:
                idx = int(stem)
            except Exception:
                continue
            aro = self.annotations_dir / f"{idx}_aro.npy"
            exp = self.annotations_dir / f"{idx}_exp.npy"
            lnd = self.annotations_dir / f"{idx}_lnd.npy"
            val = self.annotations_dir / f"{idx}_val.npy"
            if aro.exists() and exp.exists() and lnd.exists() and val.exists():
                ids.append(idx)

        if split_indices is not None:
            self.ids = [ids[i] for i in split_indices]
        else:
            self.ids = ids

        # Transforms
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.ids[idx]
        img_path = self.images_dir / f"{sample_id}.jpg"
        exp_path = self.annotations_dir / f"{sample_id}_exp.npy"
        val_path = self.annotations_dir / f"{sample_id}_val.npy"
        aro_path = self.annotations_dir / f"{sample_id}_aro.npy"
        # lnd_path = self.annotations_dir / f"{sample_id}_lnd.npy"  # Available but not used in training

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            x = self.tf(im)

        y_exp = int(np.load(exp_path))  # 0..7
        y_val = float(np.load(val_path))  # [-1,1]
        y_aro = float(np.load(aro_path))  # [-1,1]

        return {
            "image": x,
            "exp": torch.tensor(y_exp, dtype=torch.long),
            "val": torch.tensor(y_val, dtype=torch.float32),
            "aro": torch.tensor(y_aro, dtype=torch.float32),
        }


# =============================
# Models
# =============================
class GlobalAvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(F.adaptive_avg_pool2d(x, output_size=1), 1)


class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 8, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "resnet18":
            model = torchvision.models.resnet18(weights=None if not pretrained else torchvision.models.ResNet18_Weights.DEFAULT)
            in_features = model.fc.in_features
            layers = list(model.children())[:-2]  # keep until last conv stage
            self.features = nn.Sequential(*layers)
            self.pool = GlobalAvgPool()
        elif self.backbone_name == "resnet34":
            model = torchvision.models.resnet34(weights=None if not pretrained else torchvision.models.ResNet34_Weights.DEFAULT)
            in_features = model.fc.in_features
            layers = list(model.children())[:-2]
            self.features = nn.Sequential(*layers)
            self.pool = GlobalAvgPool()
        elif self.backbone_name == "efficientnet_b0":
            model = torchvision.models.efficientnet_b0(weights=None if not pretrained else torchvision.models.EfficientNet_B0_Weights.DEFAULT)
            in_features = model.classifier[1].in_features
            self.features = model.features
            self.pool = GlobalAvgPool()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.dropout = nn.Dropout(p=0.2)
        self.exp_head = nn.Linear(in_features, num_classes)
        self.val_head = nn.Linear(in_features, 1)
        self.aro_head = nn.Linear(in_features, 1)

        # Kaiming init for heads when not pretrained
        for m in [self.exp_head, self.val_head, self.aro_head]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.features(x)
        feats = self.pool(feats)
        feats = self.dropout(feats)

        exp_logits = self.exp_head(feats)
        val_pred = self.val_head(feats).squeeze(1)
        aro_pred = self.aro_head(feats).squeeze(1)
        return {"exp": exp_logits, "val": val_pred, "aro": aro_pred}


# =============================
# Metrics (classification)
# =============================
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def f1_score_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row_marginals = cm.sum(axis=1)
    col_marginals = cm.sum(axis=0)
    pe = np.dot(row_marginals, col_marginals) / (n * n)
    denom = 1 - pe
    return float((po - pe) / denom) if denom != 0 else 0.0


def krippendorff_alpha_nominal(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    # Build symmetric coincidence matrix O
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)
        if t == p:
            O[t, t] += 1.0
        else:
            O[t, p] += 1.0
            O[p, t] += 1.0
    Np = O.sum()
    if Np == 0:
        return 0.0
    Do = (Np - np.trace(O)) / Np  # proportion of off-diagonal
    m = O.sum(axis=1)
    De = (np.sum(m * (Np - m))) / (Np * Np)
    denom = De
    return float(1.0 - (Do / denom)) if denom > 0 else 0.0


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # y_true in {0,1}; trapezoidal AUC
    # Sort by score descending
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    P = y_true.sum()
    N = y_true.size - P
    if P == 0 or N == 0:
        return float("nan")
    # Compute TPR/FPR at all thresholds (unique scores)
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / P
    fpr = fps / N
    # Add (0,0) at start
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    # Trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return float(auc)


def _binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Precision-Recall AUC for binary labels
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(y_true.sum(), 1)
    # Add (recall=0, precision=1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    # AUC
    auc = np.trapz(precision, recall)
    return float(auc)


def multiclass_ovr_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Tuple[float, float]:
    # Returns (macro ROC-AUC, macro PR-AUC)
    roc_aucs: List[float] = []
    pr_aucs: List[float] = []
    for c in range(num_classes):
        binary_true = (y_true == c).astype(np.int32)
        scores = y_prob[:, c]
        roc = _binary_roc_auc(binary_true, scores)
        pr = _binary_pr_auc(binary_true, scores)
        if not np.isnan(roc):
            roc_aucs.append(roc)
        if not np.isnan(pr):
            pr_aucs.append(pr)
    roc_macro = float(np.mean(roc_aucs)) if roc_aucs else float("nan")
    pr_macro = float(np.mean(pr_aucs)) if pr_aucs else float("nan")
    return roc_macro, pr_macro


# =============================
# Metrics (regression)
# =============================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else 0.0


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    t = y_true - y_true.mean()
    p = y_pred - y_pred.mean()
    denom = np.sqrt((t ** 2).sum() * (p ** 2).sum())
    return float((t * p).sum() / denom) if denom != 0 else 0.0


def sagr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    sign_true = (y_true >= 0).astype(np.int32)
    sign_pred = (y_pred >= 0).astype(np.int32)
    return float(np.mean(sign_true == sign_pred))


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Concordance Correlation Coefficient
    if y_true.size == 0:
        return 0.0
    mu_x = y_true.mean()
    mu_y = y_pred.mean()
    vx = y_true.var()
    vy = y_pred.var()
    sxy = np.mean((y_true - mu_x) * (y_pred - mu_y))
    denom = vx + vy + (mu_x - mu_y) ** 2
    return float((2 * sxy) / denom) if denom != 0 else 0.0


# =============================
# Training / Evaluation
# =============================
@dataclass
class TrainConfig:
    model_name: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    device: torch.device
    cls_weight: float
    val_weight: float
    aro_weight: float
    pretrained: bool


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    cls_loss_fn = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for batch in loader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        y_exp = batch["exp"].to(cfg.device, non_blocking=True)
        y_val = batch["val"].to(cfg.device, non_blocking=True)
        y_aro = batch["aro"].to(cfg.device, non_blocking=True)

        out = model(images)
        loss_cls = cls_loss_fn(out["exp"], y_exp)
        loss_val = mse(out["val"], y_val)
        loss_aro = mse(out["aro"], y_aro)
        loss = cfg.cls_weight * loss_cls + cfg.val_weight * loss_val + cfg.aro_weight * loss_aro

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, {"loss": avg_loss}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    cls_loss_fn = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    losses: List[float] = []

    all_logits: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_y_exp: List[np.ndarray] = []

    all_val_pred: List[np.ndarray] = []
    all_aro_pred: List[np.ndarray] = []
    all_val_true: List[np.ndarray] = []
    all_aro_true: List[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        y_exp = batch["exp"].to(cfg.device, non_blocking=True)
        y_val = batch["val"].to(cfg.device, non_blocking=True)
        y_aro = batch["aro"].to(cfg.device, non_blocking=True)

        out = model(images)

        loss_cls = cls_loss_fn(out["exp"], y_exp)
        loss_val = mse(out["val"], y_val)
        loss_aro = mse(out["aro"], y_aro)
        loss = cfg.cls_weight * loss_cls + cfg.val_weight * loss_val + cfg.aro_weight * loss_aro
        losses.append(float(loss.item()))

        logits = out["exp"].detach().cpu().numpy()
        probs = F.softmax(out["exp"], dim=1).detach().cpu().numpy()
        y_e = y_exp.detach().cpu().numpy()

        all_logits.append(logits)
        all_probs.append(probs)
        all_y_exp.append(y_e)

        all_val_pred.append(out["val"].detach().cpu().numpy())
        all_aro_pred.append(out["aro"].detach().cpu().numpy())
        all_val_true.append(y_val.detach().cpu().numpy())
        all_aro_true.append(y_aro.detach().cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else 0.0

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 8), dtype=np.float32)
    y_true_cls = np.concatenate(all_y_exp, axis=0) if all_y_exp else np.zeros((0,), dtype=np.int64)
    y_pred_cls = y_prob.argmax(axis=1) if y_prob.size else np.zeros((0,), dtype=np.int64)

    # Classification metrics
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1m = f1_score_macro(y_true_cls, y_pred_cls, num_classes=8)
    kappa = cohen_kappa_score(y_true_cls, y_pred_cls, num_classes=8)
    alpha = krippendorff_alpha_nominal(y_true_cls, y_pred_cls, num_classes=8)
    roc_auc_macro, pr_auc_macro = multiclass_ovr_auc(y_true_cls, y_prob, num_classes=8) if y_prob.size else (float("nan"), float("nan"))

    # Regression metrics
    y_val_t = np.concatenate(all_val_true, axis=0) if all_val_true else np.zeros((0,), dtype=np.float32)
    y_val_p = np.concatenate(all_val_pred, axis=0) if all_val_pred else np.zeros((0,), dtype=np.float32)
    y_aro_t = np.concatenate(all_aro_true, axis=0) if all_aro_true else np.zeros((0,), dtype=np.float32)
    y_aro_p = np.concatenate(all_aro_pred, axis=0) if all_aro_pred else np.zeros((0,), dtype=np.float32)

    metrics: Dict[str, float] = {
        "val/total_loss": avg_loss,
        "cls/acc": acc,
        "cls/f1_macro": f1m,
        "cls/kappa": kappa,
        "cls/alpha": alpha,
        "cls/roc_auc_macro": float(roc_auc_macro) if not (isinstance(roc_auc_macro, float) and math.isnan(roc_auc_macro)) else 0.0,
        "cls/pr_auc_macro": float(pr_auc_macro) if not (isinstance(pr_auc_macro, float) and math.isnan(pr_auc_macro)) else 0.0,
        "valence/rmse": rmse(y_val_t, y_val_p),
        "valence/corr": pearson_corr(y_val_t, y_val_p),
        "valence/sagr": sagr(y_val_t, y_val_p),
        "valence/ccc": ccc(y_val_t, y_val_p),
        "arousal/rmse": rmse(y_aro_t, y_aro_p),
        "arousal/corr": pearson_corr(y_aro_t, y_aro_p),
        "arousal/sagr": sagr(y_aro_t, y_aro_p),
        "arousal/ccc": ccc(y_aro_t, y_aro_p),
    }

    return metrics


def run_training(root: Path, model_name: str, epochs: int, batch_size: int, lr: float, weight_decay: float, num_workers: int, device: torch.device, cls_weight: float, val_weight: float, aro_weight: float, pretrained: bool, val_ratio: float, test_ratio: float, subset: Optional[int], seed: int) -> Dict[str, float]:
    set_seed(seed)

    # Build full id list
    full_ds = AffectDataset(root, is_train=True)
    n = len(full_ds)
    indices = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    # Calculate split sizes
    val_size = max(1, int(n * val_ratio))
    test_size = max(1, int(n * test_ratio))
    train_size = n - val_size - test_size
    
    # Ensure we have at least some training data
    if train_size <= 0:
        train_size = max(1, n - val_size - test_size)
        if train_size <= 0:
            raise ValueError("Not enough data for train/val/test split")
    
    # Create splits
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    if subset is not None and subset > 0:
        train_indices = train_indices[:subset]
        val_indices = val_indices[:max(1, min(subset // 5, len(val_indices)))]
        test_indices = test_indices[:max(1, min(subset // 10, len(test_indices)))]

    train_ds = AffectDataset(root, split_indices=train_indices, is_train=True)
    val_ds = AffectDataset(root, split_indices=val_indices, is_train=False)
    test_ds = AffectDataset(root, split_indices=test_indices, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = MultiTaskModel(model_name, num_classes=8, pretrained=pretrained)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    cfg = TrainConfig(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        device=device,
        cls_weight=cls_weight,
        val_weight=val_weight,
        aro_weight=aro_weight,
        pretrained=pretrained,
    )

    print(f"\n=== Training {model_name} ===")
    print(f"Device: {device} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Epochs: {epochs} | Batch: {batch_size}")
    best_metrics: Dict[str, float] = {}
    best_key = "cls/f1_macro"
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, cfg)
        val_metrics = evaluate(model, val_loader, cfg)

        # Track best on macro-F1 for classification task as primary
        score = val_metrics.get(best_key, -1.0)
        if score > best_val:
            best_val = score
            best_metrics = val_metrics.copy()

        # Print per-epoch summary
        cls_line = (
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_metrics['val/total_loss']:.4f} | "
            f"ACC={val_metrics['cls/acc']:.3f} F1={val_metrics['cls/f1_macro']:.3f} Kappa={val_metrics['cls/kappa']:.3f} Alpha={val_metrics['cls/alpha']:.3f} | "
            f"ROC-AUC={val_metrics['cls/roc_auc_macro']:.3f} PR-AUC={val_metrics['cls/pr_auc_macro']:.3f}"
        )
        reg_line = (
            f"    Valence: RMSE={val_metrics['valence/rmse']:.3f} CORR={val_metrics['valence/corr']:.3f} SAGR={val_metrics['valence/sagr']:.3f} CCC={val_metrics['valence/ccc']:.3f} | "
            f"Arousal: RMSE={val_metrics['arousal/rmse']:.3f} CORR={val_metrics['arousal/corr']:.3f} SAGR={val_metrics['arousal/sagr']:.3f} CCC={val_metrics['arousal/ccc']:.3f}"
        )
        print(cls_line)
        print(reg_line)

    # Final test evaluation
    print(f"\n=== Final Test Evaluation for {model_name} ===")
    test_metrics = evaluate(model, test_loader, cfg)
    
    # Print test results
    test_cls_line = (
        f"TEST | ACC={test_metrics['cls/acc']:.3f} F1={test_metrics['cls/f1_macro']:.3f} Kappa={test_metrics['cls/kappa']:.3f} Alpha={test_metrics['cls/alpha']:.3f} | "
        f"ROC-AUC={test_metrics['cls/roc_auc_macro']:.3f} PR-AUC={test_metrics['cls/pr_auc_macro']:.3f}"
    )
    test_reg_line = (
        f"TEST | Valence: RMSE={test_metrics['valence/rmse']:.3f} CORR={test_metrics['valence/corr']:.3f} SAGR={test_metrics['valence/sagr']:.3f} CCC={test_metrics['valence/ccc']:.3f} | "
        f"Arousal: RMSE={test_metrics['arousal/rmse']:.3f} CORR={test_metrics['arousal/corr']:.3f} SAGR={test_metrics['arousal/sagr']:.3f} CCC={test_metrics['arousal/ccc']:.3f}"
    )
    print(test_cls_line)
    print(test_reg_line)

    print(f"\nBest validation metrics for {model_name} (by F1):")
    for k in sorted(best_metrics.keys()):
        print(f"  val/{k}: {best_metrics[k]:.4f}")
    
    print(f"\nFinal test metrics for {model_name}:")
    for k in sorted(test_metrics.keys()):
        print(f"  test/{k}: {test_metrics[k]:.4f}")

    # Return both validation and test metrics
    return {"validation": best_metrics, "test": test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Affective Computing - Multi-Task Baselines (Console Only)")
    parser.add_argument("--data-root", type=str, default=str(Path.cwd()), help="Dataset root containing images/ and annotations/")
    parser.add_argument("--models", type=str, nargs="*", default=["resnet34", "efficientnet_b0", "resnet18"], help="Backbones to compare")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--subset", type=int, default=0, help="Use first N train samples for quick runs; 0 = full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretraining (requires torchvision weights cache/internet)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Weights for [cls, valence, arousal]")

    args = parser.parse_args()
    root = Path(args.data_root)
    device = torch.device(args.device)
    cls_w, val_w, aro_w = args.loss_weights

    print("Configuration:")
    print(f"  data_root: {root}")
    print(f"  models: {args.models}")
    print(f"  epochs: {args.epochs} | batch_size: {args.batch_size} | lr: {args.lr} | weight_decay: {args.weight_decay}")
    print(f"  val_ratio: {args.val_ratio} | test_ratio: {args.test_ratio} | subset: {args.subset} | seed: {args.seed}")
    print(f"  device: {device} | pretrained: {args.pretrained}")
    print(f"  loss_weights [cls, valence, arousal]: {args.loss_weights}")

    results: Dict[str, Dict[str, float]] = {}
    for m in args.models:
        metrics = run_training(
            root=root,
            model_name=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            device=device,
            cls_weight=cls_w,
            val_weight=val_w,
            aro_weight=aro_w,
            pretrained=args.pretrained,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            subset=None if args.subset <= 0 else args.subset,
            seed=args.seed,
        )
        results[m] = metrics

    # Comparison table for validation metrics
    print("\n=== Comparison (Best Validation Metrics) ===")
    keys_order = [
        "cls/acc", "cls/f1_macro", "cls/kappa", "cls/alpha", "cls/roc_auc_macro", "cls/pr_auc_macro",
        "valence/rmse", "valence/corr", "valence/sagr", "valence/ccc",
        "arousal/rmse", "arousal/corr", "arousal/sagr", "arousal/ccc",
    ]
    header = "Model".ljust(18) + " | " + " | ".join(k.ljust(18) for k in keys_order)
    print(header)
    print("-" * len(header))
    for m, mets in results.items():
        row_vals = []
        val_metrics = mets.get("validation", {})
        for k in keys_order:
            v = val_metrics.get(k, float("nan"))
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row_vals.append("n/a".ljust(18))
            else:
                row_vals.append(f"{v:.3f}".ljust(18))
        print(m.ljust(18) + " | " + " | ".join(row_vals))

    # Comparison table for test metrics
    print("\n=== Comparison (Final Test Metrics) ===")
    header = "Model".ljust(18) + " | " + " | ".join(k.ljust(18) for k in keys_order)
    print(header)
    print("-" * len(header))
    for m, mets in results.items():
        row_vals = []
        test_metrics = mets.get("test", {})
        for k in keys_order:
            v = test_metrics.get(k, float("nan"))
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row_vals.append("n/a".ljust(18))
            else:
                row_vals.append(f"{v:.3f}".ljust(18))
        print(m.ljust(18) + " | " + " | ".join(row_vals))


if __name__ == "__main__":
    main()


