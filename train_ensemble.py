import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Import from the improved script
from train_resnet18_improved import (
    set_seed, ImprovedAffectDataset, ImprovedMultiTaskModel, 
    get_class_weights, create_weighted_sampler,
    accuracy_score, f1_score_macro, cohen_kappa_score, krippendorff_alpha_nominal,
    multiclass_ovr_auc, rmse, pearson_corr, sagr, ccc, evaluate
)


# =============================
# Ensemble Model
# =============================
class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        self.num_models = len(models)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get predictions from all models
        all_exp_logits = []
        all_val_preds = []
        all_aro_preds = []

        for model in self.models:
            with torch.no_grad():
                outputs = model(x)
                all_exp_logits.append(outputs["exp"])
                all_val_preds.append(outputs["val"])
                all_aro_preds.append(outputs["aro"])

        # Weighted ensemble
        exp_logits = torch.zeros_like(all_exp_logits[0])
        val_pred = torch.zeros_like(all_val_preds[0])
        aro_pred = torch.zeros_like(all_aro_preds[0])

        for i, (exp_log, val_pred_single, aro_pred_single) in enumerate(
            zip(all_exp_logits, all_val_preds, all_aro_preds)
        ):
            weight = self.weights[i]
            exp_logits += weight * exp_log
            val_pred += weight * val_pred_single
            aro_pred += weight * aro_pred_single

        return {
            "exp": exp_logits,
            "val": val_pred,
            "aro": aro_pred
        }


# =============================
# Ensemble Training Configuration
# =============================
@dataclass
class EnsembleConfig:
    model_names: List[str]
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
    use_attention: bool
    scheduler_type: str
    patience: int
    ensemble_weights: Optional[List[float]] = None


def train_individual_models(root: Path, model_names: List[str], epochs: int, batch_size: int, 
                           lr: float, weight_decay: float, num_workers: int, device: torch.device,
                           cls_weight: float, val_weight: float, aro_weight: float, 
                           pretrained: bool, val_ratio: float, test_ratio: float, 
                           subset: Optional[int], seed: int, use_attention: bool = True,
                           scheduler_type: str = "cosine", patience: int = 5) -> List[nn.Module]:
    """Train individual models and return the best checkpoints"""
    
    set_seed(seed)

    # Build full id list
    full_ds = ImprovedAffectDataset(root, is_train=True)
    n = len(full_ds)
    indices = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    # Calculate split sizes
    val_size = max(1, int(n * val_ratio))
    test_size = max(1, int(n * test_ratio))
    train_size = n - val_size - test_size
    
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

    train_ds = ImprovedAffectDataset(root, split_indices=train_indices, is_train=True)
    val_ds = ImprovedAffectDataset(root, split_indices=val_indices, is_train=False)
    test_ds = ImprovedAffectDataset(root, split_indices=test_indices, is_train=False)

    # Create weighted sampler for class balancing
    weighted_sampler = create_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=weighted_sampler, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)

    # Get class weights for loss
    class_weights = get_class_weights(train_ds)

    trained_models = []
    
    for model_name in model_names:
        print(f"\n=== Training {model_name} for Ensemble ===")
        
        model = ImprovedMultiTaskModel(model_name, num_classes=8, pretrained=pretrained, use_attention=use_attention)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, min_lr=lr * 0.01)
        else:
            scheduler = None

        cfg = EnsembleConfig(
            model_names=[model_name],
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
            use_attention=use_attention,
            scheduler_type=scheduler_type,
            patience=patience,
        )

        print(f"Device: {device} | Train: {len(train_ds)} | Val: {len(val_ds)}")
        print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Scheduler: {scheduler_type}")

        best_model_state = None
        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            total_loss = 0.0
            total_batches = 0
            
            cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
            mse = nn.MSELoss()

            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                y_exp = batch["exp"].to(device, non_blocking=True)
                y_val = batch["val"].to(device, non_blocking=True)
                y_aro = batch["aro"].to(device, non_blocking=True)

                out = model(images)
                loss_cls = cls_loss_fn(out["exp"], y_exp)
                loss_val = mse(out["val"], y_val)
                loss_aro = mse(out["aro"], y_aro)
                loss = cls_weight * loss_cls + val_weight * loss_val + aro_weight * loss_aro

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

            avg_loss = total_loss / max(total_batches, 1)
            
            # Step scheduler
            if scheduler_type in ["cosine", "step"]:
                scheduler.step()

            # Validation
            val_metrics = evaluate(model, val_loader, cfg)
            
            # Step plateau scheduler
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["cls/f1_macro"])

            # Track best model
            f1_score = val_metrics.get("cls/f1_macro", -1.0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_loss={val_metrics['val/total_loss']:.4f} | "
                      f"F1={val_metrics['cls/f1_macro']:.3f} | Best F1={best_f1:.3f}")

            # Early stopping
            if patience_counter >= patience and epoch > 10:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model for {model_name} (F1={best_f1:.3f})")
        
        trained_models.append(model)
        print(f"Completed training {model_name}")

    return trained_models


def evaluate_ensemble(ensemble_model: EnsembleModel, test_loader: DataLoader, 
                     cfg: EnsembleConfig) -> Dict[str, float]:
    """Evaluate ensemble model"""
    ensemble_model.eval()
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

    for batch in test_loader:
        images = batch["image"].to(cfg.device, non_blocking=True)
        y_exp = batch["exp"].to(cfg.device, non_blocking=True)
        y_val = batch["val"].to(cfg.device, non_blocking=True)
        y_aro = batch["aro"].to(cfg.device, non_blocking=True)

        out = ensemble_model(images)

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


def run_ensemble_training(root: Path, model_names: List[str], epochs: int, batch_size: int, 
                         lr: float, weight_decay: float, num_workers: int, device: torch.device, 
                         cls_weight: float, val_weight: float, aro_weight: float, 
                         pretrained: bool, val_ratio: float, test_ratio: float, 
                         subset: Optional[int], seed: int, use_attention: bool = True,
                         scheduler_type: str = "cosine", patience: int = 5,
                         ensemble_weights: Optional[List[float]] = None) -> Dict[str, float]:
    """Train ensemble of models"""
    
    print(f"\n=== Ensemble Training: {model_names} ===")
    print(f"Ensemble weights: {ensemble_weights}")
    
    # Train individual models
    trained_models = train_individual_models(
        root=root, model_names=model_names, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, num_workers=num_workers, device=device,
        cls_weight=cls_weight, val_weight=val_weight, aro_weight=aro_weight,
        pretrained=pretrained, val_ratio=val_ratio, test_ratio=test_ratio,
        subset=subset, seed=seed, use_attention=use_attention,
        scheduler_type=scheduler_type, patience=patience
    )
    
    # Create ensemble
    ensemble_model = EnsembleModel(trained_models, weights=ensemble_weights)
    ensemble_model.to(device)
    
    # Prepare test data
    set_seed(seed)
    full_ds = ImprovedAffectDataset(root, is_train=True)
    n = len(full_ds)
    indices = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    val_size = max(1, int(n * val_ratio))
    test_size = max(1, int(n * test_ratio))
    train_size = n - val_size - test_size
    
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    if subset is not None and subset > 0:
        train_indices = train_indices[:subset]
        val_indices = val_indices[:max(1, min(subset // 5, len(val_indices)))]
        test_indices = test_indices[:max(1, min(subset // 10, len(test_indices)))]

    test_ds = ImprovedAffectDataset(root, split_indices=test_indices, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # Evaluate ensemble
    cfg = EnsembleConfig(
        model_names=model_names,
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
        use_attention=use_attention,
        scheduler_type=scheduler_type,
        patience=patience,
        ensemble_weights=ensemble_weights,
    )
    
    print(f"\n=== Final Ensemble Evaluation ===")
    test_metrics = evaluate_ensemble(ensemble_model, test_loader, cfg)
    
    # Print results
    print(f"\nEnsemble Test Results:")
    print(f"ACC={test_metrics['cls/acc']:.3f} F1={test_metrics['cls/f1_macro']:.3f} Kappa={test_metrics['cls/kappa']:.3f}")
    print(f"Valence: RMSE={test_metrics['valence/rmse']:.3f} CORR={test_metrics['valence/corr']:.3f} CCC={test_metrics['valence/ccc']:.3f}")
    print(f"Arousal: RMSE={test_metrics['arousal/rmse']:.3f} CORR={test_metrics['arousal/corr']:.3f} CCC={test_metrics['arousal/ccc']:.3f}")
    
    return {"ensemble_test": test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble Training for Affective Computing")
    parser.add_argument("--data-root", type=str, default=str(Path.cwd()), help="Dataset root containing images/ and annotations/")
    parser.add_argument("--models", type=str, nargs="+", default=["resnet18", "resnet34"], help="Models to ensemble")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--subset", type=int, default=0, help="Use first N train samples for quick runs; 0 = full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretraining")
    parser.add_argument("--no-attention", action="store_true", help="Disable attention mechanism")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "step", "none"], help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--ensemble-weights", type=float, nargs="+", help="Ensemble weights (default: equal weights)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss-weights", type=float, nargs=3, default=[2.0, 1.0, 1.5], help="Weights for [cls, valence, arousal]")

    args = parser.parse_args()
    root = Path(args.data_root)
    device = torch.device(args.device)
    cls_w, val_w, aro_w = args.loss_weights

    print("=== Ensemble Training Configuration ===")
    print(f"  data_root: {root}")
    print(f"  models: {args.models}")
    print(f"  epochs: {args.epochs} | batch_size: {args.batch_size} | lr: {args.lr}")
    print(f"  val_ratio: {args.val_ratio} | test_ratio: {args.test_ratio} | subset: {args.subset}")
    print(f"  device: {device} | pretrained: {args.pretrained} | attention: {not args.no_attention}")
    print(f"  scheduler: {args.scheduler} | patience: {args.patience}")
    print(f"  ensemble_weights: {args.ensemble_weights}")
    print(f"  loss_weights [cls, valence, arousal]: {args.loss_weights}")

    metrics = run_ensemble_training(
        root=root,
        model_names=args.models,
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
        use_attention=not args.no_attention,
        scheduler_type=args.scheduler,
        patience=args.patience,
        ensemble_weights=args.ensemble_weights,
    )

    # Save results
    results_file = "ensemble_results.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEnsemble results saved to {results_file}")


if __name__ == "__main__":
    main()
