#!/usr/bin/env python3
"""
Comprehensive Emotion Recognition Comparison
============================================

This script implements multiple approaches for emotion recognition:

1. **Pretrained CNN Models**: ResNet50, EfficientNet-B0, MobileNetV3
2. **Multi-task Learning**: Shared backbone + regression heads
3. **Hybrid Approach**: Joint training for both classification and regression

Based on Context7 MCP research and ResEmoteNet architecture insights.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, 
    roc_auc_score, average_precision_score,
    mean_squared_error
)
from scipy.stats import pearsonr
# import cv2  # Not used in this script

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Dataset Classes
# ============================================================================

class EmotionDataset(Dataset):
    """Dataset for emotion recognition with multi-task learning."""
    
    def __init__(self, data_root: Path, split: str = "train", transform=None, subset: int = 0):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.subset = subset
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Apply subset if specified
        if subset > 0:
            self.annotations = self.annotations[:subset]
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from .npy files."""
        annotations = []
        images_dir = self.data_root / "images"
        annotations_dir = self.data_root / "annotations"
        
        if not images_dir.exists() or not annotations_dir.exists():
            raise FileNotFoundError(f"Dataset directories not found: {self.data_root}")
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg"))
        
        for img_path in image_files:
            # Find corresponding annotation files (separate files for exp, val, aro)
            exp_path = annotations_dir / f"{img_path.stem}_exp.npy"
            val_path = annotations_dir / f"{img_path.stem}_val.npy"
            aro_path = annotations_dir / f"{img_path.stem}_aro.npy"
            
            if exp_path.exists() and val_path.exists() and aro_path.exists():
                try:
                    # Load separate annotation files
                    emotion = int(np.load(exp_path))
                    valence = float(np.load(val_path))
                    arousal = float(np.load(aro_path))
                    
                    annotations.append({
                        'image_path': str(img_path),
                        'emotion': emotion,
                        'valence': valence,
                        'arousal': arousal
                    })
                except Exception as e:
                    print(f"Warning: Could not load annotation for {img_path.name}: {e}")
                    continue
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ann = self.annotations[idx]
        
        # Load image
        try:
            image = Image.open(ann['image_path']).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {ann['image_path']}: {e}")
            # Create dummy image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'emotion': torch.tensor(ann['emotion'], dtype=torch.long),
            'valence': torch.tensor(ann['valence'], dtype=torch.float32),
            'arousal': torch.tensor(ann['arousal'], dtype=torch.float32)
        }

# ============================================================================
# Model Architectures
# ============================================================================

class PretrainedCNNModel(nn.Module):
    """Pretrained CNN model for emotion classification only."""
    
    def __init__(self, model_name: str, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name.lower()
        
        # Load pretrained model using timm
        if self.model_name == "resnet50":
            self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif self.model_name == "efficientnet_b0":
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif self.model_name == "mobilenetv3":
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

class MultiTaskModel(nn.Module):
    """Multi-task model with shared backbone and separate heads."""
    
    def __init__(self, backbone_name: str, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        # Load backbone
        if self.backbone_name == "resnet50":
            self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif self.backbone_name == "efficientnet_b0":
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif self.backbone_name == "mobilenetv3":
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Task-specific heads
        self.dropout = nn.Dropout(0.2)
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Valence regression head
        self.valence_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Arousal regression head
        self.arousal_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.dropout(features)
        
        return {
            'emotion': self.emotion_head(features),
            'valence': self.valence_head(features).squeeze(1),
            'arousal': self.arousal_head(features).squeeze(1)
        }

class HybridModel(nn.Module):
    """Hybrid model with joint training for classification and regression."""
    
    def __init__(self, backbone_name: str, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        # Load backbone
        if self.backbone_name == "resnet50":
            self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif self.backbone_name == "efficientnet_b0":
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif self.backbone_name == "mobilenetv3":
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.emotion_head = nn.Linear(256, num_classes)
        self.valence_head = nn.Linear(256, 1)
        self.arousal_head = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        shared_features = self.shared_layers(features)
        
        return {
            'emotion': self.emotion_head(shared_features),
            'valence': self.valence_head(shared_features).squeeze(1),
            'arousal': self.arousal_head(shared_features).squeeze(1)
        }

# ============================================================================
# Training and Evaluation
# ============================================================================

class EmotionTrainer:
    """Trainer class for emotion recognition models."""
    
    def __init__(self, model: nn.Module, device: torch.device, loss_weights: List[float] = [1.0, 1.0, 1.0]):
        self.model = model.to(device)
        self.device = device
        self.cls_weight, self.val_weight, self.aro_weight = loss_weights
        
        # Loss functions
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        
        # Metrics tracking
        self.best_metrics = {}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        cls_loss = 0.0
        val_loss = 0.0
        aro_loss = 0.0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            emotions = batch['emotion'].to(self.device)
            valences = batch['valence'].to(self.device)
            arousals = batch['arousal'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if isinstance(self.model, PretrainedCNNModel):
                # Classification only
                outputs = self.model(images)
                loss = self.cls_criterion(outputs, emotions)
            else:
                # Multi-task or hybrid
                outputs = self.model(images)
                
                cls_loss_batch = self.cls_criterion(outputs['emotion'], emotions)
                val_loss_batch = self.reg_criterion(outputs['valence'], valences)
                aro_loss_batch = self.reg_criterion(outputs['arousal'], arousals)
                
                loss = (self.cls_weight * cls_loss_batch + 
                       self.val_weight * val_loss_batch + 
                       self.aro_weight * aro_loss_batch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            if not isinstance(self.model, PretrainedCNNModel):
                cls_loss += cls_loss_batch.item()
                val_loss += val_loss_batch.item()
                aro_loss += aro_loss_batch.item()
        
        epoch_loss = total_loss / len(dataloader)
        metrics = {'train_loss': epoch_loss}
        
        if not isinstance(self.model, PretrainedCNNModel):
            metrics.update({
                'cls_loss': cls_loss / len(dataloader),
                'val_loss': val_loss / len(dataloader),
                'aro_loss': aro_loss / len(dataloader)
            })
        
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        all_emotions = []
        all_valences = []
        all_arousals = []
        all_pred_emotions = []
        all_pred_valences = []
        all_pred_arousals = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                emotions = batch['emotion'].to(self.device)
                valences = batch['valence'].to(self.device)
                arousals = batch['arousal'].to(self.device)
                
                if isinstance(self.model, PretrainedCNNModel):
                    outputs = self.model(images)
                    pred_emotions = outputs.argmax(dim=1)
                    pred_valences = torch.zeros_like(valences)
                    pred_arousals = torch.zeros_like(arousals)
                else:
                    outputs = self.model(images)
                    pred_emotions = outputs['emotion'].argmax(dim=1)
                    pred_valences = outputs['valence']
                    pred_arousals = outputs['arousal']
                
                all_emotions.extend(emotions.cpu().numpy())
                all_valences.extend(valences.cpu().numpy())
                all_arousals.extend(arousals.cpu().numpy())
                all_pred_emotions.extend(pred_emotions.cpu().numpy())
                all_pred_valences.extend(pred_valences.cpu().numpy())
                all_pred_arousals.extend(pred_arousals.cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        acc = accuracy_score(all_emotions, all_pred_emotions)
        f1 = f1_score(all_emotions, all_pred_emotions, average='macro')
        kappa = cohen_kappa_score(all_emotions, all_pred_emotions)
        
        metrics.update({
            'accuracy': acc,
            'f1_macro': f1,
            'kappa': kappa
        })
        
        # Regression metrics (if applicable)
        if not isinstance(self.model, PretrainedCNNModel):
            val_rmse = np.sqrt(mean_squared_error(all_valences, all_pred_valences))
            aro_rmse = np.sqrt(mean_squared_error(all_arousals, all_pred_arousals))
            
            try:
                val_corr, _ = pearsonr(all_valences, all_pred_valences)
                aro_corr, _ = pearsonr(all_arousals, all_pred_arousals)
            except:
                val_corr = aro_corr = 0.0
            
            metrics.update({
                'valence_rmse': val_rmse,
                'valence_corr': val_corr,
                'arousal_rmse': aro_rmse,
                'arousal_corr': aro_corr
            })
        
        return metrics

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Emotion Recognition Comparison")
    parser.add_argument("--data-root", type=str, default=str(Path.cwd()), help="Dataset root")
    parser.add_argument("--models", type=str, nargs="*", 
                       default=["resnet50", "efficientnet_b0", "mobilenetv3"], 
                       help="Models to compare")
    parser.add_argument("--approaches", type=str, nargs="*", 
                       default=["pretrained", "multitask", "hybrid"], 
                       help="Approaches to test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset", type=int, default=500, help="Subset size for quick testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    
    print(f"🚀 Comprehensive Emotion Recognition Comparison")
    print(f"📊 Models: {args.models}")
    print(f"🔬 Approaches: {args.approaches}")
    print(f"⚙️  Epochs: {args.epochs} | Batch: {args.batch_size} | Subset: {args.subset}")
    print(f"💻 Device: {device}")
    print("=" * 80)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = EmotionDataset(data_root, "train", train_transform, args.subset)
    val_dataset = EmotionDataset(data_root, "val", val_transform, args.subset // 4)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"📁 Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print()
    
    # Results storage
    all_results = {}
    
    # Test each approach
    for approach in args.approaches:
        print(f"🔬 Testing {approach.upper()} approach...")
        print("-" * 50)
        
        approach_results = {}
        
        for model_name in args.models:
            print(f"\n📊 Training {model_name} ({approach})...")
            
            try:
                # Create model based on approach
                if approach == "pretrained":
                    model = PretrainedCNNModel(model_name, num_classes=8, pretrained=True)
                elif approach == "multitask":
                    model = MultiTaskModel(model_name, num_classes=8, pretrained=True)
                elif approach == "hybrid":
                    model = HybridModel(model_name, num_classes=8, pretrained=True)
                else:
                    raise ValueError(f"Unknown approach: {approach}")
                
                # Create trainer
                trainer = EmotionTrainer(model, device)
                
                # Training loop
                best_f1 = 0.0
                best_metrics = {}
                
                for epoch in range(args.epochs):
                    # Train
                    train_metrics = trainer.train_epoch(train_loader)
                    
                    # Evaluate
                    val_metrics = trainer.evaluate(val_loader)
                    
                    # Print progress
                    if approach == "pretrained":
                        print(f"Epoch {epoch+1:2d} | Loss={train_metrics['train_loss']:.4f} | "
                              f"Acc={val_metrics['accuracy']:.3f} | F1={val_metrics['f1_macro']:.3f}")
                    else:
                        print(f"Epoch {epoch+1:2d} | Loss={train_metrics['train_loss']:.4f} | "
                              f"Acc={val_metrics['accuracy']:.3f} | F1={val_metrics['f1_macro']:.3f} | "
                              f"ValCorr={val_metrics.get('valence_corr', 0):.3f} | "
                              f"AroCorr={val_metrics.get('arousal_corr', 0):.3f}")
                    
                    # Track best metrics
                    if val_metrics['f1_macro'] > best_f1:
                        best_f1 = val_metrics['f1_macro']
                        best_metrics = val_metrics.copy()
                
                approach_results[model_name] = best_metrics
                print(f"✅ {model_name} completed - Best F1: {best_f1:.3f}")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                approach_results[model_name] = {}
        
        all_results[approach] = approach_results
        print()
    
    # Print comprehensive results
    print("🏆 COMPREHENSIVE RESULTS")
    print("=" * 80)
    
    for approach, results in all_results.items():
        print(f"\n📈 {approach.upper()} APPROACH:")
        print("-" * 40)
        
        if approach == "pretrained":
            print(f"{'Model':<15} {'Accuracy':<10} {'F1-Macro':<10} {'Kappa':<10}")
            print("-" * 50)
            for model, metrics in results.items():
                if metrics:
                    print(f"{model:<15} {metrics.get('accuracy', 0):.3f}      "
                          f"{metrics.get('f1_macro', 0):.3f}      "
                          f"{metrics.get('kappa', 0):.3f}")
        else:
            print(f"{'Model':<15} {'Accuracy':<10} {'F1-Macro':<10} {'ValCorr':<10} {'AroCorr':<10}")
            print("-" * 60)
            for model, metrics in results.items():
                if metrics:
                    print(f"{model:<15} {metrics.get('accuracy', 0):.3f}      "
                          f"{metrics.get('f1_macro', 0):.3f}      "
                          f"{metrics.get('valence_corr', 0):.3f}      "
                          f"{metrics.get('arousal_corr', 0):.3f}")
    
    # Find best overall model
    best_overall = None
    best_score = 0.0
    
    for approach, results in all_results.items():
        for model, metrics in results.items():
            if metrics and metrics.get('f1_macro', 0) > best_score:
                best_score = metrics['f1_macro']
                best_overall = (approach, model, metrics)
    
    if best_overall:
        approach, model, metrics = best_overall
        print(f"\n🥇 BEST OVERALL MODEL:")
        print(f"   Approach: {approach}")
        print(f"   Model: {model}")
        print(f"   F1-Score: {metrics.get('f1_macro', 0):.3f}")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
        if 'valence_corr' in metrics:
            print(f"   Valence Correlation: {metrics.get('valence_corr', 0):.3f}")
            print(f"   Arousal Correlation: {metrics.get('arousal_corr', 0):.3f}")

if __name__ == "__main__":
    main()
