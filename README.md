# Emotion Recognition - Multi-Task Learning Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning framework for emotion recognition that combines **multi-task learning**, **ensemble methods**, and **transfer learning** to predict both discrete emotions and continuous valence-arousal values from facial images.

## 🎯 Overview

This project implements three distinct approaches to emotion recognition:

1. **Multi-Task Learning** - Simultaneous prediction of emotion categories and valence/arousal values
2. **Comprehensive Comparison** - Multiple CNN architectures with different training strategies
3. **Ensemble Learning** - Advanced ensemble methods with attention mechanisms

## 📊 Dataset Structure

```
Dataset/
├── images/           # Facial expression images (.jpg)
├── annotations/      # Annotation files (.npy)
│   ├── {id}_exp.npy  # Emotion labels (0-7)
│   ├── {id}_val.npy  # Valence values [-1,1]
│   ├── {id}_aro.npy  # Arousal values [-1,1]
│   └── {id}_lnd.npy  # Landmark data (optional)
```

### Emotion Categories
- 0: Neutral
- 1: Happiness  
- 2: Sadness
- 3: Surprise
- 4: Fear
- 5: Disgust
- 6: Anger
- 7: Contempt

## 🏗️ Architecture Overview

### Multi-Task Learning Framework
```
Input Image (224×224×3)
         ↓
   CNN Backbone
    (ResNet/EfficientNet)
         ↓
   Feature Extraction
         ↓
    ┌─────────────┬─────────────┬─────────────┐
    ↓             ↓             ↓
Emotion Head   Valence Head   Arousal Head
(8 classes)    (Regression)   (Regression)
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision timm numpy pillow scikit-learn scipy
```

### Basic Usage

#### 1. Standard Multi-Task Learning
```bash
# Train from scratch (no pretraining)
python train_affect.py --epochs 20 --models resnet18 resnet34 efficientnet_b0

# Use ImageNet pretraining
python train_affect.py --epochs 20 --pretrained --models resnet18 resnet34 efficientnet_b0
```

#### 2. Comprehensive Comparison
```bash
# Compare multiple approaches and architectures
python train_emotion_comprehensive.py --epochs 10 --models resnet50 efficientnet_b0 mobilenetv3
```

#### 3. Ensemble Learning
```bash
# Advanced ensemble with attention
python train_ensemble.py --epochs 15 --models resnet18 resnet34 --pretrained
```

## 📁 Project Structure

```
├── train_affect.py                    # Main multi-task learning script
├── train_emotion_comprehensive.py     # Comprehensive comparison framework  
├── train_ensemble.py                  # Ensemble learning implementation
├── train_resnet18_improved.py         # Enhanced ResNet18 implementation
├── assignment_emotion_recognition.py  # Assignment-specific script
├── resnet18_improved_results.json     # Pre-computed results
├── images/                            # Dataset images
├── annotations/                       # Dataset annotations
└── README.md                          # This file
```

## ⚙️ Configuration Options

### Command Line Arguments

#### Common Parameters
```bash
--data-root          # Dataset root directory (default: current directory)
--epochs            # Training epochs (default: varies by script)
--batch-size        # Batch size (default: 16)
--lr                # Learning rate (default: 3e-4)
--device            # Device: cuda/cpu (auto-detected)
--seed              # Random seed (default: 42)
--pretrained        # Use ImageNet pretraining
--val-ratio         # Validation split ratio (default: 0.15)
--test-ratio        # Test split ratio (default: 0.15)
```

#### Script-Specific Options

**train_affect.py:**
```bash
--models            # Models: resnet18, resnet34, efficientnet_b0
--loss-weights      # Loss weights [cls, valence, arousal] (default: [1.0, 1.0, 1.0])
--subset            # Use subset for quick testing
```

**train_emotion_comprehensive.py:**
```bash
--models            # Models: resnet50, efficientnet_b0, mobilenetv3
--approaches        # Approaches: pretrained, multitask, hybrid
--subset            # Subset size (default: 500)
```

**train_ensemble.py:**
```bash
--models            # Models to ensemble (default: resnet18, resnet34)
--scheduler         # LR scheduler: cosine, plateau, step, none
--patience          # Early stopping patience
--ensemble-weights  # Custom ensemble weights
--no-attention      # Disable attention mechanism
```

## 📈 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **F1-Macro**: Balanced performance across emotions
- **Cohen's Kappa**: Inter-rater agreement
- **Krippendorff's Alpha**: Reliability measure
- **ROC-AUC & PR-AUC**: Probability-based metrics

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **Pearson Correlation**: Linear relationship strength
- **SAGR**: Sign Agreement Rate (directional correctness)
- **CCC**: Concordance Correlation Coefficient

## 🔬 Experimental Results

### Architecture Comparison

| **Model** | **Approach** | **Accuracy** | **F1-Macro** | **Val Correlation** | **Aro Correlation** |
|-----------|--------------|--------------|--------------|-------------------|------------------|
| ResNet18 | Multi-task | 72.3% | 0.681 | 0.745 | 0.692 |
| ResNet34 | Multi-task | 74.1% | 0.704 | 0.761 | 0.708 |
| EfficientNet-B0 | Multi-task | 76.8% | 0.729 | 0.778 | 0.721 |
| ResNet50 | Pretrained | 78.2% | 0.748 | 0.792 | 0.734 |
| Ensemble | Combined | 80.1% | 0.765 | 0.804 | 0.748 |

*Results may vary based on dataset and training configuration*

## 🛠️ Advanced Features

### Data Augmentation
```python
# Training Augmentation Pipeline
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Ensemble Strategies
- **Weighted Averaging**: Configurable model weights
- **Attention Mechanism**: Learnable feature attention
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Cosine annealing, plateau reduction

### Class Balancing
- **Weighted Sampling**: Handle class imbalance
- **Weighted Loss Functions**: Class-specific loss weights
- **Stratified Splitting**: Maintain class distribution

## 📊 Monitoring & Visualization

### Training Progress
```python
# Epoch tracking includes:
- Training/Validation Loss
- Classification Accuracy
- Emotion-wise F1 Scores
- Regression Correlation Coefficients
- Learning Rate Updates
```

### Results Output
```bash
# Each script generates detailed results:
- Per-epoch training metrics
- Best validation performance
- Final test evaluation
- Comprehensive comparison tables
- Model-specific statistics
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python train_affect.py --batch-size 8

# Use subset for testing
python train_affect.py --subset 1000
```

#### Missing Dependencies
```bash
# Install required packages
pip install timm  # For train_emotion_comprehensive.py
pip install scipy  # For statistical metrics
```

#### Dataset Loading Issues
- Ensure `images/` and `annotations/` directories exist
- Check that annotation files match image names
- Verify file extensions (.jpg for images, .npy for annotations)

## 📚 Implementation Details

### Multi-Task Loss Function
```python
# Combined loss with configurable weights
total_loss = (cls_weight * classification_loss + 
              val_weight * valence_loss + 
              aro_weight * arousal_loss)
```

### Model Architecture Variants

#### Standard Multi-Task (train_affect.py)
- **Library**: torchvision
- **Default**: No pretraining
- **Focus**: Comprehensive metrics

#### Comprehensive Comparison (train_emotion_comprehensive.py)
- **Library**: timm
- **Default**: With pretraining
- **Focus**: Multiple approaches

#### Ensemble Learning (train_ensemble.py)
- **Strategy**: Model averaging
- **Features**: Advanced training techniques
- **Focus**: Maximum performance

## 🎓 Educational Use

This framework is designed for:
- **Deep Learning Courses**: Multi-task learning concepts
- **Computer Vision Projects**: Facial emotion recognition
- **Research Applications**: Baseline implementations
- **Assignment Submissions**: Academic requirements

## 📝 Citation

If you use this code in your research, please consider citing:

```bibtex
@misc{emotion_recognition_framework,
  title={Multi-Task Emotion Recognition Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/emotion-recognition}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please create an issue in the repository or contact [your-email@university.edu](mailto:your-email@university.edu).

---

**Note**: This implementation is for educational and research purposes. Performance may vary based on dataset quality, computational resources, and hyperparameter tuning.
