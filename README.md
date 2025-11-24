# MelanomaNet: Explainable Skin Lesion Classification

Explainable deep learning system for multi-class skin lesion classification with ABCDE criterion validation using attention-based visualization (GradCAM++).

## Features

- **Multi-class skin lesion classification** across all 9 ISIC 2019 diagnostic categories:
  - MEL (Melanoma), NV (Nevus), BCC (Basal cell carcinoma), AK (Actinic keratosis)
  - BKL (Benign keratosis), DF (Dermatofibroma), VASC (Vascular), SCC (Squamous cell carcinoma), UNK (Unknown)
- **EfficientNet V2 Large** backbone with high-resolution (512x512) input for maximum detail preservation
- Support for EfficientNet V2 architectures (S/M/L)
- GradCAM++ attention visualization for model explainability
- **Novel ABCDE criterion analysis with automated feature extraction:**
  - **A**symmetry detection and quantification
  - **B**order irregularity measurement
  - **C**olor variation analysis using K-means clustering
  - **D**iameter calculation
  - **E**volution tracking (future work with temporal data)
- **GradCAM-ABCDE alignment metrics** to validate model attention
- Class imbalance handling (weighted loss, focal loss)
- Comprehensive clinical interpretability reports
- Production-ready inference pipeline with checkpoint resumption

## Installation

```bash
# Install dependencies
pdm install

# If PDM is not installed then install it via pip:
pip install pdm
```

## Dataset Preparation

### ISIC 2019

1. Download from: <https://challenge.isic-archive.com/data/>
2. Extract to `data/isic_2019/`:

```text
data/isic_2019/
├── train/
│   ├── ISIC_0000001.jpg
│   └── ...
├── test/
│   ├── ISIC_0001001.jpg
│   └── ...
└── ISIC_2019_Training_GroundTruth.csv
```

## Usage

### Training

```bash
pdm run train
# Or: python scripts/train.py --config config.yaml
```

### Evaluation

```bash
pdm run eval
# Or: python scripts/eval.py --checkpoint checkpoints/best_model.pth --config config.yaml
```

### Inference

```bash
# Run inference
python scripts/infer.py --checkpoint checkpoints/best_model.pth \
                        --input path/to/image.jpg \
                        --config config.yaml
```

### Resume Training from Checkpoint

```bash
# Resume from last checkpoint
python scripts/train.py --config config.yaml --resume ./checkpoints/last_checkpoint.pth

# Resume from specific epoch
python scripts/train.py --config config.yaml --resume ./checkpoints/checkpoint_epoch_10.pth
```

## Configuration

Edit [config.yaml](config.yaml) to customize:

- Dataset path and splits
- **Model architecture**: Choose from `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l`
- **Input resolution**: Default 512x512 for high-detail medical imaging (configurable: 224, 384, 480, 512, 640)
- Training hyperparameters (batch size optimized for 512x512 on 16GB GPU)
- Augmentation settings
- GradCAM parameters
- ABCDE analysis thresholds (automatically scaled with image resolution)

## Model Architecture

MelanomaNet uses:

- **Backbone**: EfficientNet V2 Large (default, 119M params, 1280 features)
  - Also supports: V2-S (22M params), V2-M (54M params)
- **Input Resolution**: 512x512 RGB images (4.5x more pixels than standard 224x224)
- **Pretrained**: ImageNet-1K weights for transfer learning
- **Classifier Head**: Global average pooling + dropout (0.3) + linear layer
- **Explainability**: GradCAM++ for attention visualization

### Data Augmentation

- Geometric: horizontal/vertical flips, rotation, affine transforms
- Color: brightness, contrast, saturation, hue adjustments
- Normalization: ImageNet statistics

### Class Imbalance Handling

- Weighted cross-entropy loss
- Optional focal loss
- Stratified train/val/test split

### Training Features

- Mixed precision training (AMP)
- Cosine annealing learning rate scheduler
- Model checkpointing (last and best F1)

### Evaluation Metrics

For multi-class classification (weighted averaging):

- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Confusion Matrix

### ABCDE Clinical Interpretability

Automated extraction and analysis of clinical ABCDE features:

**Asymmetry (A):**

- Compares lesion halves along horizontal and vertical axes
- Quantifies asymmetry score (0-1)
- Highlights symmetry axes

**Border (B):**

- Analyzes contour irregularity
- Measures compactness and vertex count
- Detects poorly defined borders

**Color (C):**

- K-means clustering to identify distinct colors
- Counts significant color variations
- Generates color palette visualization

**Diameter (D):**

- Calculates maximum lesion diameter
- Uses minimum enclosing circle and bounding box
- Compares against clinical threshold

**Evolution (E):**

- Framework for temporal change analysis
- Requires multi-timepoint imaging (future work)

**GradCAM Alignment:**

- Quantifies how well model attention aligns with ABCDE features
- Provides border, color, and overall alignment scores
- Validates clinical interpretability of model focus

## Citation

```bibtex
@misc{melanomanet2025,
  title={MelanomaNet: Explainable Skin Lesion Classification},
  author={Ilyosbekov, Sukhrob},
  year={2025}
}
```

## License

MIT License
