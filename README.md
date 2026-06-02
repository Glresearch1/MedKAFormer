# MedKAFormer

PyTorch implementation of **MedKAFormer**, a KAN-enhanced dilated transformer model for medical image classification.

The main entry points are:

- `medkanformer.py`: MedKAFormer model definitions.
- `train_kvasir.py`: training and evaluation script for Kvasir-style image folders.

## Repository Layout

```text
.
├── medkanformer.py          # Main model
├── train_kvasir.py          # Main training script
├── KANLinear.py             # KAN linear layer
├── kan_convs/               # KAN convolution layers
├── modules/                 # Dynamic filter, ContraNorm, and KAN conv helpers
└── kvasir/
    ├── kvasir_dataset.py    # Kvasir image-folder dataset
    └── kvasir_utils.py      # Classification metrics
```

## Installation

Python 3.9+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version if the default package is not suitable.

## Dataset Format

`train_kvasir.py` expects an image-folder dataset where each class is a subdirectory:

```text
data/kvasir/
├── train/
│   ├── class_0/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   └── class_1/
│       └── image_003.jpg
└── test/
    ├── class_0/
    └── class_1/
```

The train, validation, and test folders must contain the same class folder names.

## Training

```bash
python3 train_kvasir.py \
  --train-dir data/kvasir/train \
  --test-dir data/kvasir/test \
  --model tiny \
  --num-epochs 100 \
  --batch-size 32
```

Use a separate validation split with `--val-dir`:

```bash
python3 train_kvasir.py \
  --train-dir data/kvasir/train \
  --val-dir data/kvasir/val \
  --test-dir data/kvasir/test
```

Outputs are written to `kvasir_output/<timestamp>/`, including:

- `best_model_kvasir.pth`
- `config.json`
- `metrics.txt`
- `metrics.json`
- `Tensorboard_Results/`

## Evaluation

```bash
python3 train_kvasir.py \
  --train-dir data/kvasir/train \
  --test-dir data/kvasir/test \
  --checkpoint kvasir_output/<timestamp>/best_model_kvasir.pth \
  --eval-only
```

## Model Usage

```python
import torch
from medkanformer import medkaformer_tiny

model = medkaformer_tiny(pretrained=False, num_classes=8)
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(y.shape)
```

Available model builders:

- `medkaformer_tiny`
- `medkaformer_small`
- `medkaformer_base`

## Notes

- Datasets, checkpoints, TensorBoard logs, and experiment outputs are intentionally excluded from the repository.
- `train_kvasir.py` infers the number of classes from the training folder unless `--num-classes` is provided.
- AUC is computed one-vs-rest per class and averaged over classes that have both positive and negative samples.

## License

This project is released under the MIT License unless otherwise noted.
Some bundled or adapted third-party components retain their original Apache-2.0 or upstream notices;
see `THIRD_PARTY_NOTICES.md`.
