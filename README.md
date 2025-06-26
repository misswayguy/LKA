# LKA: Large Kernel Adapter for Enhanced Medical Image Classification

This repository contains code and experiments for integrating large kernel adapters into ViT, Swin Transformer, and ConvNeXt backbones for medical image classification.

## 📁 Directory Structure

- `vit/` — Experiments using Vision Transformer
- `sw/` — Experiments using Swin Transformer
- `convnext/` — Experiments using ConvNeXt
- `*.py` — Evaluation and plotting scripts
- `*.pth` — Pretrained model weights

## 🚀 Getting Started

### Install dependencies

### 🔧 Custom Training Example

To train the model with your own dataset and pretrained weights, run:

```bash
python vit/train_five_brain.py \
  --data-path /path/to/your/dataset \
  --weights /path/to/pretrained_model.pth \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.001 \
  --lr-f 0.01 \
  --freeze-layers False


## 🧪 Evaluate the Model (Confusion Matrix)

To evaluate the trained model and visualize the confusion matrix, run the following script:

```bash
python vit/create_confusion_matrix.py \
  --data-path /path/to/your/dataset \
  --weights /path/to/your/trained_model.pth \
  --device cuda:0 \
  --batch-size 2 \
  --num-classes 2


  