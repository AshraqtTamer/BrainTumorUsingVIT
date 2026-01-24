# Brain Tumor Classification using Vision Transformer (ViT)

This repository contains a PyTorch Lightning implementation for classifying brain tumors using a pre-trained **Vision Transformer (ViT)** model. The project leverages the `timm` library for the model architecture and `pytorch-lightning` for a streamlined training workflow.

## 📌 Project Overview

The goal of this project is to automate the detection and classification of brain tumors from medical imagery. By utilizing a Vision Transformer (ViT-Base), the model can capture global dependencies in images more effectively than traditional Convolutional Neural Networks (CNNs).

## 📊 Dataset

The project uses the **Brain Tumor Dataset** hosted on Kaggle.

* **Classes:** 3 (e.g., Glioma, Meningioma, Pituitary tumor).
* **Source:** Downloaded automatically via `kagglehub`.
* **Preprocessing:** Images are resized to 224x224, normalized, and augmented with random horizontal flips and rotations to improve model generalization.

## 🛠️ Tech Stack

* **Framework:** [PyTorch](https://pytorch.org/)
* **High-Level Wrapper:** [PyTorch Lightning](https://www.pytorchlightning.ai/)
* **Model Library:** [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
* **Metrics:** [TorchMetrics](https://torchmetrics.readthedocs.io/)
* **Dataset Source:** [KaggleHub](https://github.com/Kaggle/kagglehub)

## 🚀 Model Architecture: Vision Transformer (ViT)

The model used is `vit_base_patch16_224`.

* **Patch Size:** 16x16.
* **Input Resolution:** 224x224.
* **Pre-training:** Uses ImageNet weights for transfer learning.
* **Optimization:** Fine-tuned with the Adam optimizer and Cross-Entropy Loss.

## ⚙️ Configuration

| Parameter | Value |
| --- | --- |
| Image Size | 224 x 224 |
| Batch Size | 16 |
| Epochs | 20 |
| Learning Rate | 1e-4 |
| Training/Val Split | 80% / 20% |

## 🏗️ How to Run

1. **Install Dependencies:**
```bash
pip install lightning timm torchmetrics kagglehub

```


2. **Run the Notebook/Script:**
The script will automatically download the dataset using `kagglehub`, initialize the ViT model, and begin the training process using the Lightning `Trainer`.
3. **Monitor Progress:**
The trainer logs `train_loss`, `train_acc`, `val_loss`, and `val_acc` to the console (and can be linked to TensorBoard or WandB).

## 📈 Results

The training pipeline includes:

* **Validation Step:** Evaluates the model after every epoch to prevent overfitting.
* **Reproducibility:** A global seed (42) is set using `seed_everything`.
* **Accuracy:** Tracked using `MulticlassAccuracy` from the TorchMetrics library.
