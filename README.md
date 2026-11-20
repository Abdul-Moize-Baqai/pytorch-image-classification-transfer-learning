# Pretrained Image Classification & Transfer Learning with PyTorch

A computer vision project exploring two complementary workflows with **PyTorch** and **torchvision**:

- **Pretrained image classification:** compare top-5 ImageNet predictions from VGG16, ResNet50, AlexNet, ResNet101, and MobileNetV2 on the same image.
- **Transfer learning:** reuse the feature extractor from an ImageNet-pretrained ResNet18 and train a new classifier for the ten classes in CIFAR-10.

The project includes an executable Python script and a Jupyter notebook. It supports CUDA, Apple Silicon MPS (when available), and CPU execution.

## Models and datasets

| Workflow | Architecture | Pretraining / dataset | Output |
| --- | --- | --- | --- |
| Image classification | VGG16 | ImageNet-1K | Top 5 labels and softmax scores |
| Image classification | ResNet50 | ImageNet-1K | Top 5 labels and softmax scores |
| Image classification | AlexNet | ImageNet-1K | Top 5 labels and softmax scores |
| Image classification | ResNet101 | ImageNet-1K | Top 5 labels and softmax scores |
| Image classification | MobileNetV2 | ImageNet-1K | Top 5 labels and softmax scores |
| Transfer learning | ResNet18 | ImageNet weights → CIFAR-10 | 10-class predictions and test accuracy |

## Repository structure

```text
pytorch-image-classification-transfer-learning/
├── image_classification.py
├── image_classification_transfer_learning.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The script and notebook implement the same modeling workflows; they can be used independently.

## Getting started

**1. Clone this repository** (replace `<your-username>` after creating the GitHub repository):

```bash
git clone https://github.com/<your-username>/pytorch-image-classification-transfer-learning.git
cd pytorch-image-classification-transfer-learning
```

**2. Create and activate an environment** (recommended):

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

**3. Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

For GPU-specific PyTorch installations, follow the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to install a version compatible with your CUDA environment.

> An internet connection is needed initially to download pretrained weights and the CIFAR-10 dataset. Model weights, especially VGG16, can require significant disk space; running all five models is considerably more demanding than running just ResNet18.

## Running the project

### Compare pretrained architectures

Use the included sample image (downloaded automatically on the first run):

```bash
python image_classification.py --skip-transfer
```

Or provide a JPG/PNG file:

```bash
python image_classification.py --image path/to/photo.jpg --skip-transfer
```

For each architecture, the program prints five class labels and their corresponding softmax scores. It processes one model at a time to reduce memory pressure.

### Train the CIFAR-10 classifier

```bash
python image_classification.py --skip-predictions
```

The default training configuration is:

| Setting | Default |
| --- | ---: |
| Epochs | 3 |
| Training samples | 5,000 |
| Test samples | 1,000 |
| Batch size | 64 |
| Input resolution | 160 × 160 |
| Optimizer | Adam (learning rate 0.001) |
| Trained parameters | Replacement classification head only |

To use more samples and epochs:

```bash
python image_classification.py --skip-predictions --epochs 10 --train-samples 20000 --test-samples 5000 --batch-size 64
```

After training, a `cifar10_resnet18.pth` checkpoint is saved locally. Checkpoints and downloaded data are excluded from Git by `.gitignore`.

### Run both workflows

```bash
python image_classification.py
```

### Use the Jupyter notebook

```bash
jupyter lab image_classification_transfer_learning.ipynb
```

Run the cells from top to bottom. To use a local image, change `IMAGE_PATH` in the input-image cell. The notebook includes checkpoint reloading and a compact model-comparison table.

## How the methods work

**Pretrained inference** takes a single image, applies each model's recommended ImageNet preprocessing, runs the network in evaluation mode, applies softmax to the output logits, and reports the five highest-scoring categories. No training occurs in this workflow.

**Transfer learning** starts with ResNet18 weights learned on ImageNet. Its feature-extraction layers are frozen, and the final fully connected layer is replaced with a new ten-output head. That head learns CIFAR-10 class distinctions through supervised training with cross-entropy loss. The trained checkpoint retains the model weights and class metadata.

## Results and limitations

This repository intentionally does **not** claim fixed prediction results or a test-accuracy score: actual values should come from executing the models. Training loss, training accuracy, held-out test accuracy, and sample predictions are logged during execution.

- The default transfer-learning run uses **subsets** of CIFAR-10, so its reported test accuracy is not a full 10,000-image benchmark.
- CIFAR-10's 32×32 images are upscaled for the pretrained network; resizing does not add genuine visual detail.
- Freezing the ResNet18 backbone reduces training cost but may limit the accuracy achievable compared with full fine-tuning.
- Softmax values represent predicted class probabilities, **not guaranteed calibrated confidence**.
- Results and runtime depend on model-weight versions, hardware, random variation, and training configuration.

## References

- [PyTorch](https://pytorch.org/)
- [TorchVision pretrained models](https://pytorch.org/vision/stable/models.html)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Hub sample images](https://github.com/pytorch/hub/tree/master/images)

The pretrained model weights and dataset remain subject to their respective upstream licenses and terms.
