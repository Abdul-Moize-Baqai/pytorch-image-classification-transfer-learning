

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_image(image_path):
    """Use the supplied image or download a sample photo once."""
    if image_path is not None:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path
    path = Path("data") / "dog.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading example image to {path} ...")
        try:
            urlretrieve(SAMPLE_URL, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the sample image. Run with --image path/to/photo.jpg"
            ) from exc
    return path


def predict_top5(image_path, device):
    """Run five pretrained ImageNet classifiers, loading one at a time."""
    specifications = [
        ("VGG16", models.vgg16, models.VGG16_Weights.DEFAULT),
        ("ResNet50", models.resnet50, models.ResNet50_Weights.DEFAULT),
        ("AlexNet", models.alexnet, models.AlexNet_Weights.DEFAULT),
        ("ResNet101", models.resnet101, models.ResNet101_Weights.DEFAULT),
        ("MobileNetV2", models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    ]
    picture = Image.open(image_path).convert("RGB")
    results = {}
    for name, constructor, weights in specifications:
        print(f"\nLoading {name} (first use may download weights) ...")
        network = constructor(weights=weights).to(device)
        network.eval()  # Disable training-time behavior
        image_tensor = weights.transforms()(picture).unsqueeze(0).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(network(image_tensor), dim=1)[0]
            scores, indices = torch.topk(probabilities, k=5)

        classes = weights.meta["categories"]
        top5 = [(classes[i], float(score)) for score, i in
                zip(scores.cpu().tolist(), indices.cpu().tolist())]
        results[name] = top5
        print(f"{name} — top 5 ImageNet predictions")
        for rank, (label, score) in enumerate(top5, 1):
            print(f"  {rank}. {label:30s} {score * 100:6.2f}%")

        # Discard each model before loading the next large set of weights.
        del network, image_tensor, probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def train_cifar10_transfer(device, data_dir="data", epochs=3, train_samples=5000,
                           test_samples=1000, batch_size=64, output="cifar10_resnet18.pth"):
    """Freeze pretrained ResNet18 features and train a new CIFAR-10 classifier."""
    torch.manual_seed(42)
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                 transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True,
                                transform=test_transform)
    class_names = train_set.classes

    # Use deterministic subsets for a relatively quick run.
    train_indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(42))[
        :min(train_samples, len(train_set))].tolist()
    test_indices = torch.randperm(len(test_set), generator=torch.Generator().manual_seed(43))[
        :min(test_samples, len(test_set))].tolist()
    train_loader = DataLoader(Subset(train_set, train_indices), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(test_set, test_indices), batch_size=batch_size,
                             shuffle=False, num_workers=0)

    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in net.parameters():
        parameter.requires_grad = False  # Keep learned ImageNet visual features
    net.fc = nn.Linear(net.fc.in_features, len(class_names))  # 1000 -> 10 classes
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()  # Expects raw logits, not softmax
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        net.eval()  # Keep BatchNorm statistics frozen in the backbone.
        running_loss, train_correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)

        # Compute test accuracy without gradients.
        net.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = net(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch:2d}/{epochs} | train loss {running_loss / seen:.4f} "
              f"| train accuracy {100 * train_correct / seen:.2f}% "
              f"| test accuracy {100 * correct / total:.2f}%")

    checkpoint = {
        "state_dict": net.state_dict(),
        "classes": class_names,
        "architecture": "resnet18",
        "image_size": 160,
    }
    torch.save(checkpoint, output)
    print(f"Saved the transfer-learned model to {output}")

    # Display predictions for one held-out test image.
    image, true_label = test_set[test_indices[0]]
    with torch.inference_mode():
        probabilities = net(image.unsqueeze(0).to(device)).softmax(dim=1)[0]
        top_scores, top_indices = probabilities.topk(5)
    print(f"\nExample CIFAR-10 image | true label: {class_names[true_label]}")
    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), 1):
        print(f"  {rank}. {class_names[idx]:15s} {score * 100:6.2f}%")
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=str, default=None, help="Local RGB image for 5-model comparison")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = select_device()
    print("PyTorch:", torch.__version__, "| device:", device)
    if not args.skip_predictions:
        image_path = get_image(args.image)
        print("Comparison image:", image_path)
        predict_top5(image_path, device)
    if not args.skip_transfer:
        train_cifar10_transfer(device, epochs=args.epochs, train_samples=args.train_samples,
                               test_samples=args.test_samples, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
