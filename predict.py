"""Evaluate a saved CNN checkpoint and export reproducible MNIST figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Net

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_samples(samples, path, title, correct):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    for index, ax in enumerate(axes.flat):
        if index >= len(samples):
            ax.axis("off")
            continue
        image, truth, prediction = samples[index]
        ax.imshow(image.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        color = "#176b3a" if correct else "#b42318"
        ax.set_title(f"True: {truth}  |  Pred: {prediction}", color=color)
        ax.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion(matrix, path, normalized=False):
    values = matrix.float()
    if normalized:
        values = values / values.sum(dim=1, keepdim=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    image = ax.imshow(values.numpy(), cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=ax, shrink=0.88)
    title = "Row-normalized confusion matrix" if normalized else "MNIST confusion matrix"
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted label", ylabel="True label", title=title)
    threshold = float(values.max()) * 0.55
    for row in range(10):
        for col in range(10):
            value = float(values[row, col])
            label = f"{100 * value:.1f}%" if normalized else str(int(values[row, col]))
            ax.text(col, row, label, ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "#182230")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the MNIST test split")
    parser.add_argument("--model", default="models/my_mnist_can.pt")
    parser.add_argument("--data-dir", default="data", help="MNIST data root; downloads missing files")
    parser.add_argument("--output-dir", default="assets/figures")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-accel", action="store_true", help="force CPU evaluation")
    args = parser.parse_args()

    model_path = project_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_accel else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST(project_path(args.data_dir), train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    matrix = torch.zeros((10, 10), dtype=torch.int64)
    wrong_samples, correct_samples = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            matrix += torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
            for index, (truth, prediction) in enumerate(zip(labels.tolist(), predictions.tolist())):
                sample = (images[index].cpu() * 0.3081 + 0.1307, truth, prediction)
                if truth != prediction and len(wrong_samples) < 10:
                    wrong_samples.append(sample)
                elif truth == prediction and len(correct_samples) < 10:
                    correct_samples.append(sample)

    total = int(matrix.sum())
    correct = int(matrix.diag().sum())
    accuracy = 100.0 * correct / total
    errors = total - correct
    print(f"Test accuracy: {correct}/{total} ({accuracy:.2f}%), errors: {errors}")
    print(f"Confusion matrix and example figures saved to: {output_dir}")

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/pred"] + list(range(10)))
        for label, row in enumerate(matrix.tolist()):
            writer.writerow([label] + row)
    with (output_dir / "class_accuracy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["digit", "correct", "support", "accuracy_percent"])
        for label in range(10):
            support = int(matrix[label].sum())
            count = int(matrix[label, label])
            writer.writerow([label, count, support, f"{100 * count / support:.4f}"])

    draw_samples(correct_samples, output_dir / "correct_predictions.png", "Correctly classified MNIST examples", True)
    draw_samples(wrong_samples, output_dir / "misclassified_examples.png", "Misclassified MNIST examples", False)
    save_confusion(matrix, output_dir / "confusion_matrix.png")
    save_confusion(matrix, output_dir / "normalized_confusion_matrix.png", normalized=True)
    report = {
        "checkpoint": model_path.name,
        "checkpoint_sha256": checkpoint_hash(model_path),
        "test_examples": total,
        "correct": correct,
        "errors": errors,
        "accuracy_percent": round(accuracy, 4),
        "device": str(device),
        "per_class_support": [int(value) for value in matrix.sum(dim=1).tolist()],
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
