"""Classify a handwritten digit image after MNIST-style preprocessing."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from model import Net

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def prepare_image(path):
    image = Image.open(path).convert("L")
    original = image.copy()
    mean_pixel = sum(image.getdata()) / (image.width * image.height)
    if mean_pixel > 127:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    bbox = image.point(lambda value: 255 if value > 20 else 0).getbbox()
    if bbox is None:
        raise ValueError(f"No foreground digit found in {path}")
    image = image.crop(bbox)

    width, height = image.size
    scale = 20 / max(width, height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(new_height, new_width)
    total = pixels.sum()
    if total.item() == 0:
        raise ValueError(f"No nonzero digit pixels found in {path}")
    x_coords = torch.arange(new_width, dtype=torch.float32).reshape(1, new_width)
    y_coords = torch.arange(new_height, dtype=torch.float32).reshape(new_height, 1)
    center_x = (pixels * x_coords).sum() / total
    center_y = (pixels * y_coords).sum() / total
    left = round(13.5 - center_x.item())
    top = round(13.5 - center_y.item())
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(image, (left, top))

    tensor = torch.tensor(list(canvas.getdata()), dtype=torch.float32).reshape(1, 1, 28, 28)
    tensor = (tensor / 255.0 - 0.1307) / 0.3081
    return original, canvas, tensor


def main():
    parser = argparse.ArgumentParser(description="Predict one handwritten digit from a Pillow-supported image")
    parser.add_argument("--image", default="assets/my_digits.png", help="input image path")
    parser.add_argument("--model", default="models/my_mnist_can.pt", help="trained model state_dict")
    parser.add_argument("--output", default="assets/figures/custom_digit_prediction.png")
    parser.add_argument("--no-accel", action="store_true", help="force CPU inference")
    args = parser.parse_args()

    image_path = project_path(args.image)
    model_path = project_path(args.model)
    output_path = project_path(args.output)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_accel else "cpu")
    original, canvas, tensor = prepare_image(image_path)
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        output = model(tensor.to(device))
        probabilities = output.exp()[0]
        prediction = int(probabilities.argmax())
        confidence = float(probabilities[prediction]) * 100
        top_probabilities, top_labels = probabilities.topk(3)

    print(f"Device: {device}")
    print(f"Prediction: {prediction} (confidence: {confidence:.2f}%)")
    top_three = ", ".join(f"{int(label)}: {100 * float(prob):.2f}%" for prob, label in zip(top_probabilities, top_labels))
    print(f"Top-3: {top_three}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Input image")
    axes[1].imshow(canvas, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title(f"28 x 28 model input\nPrediction: {prediction} ({confidence:.2f}%)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
