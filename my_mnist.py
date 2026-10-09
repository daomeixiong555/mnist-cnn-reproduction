"""Train the hand-written five-epoch MNIST CNN reproduction."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Net

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def run_epoch(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(images), labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}: {batch_idx * len(images)}/{len(loader.dataset)} images, loss={loss.item():.6f}")


def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += F.nll_loss(output, labels, reduction="sum").item()
            correct += output.argmax(dim=1).eq(labels).sum().item()
    count = len(loader.dataset)
    print(f"Test: average loss={total_loss / count:.4f}, correct={correct}/{count}, accuracy={100 * correct / count:.2f}%")
    return correct


def main():
    parser = argparse.ArgumentParser(description="Train the independently reimplemented MNIST CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", default="data", help="dataset directory; relative paths use the project root")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-accel", action="store_true", help="force CPU execution")
    parser.add_argument("--output", default="runs/new_mnist_cnn.pt")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing the selected output file")
    args = parser.parse_args()

    output = project_path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing checkpoint: {output} (pass --overwrite to allow it)")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_accel else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_dir = project_path(args.data_dir)
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)
        scheduler.step()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved state_dict: {output}")


if __name__ == "__main__":
    main()
