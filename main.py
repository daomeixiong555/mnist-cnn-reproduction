"""Adapted PyTorch examples/mnist training and evaluation entry point."""

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


def make_loaders(data_dir, batch_size, test_batch_size, device, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    common = {"num_workers": num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **common)
    return train_loader, test_loader


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            if args.dry_run:
                break


def evaluate(model, device, test_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            correct += output.argmax(dim=1).eq(target).sum().item()
    count = len(test_loader.dataset)
    accuracy = 100.0 * correct / count
    print(f"Test set: Average loss: {total_loss / count:.4f}, Accuracy: {correct}/{count} ({accuracy:.2f}%)")
    return correct, count


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST CNN example reproduction")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--data-dir", default="data", help="dataset directory; relative paths use the project root")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-accel", action="store_true", help="force CPU execution")
    parser.add_argument("--dry-run", action="store_true", help="train one batch per epoch")
    parser.add_argument("--save-model", action="store_true", help="save the final state_dict")
    parser.add_argument("--output", default="runs/pytorch_examples_reproduction.pt")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing the selected output file")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_accel else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = make_loaders(
        project_path(args.data_dir), args.batch_size, args.test_batch_size, device, args.num_workers
    )
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        output = project_path(args.output)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to replace existing checkpoint: {output} (pass --overwrite to allow it)")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output)
        print(f"Saved state_dict: {output}")


if __name__ == "__main__":
    main()
