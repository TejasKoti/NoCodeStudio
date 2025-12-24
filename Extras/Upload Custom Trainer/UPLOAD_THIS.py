import io
import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

# STOP FLAG (Makeshift ez)
STOP_FLAG_PATH = "stop.flag"
stop_training = False


def set_stop_flag(value: bool):
    global stop_training
    stop_training = value

    if value:
        with open(STOP_FLAG_PATH, "w") as f:
            f.write("STOP")
        print("stop_training flag set to True (file written)")
    else:
        if os.path.exists(STOP_FLAG_PATH):
            os.remove(STOP_FLAG_PATH)
        print("stop_training flag cleared (file removed)")

    sys.stdout.flush()

def should_stop():
    return stop_training or os.path.exists(STOP_FLAG_PATH)

# Infer input shape for conv-based models
def infer_input_shape(model: nn.Module):
    for size in [64, 48, 32, 28, 24, 20, 16, 14, 12, 10, 8]:
        try:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                _ = model(x)
            return (3, size, size)
        except Exception:
            continue
    return (10,)


# Auto-adjust first Conv2D layer
def auto_adjust_first_conv(model: nn.Module, dataset: str, device="cpu"):
    try:
        first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
        if not first_conv:
            return model

        if dataset and dataset.upper() == "MNIST" and first_conv.in_channels == 3:
            print("Adjusting first Conv2d layer from 3→1 channels for MNIST...")
            new_conv = nn.Conv2d(
                1, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)
                if first_conv.bias is not None:
                    new_conv.bias[:] = first_conv.bias
            model.layer_0 = new_conv.to(device)

        elif dataset and dataset.upper() == "CIFAR10" and first_conv.in_channels == 1:
            print("Adjusting first Conv2d layer from 1→3 channels for CIFAR10...")
            new_conv = nn.Conv2d(
                3, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:] = first_conv.weight.repeat(1, 3, 1, 1) / 3
                if first_conv.bias is not None:
                    new_conv.bias[:] = first_conv.bias
            model.layer_0 = new_conv.to(device)
    except Exception as e:
        print(f"Could not auto-adjust input channels: {e}")

    return model


# Auto-adjust Linear in_features
def auto_adjust_linear_layers(model: nn.Module, sample_input: torch.Tensor):
    try:
        with torch.no_grad():
            _ = model(sample_input)
        return model
    except RuntimeError as e:
        if "mat1 and mat2 shapes cannot be multiplied" not in str(e):
            raise e

        print("Detected Linear shape mismatch — auto-adjusting...")
        sys.stdout.flush()

        x = sample_input
        for name, module in model.named_children():
            try:
                with torch.no_grad():
                    x = module(x)
            except RuntimeError as err:
                if "mat1 and mat2 shapes cannot be multiplied" in str(err):
                    flat = x.flatten(1)
                    correct_in = flat.shape[1]
                    if isinstance(module, nn.Linear):
                        out_features = module.out_features
                        module.in_features = correct_in
                        module.weight = nn.Parameter(torch.randn(out_features, correct_in))
                        module.bias = nn.Parameter(torch.zeros(out_features))
                        print(f"🛠️ Auto-fixed {name}.in_features → {correct_in}")
                        sys.stdout.flush()
                        break
        return model


# Training loop with Dashboard & Json Log Style
def train_model(
    model: nn.Module,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    dataset: str = None,
    data_root: str = "./data"
):
    global stop_training
    stop_training = False

    if os.path.exists(STOP_FLAG_PATH):
        os.remove(STOP_FLAG_PATH)

    buffer = io.StringIO()
    sys.stdout = buffer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    if num_params == 0:
        print("No trainable parameters found — skipping training.")
        sys.stdout.flush()
        return buffer.getvalue(), None, None, None

    print(f"Model loaded with {num_params} parameters.")
    sys.stdout.flush()

    # Auto adjust first conv if needed
    model = auto_adjust_first_conv(model, dataset, device)

    loader, criterion = None, None

    # Load Dataset
    try:
        if dataset and dataset.upper() == "CIFAR10":
            print("Using CIFAR10 dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            trainset = torchvision.datasets.CIFAR10(
                root=data_root, train=True, download=True, transform=transform
            )
            loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            criterion = nn.CrossEntropyLoss()

        elif dataset and dataset.upper() == "MNIST":
            print("Using MNIST dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            trainset = torchvision.datasets.MNIST(
                root=data_root, train=True, download=True, transform=transform
            )
            loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            criterion = nn.CrossEntropyLoss()

    except Exception as e:
        print(f"Failed to load dataset ({dataset}): {e}")
        loader = None

    # Random Data Fallback
    if loader is None:
        print("No real dataset — using dummy random data.")
        has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
        if has_conv:
            input_shape = infer_input_shape(model)
            print(f"📏 Auto-detected input shape: {input_shape}")
            X = torch.randn(64, *input_shape)
        else:
            X = torch.randn(64, 10)

        with torch.no_grad():
            out = model(X[:1])
        if isinstance(out, torch.Tensor) and out.ndim == 2 and out.shape[1] > 1:
            y = torch.randint(0, out.shape[1], (64,))
            criterion = nn.CrossEntropyLoss()
        else:
            y = torch.randn_like(out)
            criterion = nn.MSELoss()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Automatically Fix Linear Shapes
    try:
        sample = next(iter(loader))[0][:1].to(device)
        model = auto_adjust_linear_layers(model, sample)
    except Exception as fix_err:
        print(f"Auto-fix skipped: {fix_err}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Metrics Segment
    metrics = { "loss": [], "batch_times": [], "speed": [], "eta": [] }
    last_20_times = []

    model.train()
    total_batches = len(loader) * epochs
    processed_batches = 0

    for epoch in range(epochs):
        if should_stop():
            print("Training stopped by user (epoch-level check).")
            sys.stdout.flush()
            break

        running_loss = 0.0

        for i, (inputs, targets) in enumerate(loader):
            if should_stop():
                print("Training stopped by user (batch-level check).")
                sys.stdout.flush()
                break

            batch_start = time.time()

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Stats & Metrics
            batch_time = time.time() - batch_start
            last_20_times.append(batch_time)
            if len(last_20_times) > 20:
                last_20_times.pop(0)

            avg_batch_time = sum(last_20_times) / len(last_20_times)
            batch_speed = 1.0 / avg_batch_time if avg_batch_time > 0 else 0

            processed_batches += 1
            remaining = total_batches - processed_batches
            eta_seconds = remaining * avg_batch_time

            metrics["loss"].append(loss.item())
            metrics["batch_times"].append(batch_time)
            metrics["speed"].append(batch_speed)
            metrics["eta"].append(eta_seconds)

            # Human Log
            print(f"Step {processed_batches}/{total_batches} "
                  f"| loss={loss.item():0.4f} "
                  f"| {batch_speed:.2f} batch/s "
                  f"| ETA {int(eta_seconds)}s")

            # Json Log
            print(json.dumps({
                "step": processed_batches,
                "loss": loss.item(),
                "batch_time": batch_time,
                "speed": batch_speed,
                "eta": eta_seconds
            }))
            sys.stdout.flush()

            running_loss += loss.item()

        if should_stop():
            print("\nTraining cancelled early by user.\n")
            sys.stdout.flush()
            return buffer.getvalue(), None, None, metrics

        epoch_avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | avg_loss={epoch_avg_loss:.4f}")
        sys.stdout.flush()

    # End Of Training
    print("\nTraining complete!\n")
    sys.stdout.flush()

    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", f"trained_model_{dataset or 'custom'}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    sys.stdout.flush()

    final_avg_loss = sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else None
    return buffer.getvalue(), final_avg_loss, model_path, metrics