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

# STOP FLAG (MUST be on the shared Modal Volume mount!)
STOP_FLAG_PATH = "/data/STOP_TRAINING.flag"

_LAST_RELOAD = 0.0
_RELOAD_EVERY_SEC = 1.0

def _maybe_reload_volume():
    global _LAST_RELOAD
    now = time.time()
    if now - _LAST_RELOAD < _RELOAD_EVERY_SEC:
        return
    _LAST_RELOAD = now

    try:
        import modal
        modal.Volume.from_name("nocodestudio-data").reload()
    except Exception:
        pass

def set_stop_flag():
    with open(STOP_FLAG_PATH, "w") as f:
        f.write("STOP")

def clear_stop_flag():
    if os.path.exists(STOP_FLAG_PATH):
        os.remove(STOP_FLAG_PATH)

def should_stop():
    _maybe_reload_volume()
    return os.path.exists(STOP_FLAG_PATH)

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
                        print(f"Auto-fixed {name}.in_features → {correct_in}")
                        sys.stdout.flush()
                        break
        return model

# Training loop with Dashboard & Json Log Style
def train_model(
    model: nn.Module,
    config: dict | None = None,
    dataset: str = None,
    data_root: str = "/data"
):
    # IMPORTANT: clear stop flag at the start of a NEW training run
    clear_stop_flag()

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        config = config or {}
        epochs = int(config.get("Epochs", 3))
        batch_size = int(config.get("BatchSize", 8))
        lr = float(config.get("LearningRate", 1e-3))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()

        # Ensure params are trainable
        for p in model.parameters():
            p.requires_grad = True

        # Auto adjust first conv if needed
        model = auto_adjust_first_conv(model, dataset, device)

        loader, criterion = None, None

        # ---------------- DATASET ----------------
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
                loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
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
                loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
                criterion = nn.CrossEntropyLoss()

        except Exception as e:
            print(f"Dataset load failed: {e}")
            loader = None

        # ---------------- FALLBACK DATA ----------------
        if loader is None:
            print("No dataset — using synthetic data.")
            has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())

            if has_conv:
                input_shape = infer_input_shape(model)
                X = torch.randn(64, *input_shape)
            else:
                X = torch.randn(64, 10)

            with torch.no_grad():
                out = model(X[:1])

            if not isinstance(out, torch.Tensor):
                print("Model output invalid — cannot train.")
                return buffer.getvalue(), None, None, None

            if out.ndim > 2:
                out = out.flatten(1)

            if out.ndim != 2 or out.shape[1] < 2:
                print("Model needs a Linear(out_features=classes) layer.")
                return buffer.getvalue(), None, None, None

            y = torch.randint(0, out.shape[1], (64,))
            loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
            criterion = nn.CrossEntropyLoss()

        # ---------------- AUTO FIX LINEAR ----------------
        try:
            sample = next(iter(loader))[0][:1].to(device)
            model = auto_adjust_linear_layers(model, sample)
        except Exception:
            pass

        # ---------------- FINAL PARAM CHECK ----------------
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total params: {total_params} | Trainable params: {trainable_params}")

        if total_params == 0:
            print(
                "No parameters found on the model.\n"
                "This usually means your exported model had no layers created.\n"
                "Fix: ensure Conv2d/Linear nodes have required params filled (in_channels/out_channels/kernel_size, etc)."
            )
            return buffer.getvalue(), None, None, None

        if trainable_params == 0:
            print(
                "Model has parameters, but none are trainable (requires_grad=False).\n"
                "Fix: ensure you didn't freeze params or wrap forward in torch.no_grad()."
            )
            return buffer.getvalue(), None, None, None

        print(f"Training with {trainable_params} trainable parameters.")
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

        # ---------------- TRAIN LOOP ----------------
        metrics = {"loss": [], "batch_times": [], "speed": [], "eta": []}
        last_times = []
        total_batches = len(loader) * epochs
        processed = 0

        for epoch in range(epochs):
            if should_stop():
                print("Training cancelled by user (epoch check).")
                sys.stdout.flush()
                return buffer.getvalue(), None, None, metrics

            running_loss = 0.0

            for inputs, targets in loader:
                if should_stop():
                    print("Training cancelled by user (batch check).")
                    sys.stdout.flush()
                    return buffer.getvalue(), None, None, metrics

                start = time.time()
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                if outputs.ndim > 2:
                    outputs = outputs.flatten(1)

                # Attach classifier head ONCE if missing
                if outputs.ndim != 2:
                    if not hasattr(model, "_auto_head"):
                        in_features = outputs.shape[1]
                        num_classes = int(targets.max().item()) + 1
                        model._auto_head = nn.Linear(in_features, num_classes).to(device)
                        optimizer.add_param_group({"params": model._auto_head.parameters()})
                        print(f"Auto-added classifier: {in_features} → {num_classes}")

                    outputs = model._auto_head(outputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                dt = time.time() - start
                last_times.append(dt)
                if len(last_times) > 20:
                    last_times.pop(0)

                avg = sum(last_times) / len(last_times)
                speed = 1.0 / avg if avg > 0 else 0
                processed += 1
                eta = (total_batches - processed) * avg

                metrics["loss"].append(loss.item())
                metrics["batch_times"].append(dt)
                metrics["speed"].append(speed)
                metrics["eta"].append(eta)

                print(
                    f"Step {processed}/{total_batches} | "
                    f"loss={loss.item():.4f} | "
                    f"{speed:.2f} batch/s | ETA {int(eta)}s"
                )
                print(json.dumps({
                    "step": processed,
                    "loss": loss.item(),
                    "batch_time": dt,
                    "speed": speed,
                    "eta": eta
                }))
                sys.stdout.flush()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} | avg_loss={running_loss / max(1, len(loader)):.4f}")
            sys.stdout.flush()

        # ---------------- SAVE (only if not cancelled) ----------------
        print("\nTraining complete.")
        os.makedirs("outputs", exist_ok=True)
        model_path = os.path.join("outputs", f"trained_model_{dataset or 'custom'}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        sys.stdout.flush()

        final_loss = sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else None
        return buffer.getvalue(), final_loss, model_path, metrics

    finally:
        sys.stdout = old_stdout