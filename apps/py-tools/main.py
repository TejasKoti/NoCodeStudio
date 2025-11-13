"""
FastAPI backend for No-Code Studio (Ze greatest creation)
Handles:
 • Catalog (auto layer discovery)
 • Graph ↔ Code export/import
 • Model execution (/api/run)
 • Model training (/api/train)

Author: TejasKoti
"""

# IMPORTS
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import re
import io
import torch
import torch.nn as nn
from contextlib import redirect_stdout
import traceback
import base64
import sys

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import io, sys, traceback, base64, torch, torch.nn as nn, re
from contextlib import redirect_stdout

app = FastAPI(title="No-Code DL API", version="1.1")

# CORD Middleware (Scrapped - Never worked)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("CORS middleware active")

from parser import get_torch_layers, get_layer_params
from trainer import set_stop_flag

# MODELS
class RunRequest(BaseModel):
    code: str
    input: Optional[list] = None
    dataset: Optional[str] = "CIFAR10"


class GraphData(BaseModel):
    title: Optional[str] = "UntitledProject"
    graph: dict

# HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "ok"}

# CATALOG & LAYER INFO
@app.get("/catalog")
def catalog():
    return {"layers": get_torch_layers()}


@app.get("/layer/{layer_name}")
def layer_info(layer_name: str):
    return {"name": layer_name, "params": get_layer_params(layer_name)}

# CANCEL TRAINING
@app.post("/cancel_training")
def cancel_training():
    set_stop_flag(True)
    print("Training cancellation requested by user.")
    sys.stdout.flush()
    return {"status": "cancelled"}

# EXPORT GRAPH -> PYTORCH CODE
@app.post("/export")
def export_project(data: dict = Body(...)):
    project_title = data.get("title", "Untitled Project")
    graph = data.get("graph", {}) or {}
    nodes = graph.get("nodes", []) or []

    def get_label_and_params(node: dict):
        d = node.get("data") or {}
        label = (d.get("label") or node.get("label") or node.get("type") or "").strip()
        params = d.get("params") or node.get("params") or {}
        return label, params

    lines = [
        f"# Auto-generated from project: {project_title}",
        "import torch",
        "import torch.nn as nn",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    valid_idx = []
    for i, node in enumerate(nodes):
        label, params = get_label_and_params(node)
        if not label or label.lower() in {"default", "none", "undefined", "input", "output"}:
            valid_idx.append(None)
            continue

        rendered_args = []
        for k, v in (params.items() if isinstance(params, dict) else []):
            if k.startswith("arg"):
                rendered_args.append(str(v))
            elif v is None or v == "":
                continue
            else:
                rendered_args.append(f"{k}={v}")
        arg_str = ", ".join(rendered_args)

        lines.append(f"        self.layer_{i} = nn.{label}({arg_str})")
        valid_idx.append(i)

    lines += [
        "",
        "    def forward(self, x):",
    ]

    for i, node in enumerate(nodes):
        label, _ = get_label_and_params(node)
        if not label or label.lower() in {"default", "none", "undefined", "input", "output"}:
            continue
        lines.append(f"        x = self.layer_{i}(x)")

    lines += [
        "        return x",
        "",
        "# Instantiate model",
        "model = Model()",
        "print(model)",
    ]

    code = "\n".join(lines)
    return {"filename": "model.py", "code": code}

# IMPORT PY FILE -> GRAPH
@app.post("/import")
async def import_model(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8")
    pattern = r"self\.(\w+)\s*=\s*nn\.(\w+)\((.*?)\)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    nodes, edges, y = [], [], 0
    for i, (_, layer, params) in enumerate(matches):
        params_list = [p.strip() for p in params.split(",") if p.strip()]
        param_dict = {}
        for idx, p in enumerate(params_list):
            if "=" in p:
                k, v = p.split("=", 1)
                param_dict[k.strip()] = v.strip()
            else:
                param_dict[f"arg{idx+1}"] = p.strip()
        nodes.append({
            "id": str(i),
            "type": "default",
            "position": {"x": 200, "y": y},
            "data": {"label": layer, "params": param_dict},
        })
        if i > 0:
            edges.append({"id": f"e{i-1}-{i}", "source": str(i-1), "target": str(i)})
        y += 80

    return {"graph": {"nodes": nodes, "edges": edges}}

# RUN MODEL (Main Inference)
@app.post("/api/run")
async def run_model(req: RunRequest):
    buffer = io.StringIO()
    try:
        local_env = {}
        exec(req.code, {"__builtins__": __builtins__, "torch": torch, "nn": nn}, local_env)

        ModelClass = next((v for v in local_env.values()
                           if isinstance(v, type) and issubclass(v, nn.Module)), None)
        if not ModelClass:
            raise RuntimeError("No nn.Module subclass found in code.")

        model = ModelClass()
        model.eval()

        x = torch.tensor(req.input, dtype=torch.float32) if req.input else torch.randn(1, 3, 32, 32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, x = model.to(device), x.to(device)

        with torch.no_grad():
            out = model(x)

        output_data = out.tolist() if isinstance(out, torch.Tensor) else [str(out)]
        print("Model executed successfully")
        print("Output shape:", getattr(out, "shape", None))

        return {"output": output_data, "stdout": buffer.getvalue(), "error": None}

    except Exception as e:
        tb = traceback.format_exc()
        print("Run failed:", e)
        return {"output": None, "error": f"{e}\n{tb}", "stdout": buffer.getvalue()}

# TRAIN MODEL
from threading import Thread
import asyncio
from trainer import set_stop_flag, train_model as run_training

@app.post("/api/train")
async def train_model(req: dict = Body(...)):
    import base64, traceback, torch, torch.nn as nn
    result = {
        "stdout": "",
        "error": None,
        "modelBase64": None,
        "metrics": None
    }

    try:
        code = req.get("code", "")
        config = req.get("config", {}) or {}
        dataset = req.get("dataset")
        dataset_options = req.get("datasetOptions", {}) or {}
        local_env = {}
        exec(code, {"__builtins__": __builtins__, "torch": torch, "nn": nn}, local_env)

        ModelClass = next(
            (v for v in local_env.values()
             if isinstance(v, type) and issubclass(v, nn.Module)),
            None,
        )
        if not ModelClass:
            raise RuntimeError("No nn.Module subclass found in code.")

        model = ModelClass()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded with {num_params} parameters.")

        # Training config
        epochs = int(config.get("epochs", config.get("Epochs", 3)))
        lr = float(config.get("learningRate", config.get("LearningRate", 1e-3)))
        batch_size = int(config.get("batchSize", config.get("BatchSize", 8)))
        print(f"Training config → epochs={epochs}, lr={lr}, batch_size={batch_size}")

        # Reset stop flag (Makeshift lol)
        set_stop_flag(False)

        def train_thread():
            try:
                logs, final_loss, model_path, metrics = run_training(
                    model,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    dataset=dataset,
                    data_root="./data",
                )

                result["stdout"] = logs or ""
                result["metrics"] = metrics or {}

                # Encode model if saved
                if model_path:
                    with open(model_path, "rb") as f:
                        result["modelBase64"] = base64.b64encode(f.read()).decode("utf-8")

                print("Training complete.")

            except Exception as e:
                tb = traceback.format_exc()
                result["error"] = f"Training crashed: {e}\n{tb}"

        # Start training in background
        t = Thread(target=train_thread, daemon=True)
        t.start()

        # Wait while training is running
        while t.is_alive():
            await asyncio.sleep(0.2)

        # If was cancelled mid-way
        if result["error"] is None and result["modelBase64"] is None and result["metrics"] is None:
            result["stdout"] = "Training stopped successfully"

        return result

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "stdout": "",
            "error": f"Training failed: {e}\n\n{tb}",
            "modelBase64": None,
            "metrics": None,
        }