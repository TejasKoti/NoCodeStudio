import io
import sys
import os
import re
import base64
import traceback
import torch
import torch.nn as nn

from trainer import clear_stop_flag, train_model as run_training


# ---------- EXPORT ----------
def export_project(data: dict):
    project_title = data.get("title", "Untitled Project")
    graph = data.get("graph", {}) or {}
    nodes = graph.get("nodes", []) or []

    import inspect

    def is_empty(v):
        return v is None or v == ""

    def normalize(v):
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            if s.isdigit():
                return int(s)
            try:
                return float(s)
            except ValueError:
                if s == "True":
                    return True
                if s == "False":
                    return False
                return s
        return v

    def get_signature(layer_name: str):
        cls = getattr(nn, layer_name, None)
        if cls is None:
            return []

        sig = inspect.signature(cls.__init__)
        out = []
        idx = 0
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            out.append({
                "name": name,
                "index": idx,
                "required": param.default is inspect._empty,
            })
            idx += 1
        return out

    lines = [
        f"# Auto-generated from project: {project_title}",
        "import torch",
        "import torch.nn as nn",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    built_layers = []

    for i, node in enumerate(nodes):
        d = node.get("data") or {}
        layer = (d.get("label") or "").strip()
        params = d.get("params") or {}

        if not layer:
            continue

        sig = get_signature(layer)
        if not sig:
            continue

        # Skip layer if required params missing
        missing = False
        for s in sig:
            if s["required"]:
                if s["name"] not in params or is_empty(params.get(s["name"])):
                    missing = True
                    break
        if missing:
            continue

        positional = []
        keyword = []

        for s in sig:
            name = s["name"]
            idx = s["index"]

            if name not in params:
                continue

            value = normalize(params.get(name))
            if value is None:
                continue

            if idx < 3:
                positional.append(repr(value))
            else:
                keyword.append(f"{name}={repr(value)}")

        if not positional and not keyword:
            continue

        args = ", ".join(positional + keyword)
        lines.append(f"        self.layer_{i} = nn.{layer}({args})")
        built_layers.append(i)

    lines += ["", "    def forward(self, x):"]

    for i in built_layers:
        lines.append(f"        x = self.layer_{i}(x)")

    lines.append("        return x")

    return {
        "filename": "model.py",
        "code": "\n".join(lines),
    }


# ---------- IMPORT ----------
def import_model_from_code(payload: dict):
    code = payload.get("code", "")
    pattern = r"self\.(\w+)\s*=\s*nn\.(\w+)\((.*?)\)"
    matches = re.findall(pattern, code, flags=re.DOTALL)

    nodes, edges, y = [], [], 0
    for i, (_, layer, params) in enumerate(matches):
        param_dict = {}
        for idx, p in enumerate(p.strip() for p in params.split(",") if p.strip()):
            if "=" in p:
                k, v = p.split("=", 1)
                param_dict[k.strip()] = v.strip()
            else:
                param_dict[f"arg{idx+1}"] = p

        nodes.append({
            "id": str(i),
            "type": "default",
            "position": {"x": 200, "y": y},
            "data": {"label": layer, "params": param_dict},
        })

        if i > 0:
            edges.append({
                "id": f"e{i-1}-{i}",
                "source": str(i - 1),
                "target": str(i),
            })
        y += 80

    return {"graph": {"nodes": nodes, "edges": edges}}


# ---------- RUN ----------
def run_model(payload: dict):
    try:
        local_env = {}
        exec(payload["code"], {"torch": torch, "nn": nn}, local_env)

        ModelClass = next(v for v in local_env.values() if isinstance(v, type))
        model = ModelClass().eval()

        raw_input = payload.get("input")
        if raw_input is None:
            x = torch.randn(1, 3, 32, 32)
        else:
            x = torch.tensor(raw_input, dtype=torch.float32)

        # Match input channels to first Conv2d if needed
        first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
        if first_conv and x.ndim == 4:
            if x.shape[1] != first_conv.in_channels:
                if first_conv.in_channels == 1:
                    x = x.mean(dim=1, keepdim=True)
                elif first_conv.in_channels == 3:
                    x = x.repeat(1, 3, 1, 1)

        with torch.no_grad():
            out = model(x)

        return {"output": out.tolist(), "error": None}

    except Exception:
        return {"output": None, "error": traceback.format_exc()}


# ---------- TRAIN ----------
def train_model(payload: dict, data_root="/data"):
    try:
        clear_stop_flag()

        local_env = {}
        exec(payload["code"], {"torch": torch, "nn": nn}, local_env)

        ModelClass = next(v for v in local_env.values() if isinstance(v, type))
        model = ModelClass()

        # Guard: empty model
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            return {
                "stdout": (
                    "Model has 0 parameters.\n"
                    "Your graph likely skipped all layers due to missing required params."
                ),
                "metrics": None,
                "modelBase64": None,
                "error": None,
            }

        logs, _, model_path, metrics = run_training(
            model,
            config=payload.get("config") or {},
            dataset=payload.get("dataset"),
            data_root=data_root,
        )

        encoded = None
        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

        return {
            "stdout": logs,
            "metrics": metrics,
            "modelBase64": encoded,
            "error": None,
        }

    except Exception:
        return {
            "stdout": "",
            "metrics": None,
            "modelBase64": None,
            "error": traceback.format_exc(),
        }