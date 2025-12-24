import modal

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", "/root")
)

app = modal.App("pythonpipeline")

volume = modal.Volume.from_name(
    "nocodestudio-data",
    create_if_missing=True,
)

# -------- HEALTH --------
@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    return {"status": "ok"}

# -------- CATALOG --------
@app.function(image=image)
@modal.web_endpoint(method="GET")
def catalog():
    from parser import get_torch_layers
    return {"layers": get_torch_layers()}

# -------- LAYER PARAMS --------
@app.function(image=image)
@modal.web_endpoint(method="GET")
def layer(name: str):
    from parser import get_layer_params
    return {
        "name": name,
        "params": get_layer_params(name),
    }

# -------- EXPORT --------
@app.function(image=image)
@modal.web_endpoint(method="POST")
def export(payload: dict):
    from main import export_project
    return export_project(payload)

# -------- IMPORT --------
@app.function(image=image)
@modal.web_endpoint(method="POST")
def import_model(payload: dict):
    from main import import_model_from_code
    return import_model_from_code(payload)

# -------- RUN --------
@app.function(image=image)
@modal.web_endpoint(method="POST")
def run(payload: dict):
    from main import run_model
    return run_model(payload)

# -------- CANCEL TRAINING --------
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60,
    cpu=0.25,
)
@modal.web_endpoint(method="POST")
def cancel_training():
    from trainer import set_stop_flag
    set_stop_flag()
    volume.commit()
    return {"status": "stopping"}

# -------- TRAIN --------
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60,
    cpu=2,
)
@modal.web_endpoint(method="POST")
def train(payload: dict):
    from main import train_model
    return train_model(payload, data_root="/data")