import torch
import inspect
import torch.nn.functional as F

def get_torch_layers():
    layers = {}

    for name, obj in torch.nn.__dict__.items():
        if inspect.isclass(obj):
            layers[name] = {
                "class": name,
                "function": None
            }

    for fn_name, fn_obj in F.__dict__.items():
        if callable(fn_obj) and not fn_name.startswith("_"):
            candidate = fn_name.capitalize()
            if candidate in layers:
                layers[candidate]["function"] = fn_name
            else:
                layers[fn_name] = {
                    "class": None,
                    "function": fn_name
                }
    return [ { "name": k, **v } for k,v in sorted(layers.items()) ]

def get_layer_params(layer_name: str):
    layer_cls = getattr(torch.nn, layer_name, None)
    if layer_cls is None:
        return []

    sig = inspect.signature(layer_cls.__init__)
    params = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        params.append({
            "name": pname,
            "default": None if param.default is inspect._empty else repr(param.default),
            "kind": str(param.kind),
        })
    return params