from __future__ import annotations

import importlib


def load_component(spec: dict):
    module_name = spec["module"]
    class_name = spec["class"]
    params = dict(spec.get("params") or {})

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)
