# models/registry.py
MODEL_REGISTRY = {}

def register_model(name, default_weights=None):
    def decorator(cls):
        MODEL_REGISTRY[name] = {
            "class": cls,
            "default_weights": default_weights
        }
        return cls
    return decorator

def load_models(model_names, weight_paths=None):
    models = []

    for i, name in enumerate(model_names):
        model_info = MODEL_REGISTRY[name]
        ModelClass = model_info["class"]

        # choose user-provided weight or default
        weights = (
            weight_paths[i]
            if weight_paths and i < len(weight_paths)
            else model_info["default_weights"]
        )

        print(f"→ Loading {name} with weights: {weights}")
        model = ModelClass(weights=weights)
        models.append(model)

    return models