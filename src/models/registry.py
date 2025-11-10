MODEL_REGISTRY = {}

def register_model(name, default_weights=None):

    def decorator(cls):
        MODEL_REGISTRY[name] = {
            "class": cls,
            "default_weights": default_weights
        }
        return cls
    return decorator

