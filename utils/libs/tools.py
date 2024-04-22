import numpy as np
from torchvision import transforms

def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))


def get_image_transformation():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def seconds_to_hoursminsec(amount):
    amount = int(amount)
    hours = str(amount // 3600).rjust(2, '0')
    mins = str((amount % 3600) // 60).rjust(2, '0')
    secs = str((amount % 3600) % 60).rjust(2, '0')
    return f"{hours}:{mins}:{secs}"