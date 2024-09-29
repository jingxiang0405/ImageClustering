from torchvision import models
import torch
import numpy as np


def create(type, device):
    if type == "vgg19":
        return models.vgg19().features.to(device)
    elif type == "vit":
        return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)


def extract_features(images, model):
    model.eval()
    features = []
    with torch.no_grad():
        for img in images:
            feature = model(img)

            # Flatten the features
            features.append(feature.cpu().numpy().reshape(-1))
    return np.array(features)
