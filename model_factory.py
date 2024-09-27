from torchvision import models


def create(type, device):
    if type == "vgg19":
        return models.vgg19().features.to(device)
    elif type == "vit":
        return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)
