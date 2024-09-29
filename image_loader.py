import os
from PIL import Image


def load_images_with_tag_from_directory(directory, transform, device):
    image_list = []
    image_names = []

    for entry in os.listdir(directory):
        entry = os.path.join(directory, entry)

        if entry.endswith(".png"):
            img = Image.open(entry).convert("RGB")
            img = transform(img).unsqueeze(0)  # Add batch dimension
            image_list.append(img.to(device))
            image_names.append(entry)

    return image_list, image_names
