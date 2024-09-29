import torch
import logging
import json
from torchvision.transforms import v2

# custom module
import image_loader
import model_factory
import kmeans
import view
import dimension

# Read configuration
with open("config.json", "r") as f:
    config = json.load(f)

kmeans_enabled = config["kmeans"].get("enabled", False)

if kmeans_enabled:
    kmeans_nclusters = config["kmeans"].get("ncluster", 5)

image_directory = config.get("image_directory", "./image")
output_html = config.get("output_html", "ImageClustering.html")
model_name = config.get("model", "vgg19")
dimension_reduction = config.get("dimension_reduction", "tsne")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(filename)s %(levelname)s:%(message)s"
)


def main():
    # Logging

    logging.info("Loading Vit Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory.create("vit", device)
    # Define the image transform (resize, normalize)
    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    logging.info("Loading images...")
    image_directory = config.get("image_directory", "./images")
    images, image_names = image_loader.load_images_with_tag_from_directory(
        image_directory, transform, device
    )

    logging.info("Extracting features using Vit...")
    features = model_factory.extract_features(images, model)

    logging.info("Applying dimension reduction...")
    dr_result = dimension.reduce(features, dimension_reduction)

    # TODO refactor structure
    if kmeans_enabled:
        logging.info(f"Applying K-Means clustering with {kmeans_nclusters} clusters...")
        kmeans_labels = kmeans.apply_kmeans(features, nclusters=kmeans_nclusters)

        logging.info("Generating HTML(K-Means)")
        view.generate_html(
            image_names=image_names,
            results=dr_result,
            output=output_html,
            clusters=kmeans_labels,
        )
    else:
        logging.info("Generating HTML(Normal)")
        view.generate_html(image_names, dr_result, output=output_html)

    logging.info(f"HTML visualization saved to {output_html}")


if __name__ == "__main__":
    main()
