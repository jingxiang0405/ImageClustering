import torch
import numpy as np
import logging
from torchvision.transforms import v2
from sklearn.manifold import TSNE

# custom module
import data_loader
import tag
import model_factory
import kmeans


def extract_features(images, model):
    model.eval()
    features = []
    with torch.no_grad():
        for img in images:
            feature = model(img)

            # Flatten the features
            features.append(feature.cpu().numpy().reshape(-1))
    return np.array(features)


def apply_tsne(features):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Standardize
    min_x, min_y = tsne_results.min(axis=0)
    max_x, max_y = tsne_results.max(axis=0)
    tsne_results = (tsne_results - [min_x, min_y]
                    ) / [max_x - min_x, max_y - min_y]
    tsne_results = tsne_results * 15

    return tsne_results


def generate_html(image_names, tsne_results, output_file="output.html"):
    image_tags = tag.generate_image_tag()

    with open(output_file, "w") as f:
        f.write("<html><head><title>Image t-SNE Visualization</title></head>\n")

        # css style
        f.write("<style>\n")
        f.write("""
            .container { position: relative; }
            img { position: absolute; width: 50px; height: 50px; }
            .tooltip {
                position: absolute;
                background-color: #333;
                color: #fff;
                padding: 5px;
                border-radius: 5px;
                font-size: 12px;
                visibility: hidden;
                z-index: 10;
            }
            img:hover + .tooltip {
                visibility: visible;
            }
        """)
        f.write("</style>\n")
        f.write("<body>\n")
        f.write('<div class="container">\n')

        for i, name in enumerate(image_names):
            x = tsne_results[i, 0] * 100  # Scale the x-coordinate
            y = tsne_results[i, 1] * 100  # Scale the y-coordinate
            # Get pictures' tags
            tags = image_tags[i][name[7:]]
            tag_str = f"Tags: {tags[0]}, {tags[1]}"
            f.write(f'<img src="{name}" style="left:{x}px;top:{y}px;">\n')
            f.write(f'<div class="tooltip" style="left:{
                    x+60}px;top:{y}px;">{tag_str}</div>\n')

        f.write("</div></body></html>")


def main(image_directory, output_html="output.html"):
    FORMAT = "%(asctime)s %(filename)s %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    logging.info("Loading Vit Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory.create("vit", device)
    # Define the image transform (resize, normalize)
    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]
    )

    logging.info("Loading images...")
    images, image_names = data_loader.load_images_with_tag_from_directory(
        image_directory, transform, device
    )

    logging.info("Extracting features using Vit...")
    features = extract_features(images, model)

    # n_clusters = 12
    # logging.info(f"Applying K-Means clustering with {n_clusters} clusters...")
    # cluster_labels = kmeans.apply_kmeans(features, n_clusters=n_clusters)

    logging.info("Applying t-SNE...")
    tsne_results = apply_tsne(features)

    # Step 4: Generate HTML to visualize the images
    logging.info("Generating HTML...")
    generate_html(image_names, tsne_results, output_file=output_html)
    # kmeans.generate_html_with_kmeans(
    #     image_names,
    #     tsne_results,
    #     cluster_labels,
    #     output_file=output_html,
    #     img_dir=image_directory,
    # )

    logging.info(f"HTML visualization saved to {output_html}")


# Run the main function with the directory containing images
if __name__ == "__main__":
    image_directory = "images"  # Replace with your directory
    main(image_directory, output_html="tsne_visualization.html")
