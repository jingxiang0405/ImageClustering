import torch
import numpy as np
from torchvision import models, transforms
from sklearn.manifold import TSNE

# custom module
import data_loader
import tag


# Define a function to load images from a directory


# Function to extract features using VGG19 in PyTorch
def extract_features(images, model):
    model.eval()
    features = []
    with torch.no_grad():
        for img in images:
            feature = model(img)
            # Flatten the features
            features.append(feature.cpu().numpy().reshape(-1))
    return np.array(features)


# Function to apply t-SNE on the extracted features
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


# Function to generate HTML to visualize the images
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
            x = tsne_results[i, 0] * 50  # Scale the x-coordinate
            y = tsne_results[i, 1] * 50  # Scale the y-coordinate
            # Get pictures' tags
            tags = image_tags[i][name[7:]]
            tag_str = f"Tags: {tags[0]}, {tags[1]}"
            f.write(f'<img src="{name}" style="left:{x}px;top:{y}px;">\n')
            f.write(f'<div class="tooltip" style="left:{
                    x+60}px;top:{y}px;">{tag_str}</div>\n')

        f.write("</div></body></html>")


# Main function to run the entire process
def main(image_directory, output_html="output.html"):
    # Load pre-trained VGG19 model from torchvision and remove the classifier layer
    print("Loading VGG19 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg19_model = models.vgg19().features.to(device)

    # Define the image transform (resize, normalize)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )

    # Step 1: Load all PNG images from the specified directory
    print("Loading images...")
    images, image_names = data_loader.load_images_with_tag_from_directory(
        image_directory, transform, device
    )

    # Step 2: Extract VGG19 features
    print("Extracting features using VGG19...")
    features = extract_features(images, vgg19_model)

    # Step 3: Apply t-SNE for dimensionality reduction
    print("Applying t-SNE...")
    tsne_results = apply_tsne(features)

    # Step 4: Generate HTML to visualize the images
    print("Generating HTML...")
    generate_html(image_names, tsne_results, output_file=output_html)
    print(f"HTML visualization saved to {output_html}")


# Run the main function with the directory containing images
if __name__ == "__main__":
    image_directory = "images"  # Replace with your directory
    main(image_directory, output_html="tsne_visualization.html")
