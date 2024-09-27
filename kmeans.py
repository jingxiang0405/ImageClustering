from sklearn.cluster import KMeans


def apply_kmeans(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels


def generate_html_with_kmeans(
    image_names,
    tsne_results,
    cluster_labels,
    output_file="kmeans.html",
    img_dir="images",
):
    with open(output_file, "w") as f:
        f.write("<html><head><title>Image t-SNE Visualization</title></head>\n")
        f.write("<style>\n")
        f.write("""
            .container { position: relative; }
            img { position: absolute; width: 50px; height: 50px; border: 3px solid; }
            .cluster-0 { border-color: red; }
            .cluster-1 { border-color: blue; }
            .cluster-2 { border-color: green; }
            .cluster-3 { border-color: yellow; }
            .cluster-4 { border-color: purple; }
        """)
        f.write("</style>\n")
        f.write("<body>\n")
        f.write('<div class="container">\n')

        for i, name in enumerate(image_names):
            x = tsne_results[i, 0] * 50
            y = tsne_results[i, 1] * 50

            cluster = cluster_labels[i]
            f.write(f'<img src="{
                    name}" class="cluster-{cluster}" style="left:{x}px;top:{y}px;">\n')

        f.write("</div></body></html>")
