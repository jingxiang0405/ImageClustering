import tag
import numpy as np


def generate_html(image_names, results, output, clusters=None):
    image_tags = tag.generate_image_tag()
    with open(output, "w") as f:
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

        if clusters is not None:
            f.write("""
            .container { position: relative; }
            img { position: absolute; width: 50px; height: 50px; border: 3px solid; }
            .cluster-0 { border-color: red; }
            .cluster-1 { border-color: blue; }
            .cluster-2 { border-color: green; }
            .cluster-3 { border-color: yellow; }
            .cluster-4 { border-color: purple; }
            .cluster-5 { border-color: black; }
             # """)
        f.write("</style>\n")
        f.write("<body>\n")
        f.write('<div class="container">\n')

        for i, name in enumerate(image_names):
            x = results[i, 0] * 200  # Scale the x-coordinate
            y = results[i, 1] * 200  # Scale the y-coordinate

            # TODO logic encapsulation
            # Get pictures' tags
            key = name.split("/")[-1]
            tags = image_tags[i][key]
            tag_str = f"Tags: {tags[0]}, {tags[1]}"

            if clusters is None:

                f.write(
                    f'<img src="../{name}" style="left:{x}px;top:{y}px;">\n')
            else:
                cluster = clusters[i]
                f.write(f'<img src="../{
                    name}" class="cluster-{cluster}" style="left:{x}px;top:{y}px;">\n')

            f.write(f'<div class="tooltip" style="left:{
                    x+60}px;top:{y}px;">{tag_str}</div>\n')

        f.write("</div></body></html>")
