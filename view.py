import tag


def generate_html(image_names, results, output, clusters=None):
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

            img:hover {
                z-index: 100;
                transform: scale(1.2);
            }

            input[type="checkbox"] {
                transform: scale(1.5);
                margin-left: 50px;
                margin-top : 50px;
            }
            label {
                font-size: 18px;
            }

            .form-inline label {
                display: inline-block;
                margin-right: 10px;
            }

            .form-inline input {
                display: inline-block;
                margin-right: 10px;
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
              """)
        f.write("</style>\n")
        f.write("<body>\n")

        # Checkbox for filtering
        f.write("""
        <div>
            <form id="tag-form" class="form-inline">
        """)

        # Dynamically get all tags
        image_tags = tag.generate_image_tag()
        max_length = max(len(v) for v in image_tags.values())
        tag_set = [set() for _ in range(max_length)]

        for values in image_tags.values():
            for i, value in enumerate(values):
                tag_set[i].add(value)

        for s in tag_set:
            for t in s:
                f.write(f"""
                        <input type="checkbox" name="tag" id="{t}" value="{t}" onchange="filterImages()">\n
                        <label for="{t}">{t}</label>\n

                    """)
            f.write("<br>")
        f.write("</form></div><hr>\n")

        f.write('<div class="container">\n')

        for i, name in enumerate(image_names):
            x = results[i, 0] * 150  # Scale the x-coordinate
            y = results[i, 1] * 150  # Scale the y-coordinate

            # TODO logic encapsulation
            # Get pictures' tags
            key = name.split("/")[-1]
            tags = image_tags[key]
            tag_str = f"{tags[0]},{tags[1]}"

            if clusters is None:
                f.write(
                    f'<img src="../{name}" style="left:{
                        x}px;top:{y}px;" data-tags="{tag_str}">\n'
                )
            else:
                cluster = clusters[i]
                f.write(f'<img src="../{name}" class="cluster-{cluster}" style="left:{
                        x}px;top:{y}px;" data-tags="{tag_str}">\n')
            f.write(f'<div class="tooltip" style="left:{
                    x+60}px;top:{y}px;">{tag_str}</div>\n')

        f.write("</div>")

        # Script for filtering images
        f.write("""
        <script>
          let checkSubset = (parentArray, subsetArray) => {
            return subsetArray.every((el) => {
                return parentArray.includes(el)
            })
        }
        function filterImages() {

            let checkedTags = Array.from(document.querySelectorAll('input[name="tag"]:checked'))
                .map(cb => cb.value);

            let flag;

            let images = Array.from(document.getElementsByTagName('img'));
            let tags;
            let show;
            images.forEach((img) => {
                tags = img.getAttribute('data-tags').split(",");
                flag = checkedTags.length < tags.length;

                if (flag) {
                    console.log(flag)
                    let count = 0;
                    checkedTags.forEach((reg) => {

                        tags.forEach((tag) => {
                            if (tag == reg) {
                                count++;
                            }
                        })

                    })
                    show = (count == checkedTags.length);
                }
                else {

                    show = checkSubset(checkedTags, tags);

                }
                img.style.display = show || checkedTags.length === 0 ? 'block' : 'none';
            });

        }         
        </script>
        """)
        f.write("</body></html>")
