import os

# Base directory containing the files
base_directory = "images/"


tag_grid_dict = {"1": "barbie", "2": "cigarette"}
tag_row_dict = {
    "0": "boy",
    "1": "girl",
    "2": "male",
    "3": "female",
    "4": "man",
    "5": "woman",
}


def generate_image_tag():
    infos = []
    for file_name in os.listdir(base_directory):
        file_path = os.path.join(base_directory, file_name)

        if os.path.isfile(file_path):
            file_stem = os.path.splitext(file_name)[0]
            info = str(file_stem).split("_")
            info[0] = tag_grid_dict[info[0]]
            info[1] = tag_row_dict[info[1]]
            info.pop()
            infos.append({file_name: info})
    # print(infos)
    return infos
