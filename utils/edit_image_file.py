from numpy import random
from PIL import Image, ExifTags

def remove_every_other_line_from_line_6(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Remove every other line starting from line 6 (index 5)
    #filtered_lines = [
    #    line for i, line in enumerate(lines)
    #    if i < 5 or (i - 5) % 2 == 0
    #]

    filtered_lines = []
    for i, line in enumerate(lines):
        if i < 5:
            filtered_lines.append(line)
        elif i % 2 == 0:
            filtered_lines.append(line)
        else:
            filtered_lines.append("\n")
    print(filtered_lines)
    with open(output_path, 'w') as f:
        f.writelines(filtered_lines)

def reorder_image_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        split_line = line.split(" ")
        split_line[-1] = split_line[-1][:-1]
        split_line.pop(0)
        if split_line != []:
            new_lines.append(split_line)
    new_lines.sort(key=lambda x : x[-1])
    for i, line in enumerate(new_lines):
        line.insert(0, i+1)
    
    new_new_lines = [' '.join(str(x) for x in line) + "\n\n" for line in new_lines]

    print(new_new_lines)
    with open(output_path, 'w') as f:
        f.writelines(new_new_lines)

def change_filename(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        split_line = line.split(" ")
        split_line[-1] = split_line[-1][7:-1]
        print(split_line)
        if split_line != ['']:
            new_lines.append(split_line)
    
    new_new_lines = [' '.join(str(x) for x in line) + "\n\n" for line in new_lines]

    print(new_new_lines)
    with open(output_path, 'w') as f:
        f.writelines(new_new_lines)

def read_metadata(path):
    img = Image.open(path)
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    return exif

def wiggle_values(input_path, output_path, wiggle_value):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        split_line = line.split(" ")
        if split_line[0] == '#':
            continue
        split_line = [float(value) * (1 + wiggle_value * (random.rand()-0.5)) if i in [1,2,3,4,5,6,7] else value for i, value in enumerate(split_line)]
        print(split_line)
        if split_line != ['']:
            new_lines.append(split_line)
    
    new_new_lines = [' '.join(str(x) for x in line) for line in new_lines]

    print(new_new_lines)
    with open(output_path, 'w') as f:
        f.writelines(new_new_lines)

wiggle_values("/workspaces/BEP-3D-scanner/datasets/peer_constant_f/known_parameters/images_gt.txt", "/workspaces/BEP-3D-scanner/datasets/peer_constant_f/known_parameters/images.txt", 0.1)
#change_filename("/workspaces/BEP-3D-scanner/datasets/ccvpeer_downscaledPY/sparse/images.txt", "/workspaces/BEP-3D-scanner/datasets/ccvpeer_downscaledPY/sparse/images4.txt")
#reorder_image_file("/workspaces/BEP-3D-scanner/datasets/ccvpeer_downscaledPY/sparse/images2.txt", "/workspaces/BEP-3D-scanner/datasets/ccvpeer_downscaledPY/sparse/images3.txt")
# Example usage:
#remove_every_other_line_from_line_6("/workspaces/BEP-3D-scanner/datasets/peer_constant_f/sparsetxt4/images.txt", "/workspaces/BEP-3D-scanner/datasets/peer_constant_f/sparsetxt4/images2.txt")

#print(read_metadata("/workspaces/BEP-3D-scanner/datasets/ccvpeer/images/PXL_20250429_154045599.jpg"))