from concurrent.futures import ThreadPoolExecutor
import os
from os.path import splitext
from PIL import Image

def JPG2jpg(folder_path):
    """changes the extension from 'JPG' to 'jpg'"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.JPG'):
            old_path = os.path.join(folder_path, filename)
            new_filename = filename[:-4] + '.jpg'  # Replace extension
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed {filename} -> {new_filename}')


def convert_jpg2png(file, folder_path):
    """changes a image to a png image"""
    filename, extension = splitext(file)
    if not os.path.isdir(f'{folder_path}/edited'):
        os.mkdir(f'{folder_path}/edited')
    if extension == '.jpg':
        print(file)
        im = Image.open(f'{folder_path}/{file}')
        im.save(f'{folder_path}/edited/{filename}.png')


# def convert(file, folder_path):
#     """converts """
#     filename, extension = splitext(file)
#     if extension.lower() == '.jpg':
#         im = Image.open(os.path.join(folder_path, file))
#         im.save(os.path.join(folder_path, f'{filename}.png'))

def jpg2png(folder_path):
    """Changes jpg images in a folder to png images using threads"""
    files = os.listdir(folder_path)
    with ThreadPoolExecutor() as executor:
        for file in files:
            executor.submit(convert_jpg2png, file, folder_path)

def downscale_pngs(input_folder, output_folder, scale_factor=0.5):
    """Downscale all PNG images in a folder by a scale factor."""
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith('.png'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            with Image.open(input_path) as img:
                # Calculate new size
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                # Resize image
                img_resized = img.resize(new_size, Image.LANCZOS)
                # Save resized image
                img_resized.save(output_path)

            print(f"Downscaled {file} to {new_size}")


def downscale_jpgs(input_folder, output_folder, scale_factor=0.5):
    """Downscale all PNG images in a folder by a scale factor."""
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith('.jpg'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            with Image.open(input_path) as img:
                # Calculate new size
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                # Resize image
                img_resized = img.resize(new_size, Image.LANCZOS)
                # Save resized image
                img_resized.save(output_path)

            print(f"Downscaled {file} to {new_size}")

if __name__ == "__main__":
    folder_path = "/workspaces/BEP-3D-scanner/datasets/peer_constant_f/images_HQ"
    output_folder = "/workspaces/BEP-3D-scanner/datasets/peer_constant_f/images2"
    #JPG2jpg("/workspaces/BEP-3D-scanner/datasets/pear/Peertje/images")
    #jpg2png(folder_path)
    downscale_jpgs(folder_path, output_folder, 0.4)
    
