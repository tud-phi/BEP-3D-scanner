# Importing Required Modules
import os
from rembg import remove
from PIL import Image
import numpy as np

def remove_background_rembg(input_path, output_path=None):
    # Processing the image
    input = Image.open(input_path)

    # Removing the background from the given Image
    output = remove(input)

    output = np.array(output)

    mask = (output != 0).astype(np.uint8)
    mask = (mask[:,:,0])

    if output_path is not None:
        mask_rbg = mask * 255
        im = Image.fromarray(mask_rbg)
        im.save(output_path)

    return mask

def remove_background_white(input_path, output_path=None):
    # Processing the image
    input = Image.open(input_path)

    # Removing the background from the given Image
    mask = (np.sum(input, axis=-1) < 300)
    #mask = np.array([input]) < 90
    #mask = mask.min(axis=-1)[0].astype(np.uint8) * 255

    if output_path is not None:
        im = Image.fromarray(mask)
        im.save(output_path)

    return mask

def remove_background_blue(input_path, output_path=None, threshold=70):
    # Processing the image
    input = Image.open(input_path)
    img = input.convert("RGB")
    data = np.array(img)

    # Removing the background from the given Image
    #mask = ((data[:, :, 0] / (data[:, :, 2])) < 0.9) & ((data[:, :, 1] / (data[:, :, 2])) < 0.9 )
    # Treat blue-dominant pixels as background
    mask = (data[:, :, 2] > threshold) & (data[:, :, 2] > data[:, :, 0] + 5) & (data[:, :, 2] > data[:, :, 1] + 5)
    mask = mask == False
    if output_path is not None:
        im = Image.fromarray(mask)
        im.save(output_path)

    return mask

    
if __name__ == '__main__':
    # Store path of the image in the variable input_path
    input_path =  "datasets/ignore_mouse/IMG_20250522_134923.jpg"

    # Store path of the output image in the variable output_path
    output_path = 'datasets/ignore_mouse/sil2.jpg'

    dir = "datasets/machine4/images_cropped/"
    paths = os.listdir(dir)
    for path in paths:
        output_path = "datasets/machine4/silhouettes/sil_"+path
        sil = remove_background_blue(dir+path, output_path, threshold=50)
    #Saving the image in the given path
    #output.save(output_path)