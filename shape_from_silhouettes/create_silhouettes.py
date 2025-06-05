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

    
if __name__ == '__main__':
    # Store path of the image in the variable input_path
    input_path =  "datasets/ignore_mouse/IMG_20250522_134923.jpg"

    # Store path of the output image in the variable output_path
    output_path = 'datasets/ignore_mouse/sil2.jpg'

    dir = "datasets/peer_constant_f/images/"
    paths = os.listdir(dir)
    for path in paths:
        output_path = "datasets/peer_constant_f/silhouettes/sil_"+path
        sil = remove_background_rembg(dir+path, output_path)
    #Saving the image in the given path
    #output.save(output_path)