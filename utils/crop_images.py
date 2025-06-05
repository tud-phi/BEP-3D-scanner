import os
from PIL import Image
import cv2
import numpy as np

def crop_images(input_folder, output_folder, crop=[130, 400, 180, 300],  intensity=0.0):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            image = Image.open(input_path)

            width, height = image.size
            left = crop[3]
            top = crop[0]
            right = width - crop[1]
            bottom = height - crop[2]

            if right <= left or bottom <= top:
                print(f"Skipping {filename}: image too small to crop.")
                continue

            cropped = image.crop((left, top, right, bottom))

            img_float = np.array(cropped).astype(np.float32) / 255.0

            # Convert to HSV to target bright regions
            hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]

            # Reduce highlights: darken pixels with high brightness
            v = np.where(v > 0.8, v * (1 - intensity), v)
            hsv[:, :, 2] = np.clip(v, 0, 1)

            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            result = (result * 255).astype(np.uint8)
            result = Image.fromarray(result)


            output_path = os.path.join(output_folder, filename)
            result.save(output_path)
            print(f"Cropped and saved: {output_path}")

# Example usage:
# crop_images("input_folder_path", "output_folder_path")
    
if __name__ == '__main__':
    crop_images("datasets/ignore_machine5/images", "datasets/ignore_machine5/images_cropped", crop=[130, 350, 180, 350])