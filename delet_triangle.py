import os
import cv2
import numpy as np

def is_blank(image):
    return np.all(image == 255)

def delete_blank_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Update the file extensions as needed
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if is_blank(image):
                print(f"Deleting blank image: {filename}")
                os.remove(image_path)

if __name__ == "__main__":
    folder_path = r'C:\Users\Law Yi Yang\Downloads\shape\x\New folder (6)\three_shapes_filled\three_shapes_filled\train\triangle'
    delete_blank_images(folder_path)
