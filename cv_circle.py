import os
import cv2
import numpy as np

def augment_circle_images(input_folder, output_folder, num_augmentations=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  
            image_path = os.path.join(input_folder, filename)
            original_image = cv2.imread(image_path)

            # Rotate the image
            for angle in range(0, 360, 360 // num_augmentations):
                rotated_image = rotate_image(original_image, angle)
                save_augmented_image(output_folder, filename, 'rotate', angle, rotated_image)

            # Flip the image horizontally
            flipped_image = flip_image(original_image)
            save_augmented_image(output_folder, filename, 'flip', 0, flipped_image)

            # Shift the image
            shifted_image = shift_image(original_image, shift_range=8)  # Adjust shift_range as needed
            save_augmented_image(output_folder, filename, 'shift', 0, shifted_image)

def rotate_image(image, angle):
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderValue=(255, 255, 255))
    return rotated_image

def flip_image(image):
    flipped_image = cv2.flip(image, 1)
    return fill_background_with_white(flipped_image)

def shift_image(image, shift_range=10):
    rows, cols, _ = image.shape
    x_shift = np.random.randint(-shift_range, shift_range + 1)
    y_shift = np.random.randint(-shift_range, shift_range + 1)
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (cols, rows), borderValue=(255, 255, 255))
    return shifted_image

def fill_background_with_white(image):
    # Create a white background with the same size as the input image
    white_background = np.ones_like(image) * 255
    # Combine the input image with the white background
    filled_image = cv2.bitwise_or(image, white_background)
    return filled_image

def save_augmented_image(output_folder, original_filename, augmentation_type, angle, augmented_image):
    base_filename, file_extension = os.path.splitext(original_filename)
    new_filename = f"{base_filename}_{augmentation_type}_{angle}{file_extension}"
    output_path = os.path.join(output_folder, new_filename)
    cv2.imwrite(output_path, augmented_image)

if __name__ == "__main__":
    input_folder = r'C:\Users\Law Yi Yang\Downloads\shape\New folder (6)\three_shapes_filled\three_shapes_filled\train\circle'
    output_folder = r'C:\Users\Law Yi Yang\Downloads\shape\New folder (6)\three_shapes_filled\three_shapes_filled\train\circle'  
    augment_circle_images(input_folder, output_folder, num_augmentations=5)
