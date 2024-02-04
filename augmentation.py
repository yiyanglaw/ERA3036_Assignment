import os
import cv2
import numpy as np

#data augmentation
def augment_images(input_folder, output_folder, shape_type, num_augmentations=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image_path = os.path.join(input_folder, filename)
            original_image = cv2.imread(image_path)

            # Rotate the image
            for angle in range(0, 360, 360 // num_augmentations):
                rotated_image = rotate_image(original_image, angle)
                save_augmented_image(output_folder, filename, 'rotate', angle, rotated_image, shape_type)

            # Flip the image horizontally
            flipped_image = flip_image(original_image)
            save_augmented_image(output_folder, filename, 'flip', 0, flipped_image, shape_type)

            # Shift the image
            shifted_image = shift_image(original_image, shift_range=8)  
            save_augmented_image(output_folder, filename, 'shift', 0, shifted_image, shape_type)

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


#fill background with white color
def fill_background_with_white(image):
    white_background = np.ones_like(image) * 255
    filled_image = cv2.bitwise_or(image, white_background)
    return filled_image

def save_augmented_image(output_folder, original_filename, augmentation_type, angle, augmented_image, shape_type):
    base_filename, file_extension = os.path.splitext(original_filename)
    new_filename = f"{base_filename}_{shape_type}_{augmentation_type}_{angle}{file_extension}"
    output_path = os.path.join(output_folder, new_filename)
    cv2.imwrite(output_path, augmented_image)

def is_blank(image):
    return np.all(image == 255)


#some blank and meaningless images generated after augmentation, delete them

def delete_blank_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if is_blank(image):
                print(f"Deleting blank image: {filename}")
                os.remove(image_path)


#change the file path according to user needs
if __name__ == "__main__":
    circle_input_folder = r'three_shapes_filled\three_shapes_filled\train\circle'
    square_input_folder = r'three_shapes_filled\three_shapes_filled\train\square'
    triangle_input_folder = r'three_shapes_filled\three_shapes_filled\train\triangle'

    circle_output_folder = r'three_shapes_filled\three_shapes_filled\train\circle'
    square_output_folder = r'three_shapes_filled\three_shapes_filled\train\square'
    triangle_output_folder = r'three_shapes_filled\three_shapes_filled\train\triangle'

    # Augment circle images
    augment_images(circle_input_folder, circle_output_folder, 'circle', num_augmentations=5)

    # Augment square images
    augment_images(square_input_folder, square_output_folder, 'square', num_augmentations=5)

    # Augment triangle images
    augment_images(triangle_input_folder, triangle_output_folder, 'triangle', num_augmentations=5)

    # Delete blank images for each shape
    delete_blank_images(circle_output_folder)
    delete_blank_images(square_output_folder)
    delete_blank_images(triangle_output_folder)
