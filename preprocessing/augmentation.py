import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
from pathlib import Path
from PIL import Image
import numpy as np
from copy import deepcopy

# Directories
input_dir = 'DATA/default/train'

# Define augmentors with different augmentation settings
augmentors = [
    ImageDataGenerator(
        # rescale=1.0/255,
        rotation_range=45,         # Rotate up to 15 degrees
        # width_shift_range=0.2,     # Shift horizontally up to 20%
        # height_shift_range=0.2,    # Shift vertically up to 20%
        shear_range=0.2,           # She ar intensity
        zoom_range=0.2,            # Zoom in/out up to 20%
        brightness_range=[0.8, 1.2], # Adjust brightness
        vertical_flip=True       # Flip vertically
    ),
    ImageDataGenerator(
        # rescale=1.0/255,
        rotation_range=15,         # Rotate up to 15 degrees
        width_shift_range=0.2,     # Shift horizontally up to 20%
        height_shift_range=0.2,    # Shift vertically up to 20%
        shear_range=0.2,           # Shear intensity
        zoom_range=0.2,            # Zoom in/out up to 20%
        brightness_range=[0.8, 1.2], # Adjust brightness
        vertical_flip=True       # Flip vertically
    ),
    ImageDataGenerator(
        # rescale=1.0/255,
        rotation_range=180,         # Rotate up to 15 degrees
        # width_shift_range=0.2,     # Shift horizontally up to 20%
        # height_shift_range=0.2,    # Shift vertically up to 20%
        shear_range=0.1,           # Shear intensity
        zoom_range=0.1,            # Zoom in/out up to 20%
        brightness_range=[0.8, 1.2], # Adjust brightness
        horizontal_flip=True       # Flip horizontally
    )
]

# Parameters
img_size = (96, 96)
N = 81 # vertical slicer
slice1_height = N  # Height of the first slice
slice2_height = img_size[0] - N # Height of the second slice
total_height, width = slice1_height + slice2_height, 96  # Combined image dimensions
batch_size = 32      # Batch size for processing
augmentation_factor = 5  # Number of augmentations per original image

# Apply each augmentor
for index, augmentor in enumerate(augmentors):
    output_dir = f'DATA/augmented_{index}/train'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Processing augmentor {index}...")

    # Process each class folder
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        # Output class directory
        class_output_dir = os.path.join(output_dir, class_dir)
        Path(class_output_dir).mkdir(parents=True, exist_ok=True)

        # Load images from the class directory
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Load image and resize
            img = Image.open(img_path)  # Convert to greyscale
            img = img.resize((width, total_height))
            img_array = np.array(img)

            # Split the image into two slices
            slice1 = img_array[:slice1_height, :]
            slice2 = img_array[slice1_height:, :]

            # Augment the first slice
            slice1 = slice1.reshape((1, slice1_height, width, 3))  # Add batch and channel dimensions
            augment_count = 0
            for batch in augmentor.flow(slice1, batch_size=1, seed=np.random.randint(1)):
                # Reconstruct the combined image
                augmented_slice1 = batch[0].reshape((slice1_height, width, 3))  # Remove batch and channel dimensions

                combined_image = np.zeros(img_array.shape)
                combined_image[:slice1_height, :] = augmented_slice1
                combined_image[slice1_height:, :] = slice2
                # combined_image = np.vstack((slice2, augmented_slice1))  # Combine slices

                # Save the combined image
                combined_image = Image.fromarray(combined_image.astype(np.uint8))  # Convert back to image
                combined_image.save(os.path.join(class_output_dir, f"{os.path.splitext(img_name)[0]}_aug{augment_count}.png"))

                augment_count += 1
                if augment_count >= augmentation_factor:  # Stop after generating enough augmentations
                    break

    print(f"Augmentation for augmentor {index} complete. Saved to: {output_dir}")

print("All augmentations complete.")