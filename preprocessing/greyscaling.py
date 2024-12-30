import tensorflow as tf
import os
from PIL import Image

# Paths
source_dir = 'DATA/default'
target_dir = 'DATA/greyscale'

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# Function to process images
def convert_to_grayscale(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding target directories
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        os.makedirs(target_path, exist_ok=True)
        print(f"Processing: {root} -> {target_path}")
        for file in files:
            if file.endswith('.png'):
                # Load image
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                
                # Save grayscale image
                target_img_path = os.path.join(target_path, file)
                img.save(target_img_path)
                print(f"Converted: {img_path} -> {target_img_path}")

# Convert dataset
convert_to_grayscale(source_dir, target_dir)
print("Dataset conversion to grayscale completed!")