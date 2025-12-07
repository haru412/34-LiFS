import os
from Nii_utils import NiiDataRead, NiiDataWrite
import numpy as np
from skimage import transform

# Common parameter settings
target_size = (32, 160, 192)
values_clip = (-55, 145)
images_data_dir = "./data/train/images/"
masks_data_dir = "./data/train/masks/"
save_dir = "./data/preprocessed_train/"


def process_images():
    """Process data in the images directory, create output folders, and save preprocessing results"""
    for ID in os.listdir(images_data_dir):
        # Create output folder (only created during image processing, reused directly in mask processing stage)
        output_id_dir = os.path.join(save_dir, ID)
        os.makedirs(output_id_dir, exist_ok=True)
        
        for mode in ['1', '2', '3', '4', '5', '6', '7']:
            file_path = os.path.join(images_data_dir, ID, f'raw data{mode}.nii.gz')

            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist. Create an all-zero placeholder image")
                img = np.zeros(target_size, dtype=np.float32)
                spacing = (1.0, 1.0, 1.0)
                origin = (0.0, 0.0, 0.0)
                direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            else:
                img, spacing, origin, direction = NiiDataRead(file_path)
                # Process 4D data
                if len(img.shape) == 4:
                    img = img[:, :, :, 0]
                # Normalization processing
                img = np.clip(img, values_clip[0], values_clip[1])
                img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

            # Calculate new spacing
            spacing_z = (spacing[0] * img.shape[0]) / target_size[0]
            spacing_x = (spacing[1] * img.shape[1]) / target_size[1]
            spacing_y = (spacing[2] * img.shape[2]) / target_size[2]
            new_spacing = np.array([spacing_z, spacing_x, spacing_y])

            # Only resize existing files (no resize needed for placeholders)
            if os.path.exists(file_path):
                img = transform.resize(
                    img, target_size, order=0, mode='constant',
                    clip=False, preserve_range=True, anti_aliasing=False
                )
                # Check if dimensions match
                if img.shape != tuple(target_size):
                    print(f"Dimension mismatch after resizing for {ID}, mode {mode}. Skipping...")
                    continue

            # Save preprocessing results
            save_path = os.path.join(output_id_dir, f'{mode}_img.nii.gz')
            NiiDataWrite(save_path, img, new_spacing, origin, direction)


def process_masks():
    """Process data in the masks directory and reuse the already created output folders"""
    for ID in os.listdir(masks_data_dir):
        output_id_dir = os.path.join(save_dir, ID)
        # Skip non-existent output folders (ensure only IDs preprocessed via images are processed)
        if not os.path.exists(output_id_dir):
            print(f"Warning: Output directory {output_id_dir} not found. Skipping mask processing for {ID}")
            continue

        # Only process mask data for mode=7
        mode = '7'
        try:
            # Read original image and mask
            img_path = os.path.join(masks_data_dir, ID, f'raw data{mode}.nii.gz')
            mask_path = os.path.join(masks_data_dir, ID, f'Mask{mode}.nii.gz')
            img, spacing, origin, direction = NiiDataRead(img_path)
            mask_tumor, _, _, _ = NiiDataRead(mask_path)
        except Exception as e:
            print(f"Error reading data for {ID}, mode {mode}: {e}. Skipping...")
            continue

        # Process 4D data
        if len(img.shape) == 4:
            img = img[:, :, :, 0]

        # Crop image and mask based on mask (only retain regions containing tumors)
        z_, x_, y_ = mask_tumor.nonzero()
        if len(z_) == 0 or len(x_) == 0 or len(y_) == 0:
            print(f"Empty mask for {ID}, mode {mode}. Skipping...")
            continue
        
        z1, z2 = z_.min(), z_.max()
        x1, x2 = x_.min(), x_.max()
        y1, y2 = y_.min(), y_.max()

        img = img[z1:z2+1, x1:x2+1, y1:y2+1]
        mask_tumor = mask_tumor[z1:z2+1, x1:x2+1, y1:y2+1]

        # Check if empty after cropping
        if img.size == 0 or mask_tumor.size == 0:
            print(f"Empty image or mask after cropping for {ID}, mode {mode}. Skipping...")
            continue

        # Normalization processing
        img = np.clip(img, values_clip[0], values_clip[1])
        img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

        # Calculate new spacing
        spacing_z = (spacing[0] * img.shape[0]) / target_size[0]
        spacing_x = (spacing[1] * img.shape[1]) / target_size[1]
        spacing_y = (spacing[2] * img.shape[2]) / target_size[2]
        new_spacing = np.array([spacing_z, spacing_x, spacing_y])

        # Resize dimensions
        img = transform.resize(
            img, target_size, order=0, mode='constant',
            clip=False, preserve_range=True, anti_aliasing=False
        )
        mask_tumor = transform.resize(
            mask_tumor, target_size, order=0, mode='constant',
            clip=False, preserve_range=True, anti_aliasing=False
        )

        # Check if dimensions match
        if img.shape != tuple(target_size) or mask_tumor.shape != tuple(target_size):
            print(f"Dimension mismatch after resizing for {ID}, mode {mode}. Skipping...")
            continue

        # Generate mask-integrated image (set non-tumor regions to -1)
        img_tumor = np.copy(img)
        img_tumor[mask_tumor == 0] = -1

        # Save results (will overwrite mode=7 results generated during image processing)
        save_path = os.path.join(output_id_dir, f'{mode}_img.nii.gz')
        NiiDataWrite(save_path, img_tumor, new_spacing, origin, direction)


if __name__ == "__main__":
    # First process images to create output folders
    print("Starting image processing...")
    process_images()
    # Then process masks, reusing the already created folders
    print("Starting mask processing...")
    process_masks()
    print("Preprocessing completed.")