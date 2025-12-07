import os
import shutil
from pathlib import Path

# ===================== Configuration Parameters =====================
# Root directory containing Vendor_A/B1/B2 source data
SOURCE_ROOT = "/data/yanglida/projects/CARE-Liver/LiQA_training_data"
# Target directory for image data
IMAGE_TARGET_ROOT = "./train/images"
# Target directory for mask data
MASK_TARGET_ROOT = "./train/masks"

# File name mapping rules (original filename → new filename)
FILE_MAPPING = {
    "T1.nii.gz": "raw data1.nii.gz",
    "T2.nii.gz": "raw data2.nii.gz",
    "GED1.nii.gz": "raw data4.nii.gz",
    "GED2.nii.gz": "raw data5.nii.gz",
    "GED3.nii.gz": "raw data6.nii.gz",
    "GED4.nii.gz": "raw data7.nii.gz"
}
# DWI file prefix (matches all DWI_<b-value>.nii.gz files)
DWI_PREFIX = "DWI_"
DWI_NEW_NAME = "raw data3.nii.gz"
# GED4 mask renaming rule
MASK_GED4_NEW_NAME = "Mask7.nii.gz"

# List of vendors to process
VENDORS = ["Vendor_A", "Vendor_B1", "Vendor_B2"]

# ===================== Utility Functions =====================
def safe_copy(src: str, dst: str) -> bool:
    """
    Safely copy file with error handling
    :param src: Source file path
    :param dst: Target file path
    :return: True if copy successful, False otherwise
    """
    try:
        # copy2 preserves file metadata (recommended over shutil.copy)
        shutil.copy2(src, dst)
        print(f"Success: Copied {src} to {dst}")
        return True
    except FileNotFoundError:
        print(f"Error: Source file not found - {src}")
        return False
    except PermissionError:
        print(f"Error: Permission denied when copying {src} to {dst}")
        return False
    except Exception as e:
        print(f"Error: Failed to copy {src} to {dst} - {str(e)}")
        return False

# ===================== Main Processing Logic =====================
def process_subject(subject_name: str, subject_source_path: Path):
    """
    Process individual subject folder (e.g., 0888-A-S1)
    :param subject_name: Name of subject folder (e.g., 0888-A-S1)
    :param subject_source_path: Source path of subject folder
    """
    print(f"\nProcessing subject: {subject_name}")
    
    # 1. Check if GED4 mask exists
    mask_ged4_src = subject_source_path / "mask_GED4.nii.gz"
    has_mask_ged4 = mask_ged4_src.exists()
    
    # 2. Process GED4 mask (if exists)
    ged4_processed = False  # Flag to track if GED4 was copied to mask directory
    if has_mask_ged4:
        # Create mask target folder
        mask_subject_path = Path(MASK_TARGET_ROOT) / subject_name
        mask_subject_path.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename GED4 mask to Mask7.nii.gz
        mask_dst = mask_subject_path / MASK_GED4_NEW_NAME
        safe_copy(str(mask_ged4_src), str(mask_dst))
        
        # Copy GED4 image to mask directory (rename to raw data7.nii.gz)
        ged4_src = subject_source_path / "GED4.nii.gz"
        if ged4_src.exists():
            ged4_dst = mask_subject_path / FILE_MAPPING["GED4.nii.gz"]
            if safe_copy(str(ged4_src), str(ged4_dst)):
                ged4_processed = True
        else:
            print(f"Warning: GED4 mask exists but GED4 image is missing - {subject_name}")
    
    # 3. Process all image files (copy to image directory, except processed GED4)
    # Create image target folder
    image_subject_path = Path(IMAGE_TARGET_ROOT) / subject_name
    image_subject_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate through all nii.gz files in subject folder
    for file in subject_source_path.glob("*.nii.gz"):
        file_name = file.name
        
        # Skip all mask files (mask_GED4 handled separately)
        if file_name.startswith("mask_"):
            continue
        
        # Process DWI files (match DWI_ prefix)
        if file_name.startswith(DWI_PREFIX):
            dst_path = image_subject_path / DWI_NEW_NAME
            safe_copy(str(file), str(dst_path))
        
        # Process files in mapping table (T1/T2/GED1-4)
        elif file_name in FILE_MAPPING:
            # Skip GED4 if already copied to mask directory
            if file_name == "GED4.nii.gz" and ged4_processed:
                continue
            # Copy to image directory
            dst_name = FILE_MAPPING[file_name]
            dst_path = image_subject_path / dst_name
            safe_copy(str(file), str(dst_path))
        
        # Skip unknown files with warning
        else:
            print(f"Warning: Unknown file type, skipping - {file}")

def main():
    """Main function: iterate through all Vendor and subject folders"""
    # Check if source root directory exists
    if not Path(SOURCE_ROOT).exists():
        print(f"Error: Source root directory does not exist - {SOURCE_ROOT}")
        return
    
    # Iterate through each vendor
    for vendor in VENDORS:
        vendor_path = Path(SOURCE_ROOT) / vendor
        if not vendor_path.exists():
            print(f"\nWarning: Vendor directory does not exist, skipping - {vendor_path}")
            continue
        
        print(f"\n=====================================")
        print(f"Processing Vendor: {vendor}")
        print(f"=====================================")
        
        # Iterate through all subject folders in vendor directory
        for item in vendor_path.iterdir():
            # Only process directories (skip files)
            if item.is_dir():
                process_subject(item.name, item)
    
    print("\nAll data processing completed!")

if __name__ == "__main__":
    main()