import os
import shutil
from pathlib import Path

# ===================== Configuration Parameters =====================
# Root directory containing Vendor_A/B1/B2 validation data
SOURCE_ROOT = "/data/yanglida/projects/CARE-Liver/LiQA_val/Data"
# Target directory for validation data (all files go here)
VAL_TARGET_ROOT = "./val"

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
        # copy2 preserves file metadata (recommended over basic copy)
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
def process_val_subject(subject_name: str, subject_source_path: Path):
    """
    Process individual validation subject folder (e.g., 0888-A-S1)
    :param subject_name: Name of subject folder (e.g., 0888-A-S1)
    :param subject_source_path: Source path of subject folder
    """
    print(f"\nProcessing validation subject: {subject_name}")
    
    # Create target folder for this subject (auto-create parent dirs if needed)
    val_subject_path = Path(VAL_TARGET_ROOT) / subject_name
    val_subject_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate through all nii.gz files in subject folder
    for file in subject_source_path.glob("*.nii.gz"):
        file_name = file.name
        
        # Skip mask files (defensive check - no masks in validation data)
        if file_name.startswith("mask_"):
            print(f"Warning: Found mask file in validation data (unexpected) - {file}, skipping")
            continue
        
        # Process DWI files (match DWI_ prefix)
        if file_name.startswith(DWI_PREFIX):
            dst_path = val_subject_path / DWI_NEW_NAME
            safe_copy(str(file), str(dst_path))
        
        # Process mapped files (T1/T2/GED1-4)
        elif file_name in FILE_MAPPING:
            dst_name = FILE_MAPPING[file_name]
            dst_path = val_subject_path / dst_name
            safe_copy(str(file), str(dst_path))
        
        # Skip unknown file types with warning
        else:
            print(f"Warning: Unknown file type in validation data, skipping - {file}")

def main():
    """Main function: iterate through all Vendor and validation subject folders"""
    # Validate source root directory exists
    if not Path(SOURCE_ROOT).exists():
        print(f"Error: Validation source root directory does not exist - {SOURCE_ROOT}")
        return
    
    # Iterate through each vendor directory
    for vendor in VENDORS:
        vendor_path = Path(SOURCE_ROOT) / vendor
        if not vendor_path.exists():
            print(f"\nWarning: Validation vendor directory does not exist, skipping - {vendor_path}")
            continue
        
        print(f"\n=====================================")
        print(f"Processing Validation Vendor: {vendor}")
        print(f"=====================================")
        
        # Process all subject directories in current vendor folder
        for item in vendor_path.iterdir():
            if item.is_dir():
                process_val_subject(item.name, item)
    
    print("\nAll validation data processing completed!")

if __name__ == "__main__":
    main()