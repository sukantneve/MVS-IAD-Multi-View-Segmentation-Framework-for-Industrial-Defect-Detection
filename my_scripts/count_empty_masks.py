import cv2
import numpy as np
import glob
import os

def count_empty_and_non_empty_masks(mask_dir):
    empty_count = 0
    non_empty_count = 0
    total_files = 0

    # Gather all image files matching the extensions
    mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))

    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read image {path}")
            continue

        total_files += 1
        if np.count_nonzero(mask) == 0:
            empty_count += 1
        else:
            non_empty_count += 1

    print(f"Total masks: {total_files}")
    print(f"Empty masks: {empty_count}")
    print(f"Non-empty masks: {non_empty_count}")

# Example usage:
mask_folder = '/home/exouser/Downloads/pcb/NG/ZW/*'
count_empty_and_non_empty_masks(mask_folder)
