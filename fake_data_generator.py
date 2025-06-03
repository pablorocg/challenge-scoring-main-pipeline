#!/usr/bin/env python3
"""
FOMO Challenge Fake Data Generator

Generates fake NIfTI data for all three FOMO challenge tasks:
- Task 1: Infarct classification (80 cases)
- Task 2: Meningioma segmentation (40 cases) 
- Task 3: Brain age prediction (200 cases)

Each volume is 128x128x128 with random intensities.
"""

import os
import numpy as np
import nibabel as nib
import random
from pathlib import Path

def create_fake_nifti(shape=(128, 128, 128), dtype=np.float32, min_val=0, max_val=1000):
    """Create a fake NIfTI image with random data."""
    # Generate random data
    data = np.random.uniform(min_val, max_val, shape).astype(dtype)
    
    # Create NIfTI image with identity affine matrix
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)
    
    return nifti_img

def create_fake_segmentation(shape=(128, 128, 128)):
    """Create a fake binary segmentation mask."""
    # Create mostly zeros with some random 1s (simulating tumor regions)
    data = np.zeros(shape, dtype=np.uint8)
    
    # Add some random connected regions as "tumors"
    num_regions = random.randint(0, 3)  # 0-3 tumor regions
    
    for _ in range(num_regions):
        # Random center point
        center = [random.randint(20, 108) for _ in range(3)]
        # Random size
        size = random.randint(5, 15)
        
        # Create a rough spherical region
        for i in range(max(0, center[0]-size), min(128, center[0]+size)):
            for j in range(max(0, center[1]-size), min(128, center[1]+size)):
                for k in range(max(0, center[2]-size), min(128, center[2]+size)):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                    if dist < size:
                        data[i, j, k] = 1
    
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)

def generate_subject_numbers(n_subjects, max_subject_num=None):
    """Generate random subject numbers (non-consecutive to simulate QC failures)."""
    if max_subject_num is None:
        max_subject_num = n_subjects * 2  # Allow for gaps
    
    # Generate more numbers than needed, then sample
    possible_nums = list(range(1, max_subject_num + 1))
    return sorted(random.sample(possible_nums, n_subjects))

def create_task1_data(base_dir, n_cases=80):
    """Generate fake data for Task 1: Infarct classification."""
    print(f"Generating Task 1 data ({n_cases} cases)...")
    
    task_dir = Path(base_dir) / "fomo-task1-val"
    preprocessed_dir = task_dir / "preprocessed"
    labels_dir = task_dir / "labels"
    
    subject_nums = generate_subject_numbers(n_cases, 120)
    
    for sub_num in subject_nums:
        sub_dir = preprocessed_dir / f"sub_{sub_num}" / "ses_1"
        label_dir = labels_dir / f"sub_{sub_num}"
        
        # Create directories
        sub_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate required files
        modalities = ["flair", "adc", "dwi_b1000"]
        
        # Add either t2s or swi (randomly choose)
        extra_modality = random.choice(["t2s", "swi"])
        modalities.append(extra_modality)
        
        for modality in modalities:
            if modality in ["adc"]:
                # ADC values are typically lower
                nifti_img = create_fake_nifti(min_val=0, max_val=2000)
            elif modality in ["dwi_b1000"]:
                # DWI b1000 values
                nifti_img = create_fake_nifti(min_val=0, max_val=3000)
            else:
                # Standard intensity range for FLAIR, T2*, SWI
                nifti_img = create_fake_nifti(min_val=0, max_val=4000)
            
            filename = sub_dir / f"{modality}.nii.gz"
            nib.save(nifti_img, filename)
        
        # Generate binary label (0 or 1)
        label = random.randint(0, 1)
        label_file = label_dir / "label.txt"
        with open(label_file, 'w') as f:
            f.write(str(label))
    
    print(f"Task 1 complete: {len(subject_nums)} cases generated")
    return subject_nums

def create_task2_data(base_dir, n_cases=40):
    """Generate fake data for Task 2: Meningioma segmentation."""
    print(f"Generating Task 2 data ({n_cases} cases)...")
    
    task_dir = Path(base_dir) / "fomo-task2-val"
    preprocessed_dir = task_dir / "preprocessed"
    labels_dir = task_dir / "labels"
    
    subject_nums = generate_subject_numbers(n_cases, 60)
    
    for sub_num in subject_nums:
        sub_dir = preprocessed_dir / f"sub_{sub_num}" / "ses_1"
        label_dir = labels_dir / f"sub_{sub_num}"
        
        # Create directories
        sub_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate required files
        modalities = ["flair", "dwi_b1000"]
        
        # Add either t2s or swi (randomly choose)
        extra_modality = random.choice(["t2s", "swi"])
        modalities.append(extra_modality)
        
        for modality in modalities:
            if modality in ["dwi_b1000"]:
                nifti_img = create_fake_nifti(min_val=0, max_val=3000)
            else:
                nifti_img = create_fake_nifti(min_val=0, max_val=4000)
            
            filename = sub_dir / f"{modality}.nii.gz"
            nib.save(nifti_img, filename)
        
        # Generate binary segmentation mask
        seg_img = create_fake_segmentation()
        seg_file = label_dir / "seg.nii.gz"
        nib.save(seg_img, seg_file)
    
    print(f"Task 2 complete: {len(subject_nums)} cases generated")
    return subject_nums

def create_task3_data(base_dir, n_cases=200):
    """Generate fake data for Task 3: Brain age prediction."""
    print(f"Generating Task 3 data ({n_cases} cases)...")
    
    task_dir = Path(base_dir) / "fomo-task3-val"
    preprocessed_dir = task_dir / "preprocessed"
    labels_dir = task_dir / "labels"
    
    subject_nums = generate_subject_numbers(n_cases, 300)
    
    for sub_num in subject_nums:
        sub_dir = preprocessed_dir / f"sub_{sub_num}" / "ses_1"
        label_dir = labels_dir / f"sub_{sub_num}"
        
        # Create directories
        sub_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate T1 and T2 images
        modalities = ["t1", "t2"]
        
        for modality in modalities:
            if modality == "t1":
                # T1 intensity characteristics
                nifti_img = create_fake_nifti(min_val=0, max_val=4000)
            else:  # t2
                # T2 intensity characteristics
                nifti_img = create_fake_nifti(min_val=0, max_val=3500)
            
            filename = sub_dir / f"{modality}.nii.gz"
            nib.save(nifti_img, filename)
        
        # Generate age label (18-90 years, as float)
        age = random.uniform(18.0, 90.0)
        label_file = label_dir / "label.txt"
        with open(label_file, 'w') as f:
            f.write(f"{age:.2f}")
    
    print(f"Task 3 complete: {len(subject_nums)} cases generated")
    return subject_nums

def main():
    """Main function to generate all fake data."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Base directory - change this to your desired location
    base_dir = "data/fomo25"
    
    print("FOMO Challenge Fake Data Generator")
    print("==================================")
    print(f"Output directory: {base_dir}")
    print("Volume dimensions: 128 x 128 x 128")
    print()
    
    # Create base directory
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data for all tasks
    task1_subjects = create_task1_data(base_dir, n_cases=10)
    print()
    task2_subjects = create_task2_data(base_dir, n_cases=10)
    print()
    task3_subjects = create_task3_data(base_dir, n_cases=10)
    
    print()
    print("Data generation complete!")
    print(f"Task 1: {len(task1_subjects)} cases")
    print(f"Task 2: {len(task2_subjects)} cases") 
    print(f"Task 3: {len(task3_subjects)} cases")
    print()
    print("Directory structure created:")
    print(f"  {base_dir}/fomo-task1-val/")
    print(f"  {base_dir}/fomo-task2-val/")
    print(f"  {base_dir}/fomo-task3-val/")

if __name__ == "__main__":
    main()