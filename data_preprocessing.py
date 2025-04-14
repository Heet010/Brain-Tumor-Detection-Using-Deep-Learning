"""
Data preprocessing module for brain tumor detection
Handles data loading, augmentation, and preparation for training
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor images"""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str] = None, 
                 labels: List[int] = None, transform=None, mode='segmentation'):
        """
        Args:
            image_paths: List of paths to brain MRI images
            mask_paths: List of paths to segmentation masks (for segmentation task)
            labels: List of labels (for classification task)
            transform: Data augmentation transforms
            mode: 'segmentation' or 'classification'
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'segmentation':
            # Load mask for segmentation
            if self.mask_paths and idx < len(self.mask_paths):
                mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            return image, mask.long()
        
        else:  # classification mode
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            label = self.labels[idx] if self.labels else 0
            return image, label
        
class DataPreprocessor:
    """Main class for data preprocessing operations"""
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup data augmentation transforms"""
        # Training transforms with augmentation
        self.train_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, 
                             rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                     contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation/Test transforms without augmentation
        self.val_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Classification-only transforms
        self.classification_train_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, 
                             rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                     contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def load_kaggle_lgg_dataset(self, data_path: str) -> Tuple[List[str], List[str], List[int]]:
        """
        Load the LGG MRI Segmentation dataset from Kaggle
        
        Args:
            data_path: Path to the dataset directory
            
        Returns:
            Tuple of (image_paths, mask_paths, labels)
        """
        image_paths = []
        mask_paths = []
        labels = []
        
        # Navigate through the dataset structure
        for patient_folder in os.listdir(data_path):
            patient_path = os.path.join(data_path, patient_folder)
            if os.path.isdir(patient_path):
                for file in os.listdir(patient_path):
                    if file.endswith('.tif'):
                        if '_mask' in file:
                            continue  # Skip mask files in this loop
                        
                        image_path = os.path.join(patient_path, file)
                        mask_file = file.replace('.tif', '_mask.tif')
                        mask_path = os.path.join(patient_path, mask_file)
                        
                        image_paths.append(image_path)
                        
                        # Check if mask exists
                        if os.path.exists(mask_path):
                            mask_paths.append(mask_path)
                            # Check if mask has tumor (non-zero pixels)
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            has_tumor = 1 if np.any(mask > 0) else 0
                            labels.append(has_tumor)
                        else:
                            mask_paths.append(None)
                            labels.append(0)  # No tumor if no mask
        
        return image_paths, mask_paths, labels
    
    def create_data_splits(self, image_paths: List[str], mask_paths: List[str], 
                          labels: List[int], test_size: float = 0.2, 
                          val_size: float = 0.1) -> dict:
        """
        Create train/validation/test splits
        
        Args:
            image_paths: List of image paths
            mask_paths: List of mask paths
            labels: List of labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Dictionary containing train/val/test splits
        """
        # First split: train+val vs test
        train_val_images, test_images, train_val_masks, test_masks, \
        train_val_labels, test_labels = train_test_split(
            image_paths, mask_paths, labels, 
            test_size=test_size, stratify=labels, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_images, val_images, train_masks, val_masks, \
        train_labels, val_labels = train_test_split(
            train_val_images, train_val_masks, train_val_labels,
            test_size=val_ratio, stratify=train_val_labels, random_state=42
        )
        
        return {
            'train': {
                'images': train_images,
                'masks': train_masks,
                'labels': train_labels
            },
            'val': {
                'images': val_images,
                'masks': val_masks,
                'labels': val_labels
            },
            'test': {
                'images': test_images,
                'masks': test_masks,
                'labels': test_labels
            }
        }
    
    def create_dataloaders(self, data_splits: dict, batch_size: int = 16, 
                          num_workers: int = 4, mode: str = 'segmentation') -> dict:
        """
        Create PyTorch DataLoaders
        
        Args:
            data_splits: Dictionary with train/val/test splits
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            mode: 'segmentation' or 'classification'
            
        Returns:
            Dictionary containing DataLoaders
        """
        dataloaders = {}
        
        # Training dataloader
        train_dataset = BrainTumorDataset(
            image_paths=data_splits['train']['images'],
            mask_paths=data_splits['train']['masks'],
            labels=data_splits['train']['labels'],
            transform=self.train_transform if mode == 'segmentation' 
                     else self.classification_train_transform,
            mode=mode
        )
        dataloaders['train'] = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        
        # Validation dataloader
        val_dataset = BrainTumorDataset(
            image_paths=data_splits['val']['images'],
            mask_paths=data_splits['val']['masks'],
            labels=data_splits['val']['labels'],
            transform=self.val_transform,
            mode=mode
        )
        dataloaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        # Test dataloader
        test_dataset = BrainTumorDataset(
            image_paths=data_splits['test']['images'],
            mask_paths=data_splits['test']['masks'],
            labels=data_splits['test']['labels'],
            transform=self.val_transform,
            mode=mode
        )
        dataloaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        return dataloaders
    
    def visualize_samples(self, dataloader: DataLoader, num_samples: int = 4, 
                         mode: str = 'segmentation'):
        """Visualize sample data"""
        data_iter = iter(dataloader)
        images, targets = next(data_iter)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
        
        for i in range(num_samples):
            # Denormalize image for visualization
            img = images[i].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            if mode == 'segmentation':
                mask = targets[i].numpy()
                axes[1, i].imshow(mask, cmap='gray')
                axes[1, i].set_title(f'Mask {i+1}')
            else:
                axes[1, i].text(0.5, 0.5, f'Label: {targets[i].item()}', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'Label {i+1}')
            
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_statistics(self, labels: List[int]) -> dict:
        """Get basic statistics about the dataset"""
        total_samples = len(labels)
        tumor_samples = sum(labels)
        no_tumor_samples = total_samples - tumor_samples
        
        stats = {
            'total_samples': total_samples,
            'tumor_samples': tumor_samples,
            'no_tumor_samples': no_tumor_samples,
            'tumor_percentage': (tumor_samples / total_samples) * 100,
            'class_distribution': {
                'no_tumor': no_tumor_samples,
                'tumor': tumor_samples
            }
        }
        
        return stats
    
def main():
    """Example usage of the data preprocessing module"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(image_size=(256, 256))
    
    # Load dataset (adjust path as needed)
    data_path = r"/home/heet_bhalani/Downloads/archive/lgg-mri-segmentation/kaggle_3m"
    
    try:
        image_paths, mask_paths, labels = preprocessor.load_kaggle_lgg_dataset(data_path)
        print(f"Loaded {len(image_paths)} images")
        
        # Get dataset statistics
        stats = preprocessor.get_dataset_statistics(labels)
        print(f"Dataset Statistics: {stats}")
        
        # Create data splits
        data_splits = preprocessor.create_data_splits(image_paths, mask_paths, labels)
        
        # Create dataloaders for segmentation
        seg_dataloaders = preprocessor.create_dataloaders(
            data_splits, batch_size=8, mode='segmentation'
        )
        
        # Create dataloaders for classification
        cls_dataloaders = preprocessor.create_dataloaders(
            data_splits, batch_size=16, mode='classification'
        )
        
        print("Data preprocessing completed successfully!")
        
        # Visualize samples
        print("Visualizing segmentation samples...")
        preprocessor.visualize_samples(seg_dataloaders['train'], num_samples=4, mode='segmentation')
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        print("Please ensure the dataset path is correct and the dataset is downloaded.")


if __name__ == "__main__":
    main()