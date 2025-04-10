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