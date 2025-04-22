"""
Training module for brain tumor detection models
Supports both classification and segmentation tasks with TensorBoard logging
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import get_model, BrainTumorResNet, ResUNet, UNet
from data_preprocessing import DataPreprocessor


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.softmax(predictions, dim=1)
        
        # Convert to one-hot encoding
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for i in range(predictions.shape[1]):
            pred_flat = predictions[:, i].contiguous().view(-1)
            target_flat = targets_one_hot[:, i].contiguous().view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
            dice_scores.append(dice)
        
        return 1 - torch.mean(torch.stack(dice_scores))


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(predictions, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss for segmentation (CrossEntropy + Dice)"""
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class Trainer:
    """Main trainer class for brain tumor detection"""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 task: str = 'classification', device: str = 'cuda',
                 log_dir: str = 'runs', save_dir: str = 'checkpoints'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            task: 'classification' or 'segmentation'
            device: Device to run on
            log_dir: Directory for TensorBoard logs
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.device = device
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        self.save_dir = save_dir
        
        # Setup loss function
        self._setup_loss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        self.best_val_metric = 0.0
        self.best_model_path = None
        
    def _setup_loss(self):
        """Setup loss function based on task"""
        if self.task == 'classification':
            # Use Focal Loss to handle class imbalance
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:  # segmentation
            # Use combined loss for better segmentation
            self.criterion = CombinedLoss(ce_weight=0.4, dice_weight=0.6)
    
    def _calculate_metrics(self, predictions: torch.Tensor, 
                          targets: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics based on task"""
        metrics = {}
        
        if self.task == 'classification':
            preds = torch.argmax(predictions, dim=1)
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            metrics['accuracy'] = accuracy_score(targets_np, preds_np)
            metrics['precision'] = precision_score(targets_np, preds_np, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(targets_np, preds_np, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(targets_np, preds_np, average='weighted', zero_division=0)
            
        else:  # segmentation
            preds = torch.argmax(predictions, dim=1)
            
            # Calculate IoU (Intersection over Union)
            intersection = torch.logical_and(targets, preds)
            union = torch.logical_or(targets, preds)
            iou = torch.sum(intersection, dim=(1, 2)) / torch.sum(union, dim=(1, 2))
            metrics['iou'] = torch.mean(iou[~torch.isnan(iou)]).item()
            
            # Calculate pixel accuracy
            correct = (preds == targets).float()
            metrics['pixel_accuracy'] = torch.mean(correct).item()
            
            # Calculate Dice coefficient
            dice_score = self._calculate_dice(preds, targets)
            metrics['dice'] = dice_score
        
        return metrics
    
    def _calculate_dice(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Dice coefficient"""
        smooth = 1e-6
        predictions = predictions.float()
        targets = targets.float()
        
        intersection = (predictions * targets).sum(dim=(1, 2))
        dice = (2. * intersection + smooth) / (predictions.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + smooth)
        return torch.mean(dice).item()
    
    def train_epoch(self, optimizer: optim.Optimizer, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(targets.detach())
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for data, targets in pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                all_predictions.append(outputs)
                all_targets.append(targets)
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return epoch_loss, epoch_metrics
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       val_metric: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metric': val_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"New best model saved at epoch {epoch} with metric: {val_metric:.4f}")
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: float,
                   train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to TensorBoard"""
        # Log losses
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Log metrics
        for metric_name, metric_value in train_metrics.items():
            self.writer.add_scalar(f'Metrics/Train_{metric_name}', metric_value, epoch)
        
        for metric_name, metric_value in val_metrics.items():
            self.writer.add_scalar(f'Metrics/Val_{metric_name}', metric_value, epoch)
        
        # Log learning rate
        for param_group in self.optimizer.param_groups:
            self.writer.add_scalar('Learning_Rate', param_group['lr'], epoch)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot main metric
        if self.task == 'classification':
            main_metric = 'accuracy'
        else:
            main_metric = 'dice'
        
        train_main_metric = [m[main_metric] for m in self.train_metrics]
        val_main_metric = [m[main_metric] for m in self.val_metrics]
        
        axes[0, 1].plot(train_main_metric, label=f'Train {main_metric.capitalize()}', color='blue')
        axes[0, 1].plot(val_main_metric, label=f'Val {main_metric.capitalize()}', color='red')
        axes[0, 1].set_title(f'Training and Validation {main_metric.capitalize()}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel(main_metric.capitalize())
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot additional metrics
        if self.task == 'classification':
            train_f1 = [m['f1'] for m in self.train_metrics]
            val_f1 = [m['f1'] for m in self.val_metrics]
            
            axes[1, 0].plot(train_f1, label='Train F1', color='blue')
            axes[1, 0].plot(val_f1, label='Val F1', color='red')
            axes[1, 0].set_title('Training and Validation F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            train_iou = [m['iou'] for m in self.train_metrics]
            val_iou = [m['iou'] for m in self.val_metrics]
            
            axes[1, 0].plot(train_iou, label='Train IoU', color='blue')
            axes[1, 0].plot(val_iou, label='Val IoU', color='red')
            axes[1, 0].set_title('Training and Validation IoU')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self, num_epochs: int, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, patience: int = 10,
              scheduler_type: str = 'reduce_lr'):
        """Main training loop"""
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=learning_rate, weight_decay=weight_decay)
        
        # Setup scheduler
        if scheduler_type == 'reduce_lr':
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                        patience=patience//2, verbose=True)
        else:
            scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Task: {self.task}")
        print(f"Model: {self.model.__class__.__name__}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(self.optimizer, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Log to TensorBoard
            self.log_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Get main validation metric
            if self.task == 'classification':
                val_metric = val_metrics['f1']
            else:
                val_metric = val_metrics['dice']
            
            # Check for best model
            is_best = val_metric > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, self.optimizer, val_metric, is_best)
            
            # Update scheduler
            if scheduler_type == 'reduce_lr':
                scheduler.step(val_metric)
            else:
                scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if self.task == 'classification':
                print(f"  Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            else:
                print(f"  Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
                print(f"  Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Plot training history
        self.plot_training_history()
        
        return self.best_model_path


def main():
    """Example usage of the training module"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(image_size=(256, 256))
    
    # Load and prepare data (replace with actual data path)
    try:
        # This would load real data
        data_path = r"C:\Users\heet\Downloads\kaggle_3m"
        image_paths, mask_paths, labels = preprocessor.load_kaggle_lgg_dataset(data_path)
        
        # Create data splits
        data_splits = preprocessor.create_data_splits(image_paths, mask_paths, labels)
        
        # Create dataloaders
        seg_dataloaders = preprocessor.create_dataloaders(
            data_splits, batch_size=8, mode='segmentation'
        )
        
        cls_dataloaders = preprocessor.create_dataloaders(
            data_splits, batch_size=16, mode='classification'
        )
        
        # Train classification model
        print("\n=== Training Classification Model ===")
        cls_model = get_model('resnet', num_classes=2, pretrained=True)
        cls_trainer = Trainer(
            model=cls_model,
            train_loader=cls_dataloaders['train'],
            val_loader=cls_dataloaders['val'],
            task='classification',
            device=device,
            log_dir='runs/classification',
            save_dir='checkpoints/classification'
        )
        
        best_cls_model = cls_trainer.train(
            num_epochs=50,
            learning_rate=0.001,
            patience=10
        )
        
        # Train segmentation model
        print("\n=== Training Segmentation Model ===")
        seg_model = get_model('resunet', n_classes=2)
        seg_trainer = Trainer(
            model=seg_model,
            train_loader=seg_dataloaders['train'],
            val_loader=seg_dataloaders['val'],
            task='segmentation',
            device=device,
            log_dir='runs/segmentation',
            save_dir='checkpoints/segmentation'
        )
        
        best_seg_model = seg_trainer.train(
            num_epochs=100,
            learning_rate=0.001,
            patience=15
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        print("Please ensure the dataset path is correct and dependencies are installed.")


if __name__ == "__main__":
    main()