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
import gc
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
        """Train for one epoch with memory optimization"""
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
            
            # Store predictions in CPU memory to save GPU memory
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Clear GPU cache periodically
            if batch_idx % 50 == 0:
                self.clear_memory()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Delete variables to free memory
            del data, targets, outputs, loss
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # Clear memory after epoch
        self.clear_memory()
        
        return epoch_loss, epoch_metrics

    # Also modify validate_epoch similarly
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch with memory optimization"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Store in CPU memory
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
                
                # Clear GPU cache periodically
                if batch_idx % 50 == 0:
                    self.clear_memory()
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                # Delete variables
                del data, targets, outputs, loss
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # Clear memory after validation
        self.clear_memory()
        
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
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
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
                                        patience=patience//2)
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

# New Memory-Optimized Trainer Class
class MemoryOptimizedTrainer(Trainer):
    """Memory-optimized trainer with fixed IoU calculation - Simple Version"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enable mixed precision training (fixed deprecated warning)
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def clear_memory(self):
        """Aggressive memory clearing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
    
    def calculate_segmentation_metrics(self, predictions, targets):
        """Simple and robust segmentation metrics calculation"""
        # Get predictions
        preds = torch.argmax(predictions, dim=1)
        
        # Convert to numpy for easier calculation
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Initialize metrics
        batch_ious = []
        batch_dices = []
        batch_accuracies = []
        
        for i in range(preds_np.shape[0]):
            pred_i = preds_np[i]
            target_i = targets_np[i]
            
            # Focus on tumor class (class 1)
            pred_tumor = (pred_i == 1).astype(np.float32)
            target_tumor = (target_i == 1).astype(np.float32)
            
            # Calculate intersection and union
            intersection = np.sum(pred_tumor * target_tumor)
            union = np.sum(pred_tumor) + np.sum(target_tumor) - intersection
            
            # Calculate IoU
            if union > 0:
                iou = intersection / union
            else:
                # Perfect prediction if both are empty
                iou = 1.0 if (np.sum(target_tumor) == 0 and np.sum(pred_tumor) == 0) else 0.0
            
            # Calculate Dice
            if (np.sum(pred_tumor) + np.sum(target_tumor)) > 0:
                dice = 2.0 * intersection / (np.sum(pred_tumor) + np.sum(target_tumor))
            else:
                # Perfect prediction if both are empty
                dice = 1.0 if (np.sum(target_tumor) == 0 and np.sum(pred_tumor) == 0) else 0.0
            
            # Calculate pixel accuracy
            pixel_accuracy = np.mean(pred_i == target_i)
            
            batch_ious.append(iou)
            batch_dices.append(dice)
            batch_accuracies.append(pixel_accuracy)
        
        return {
            'iou': np.mean(batch_ious),
            'dice': np.mean(batch_dices),
            'pixel_accuracy': np.mean(batch_accuracies)
        }
    
    def train_epoch(self, optimizer: optim.Optimizer, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Memory-optimized training epoch with simple metrics"""
        # Handle different autocast versions
        try:
            from torch.amp import autocast
            autocast_context = autocast('cuda')
        except ImportError:
            from torch.cuda.amp import autocast
            autocast_context = autocast()
        
        self.model.train()
        running_loss = 0.0
        
        # Collect metrics
        all_metrics = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast_context:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = self.calculate_segmentation_metrics(outputs, targets)
                all_metrics.append(batch_metrics)
            
            running_loss += loss.item()
            
            # Clear memory periodically
            if batch_idx % 20 == 0:
                self.clear_memory()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{batch_metrics["dice"]:.4f}',
                'IoU': f'{batch_metrics["iou"]:.4f}'
            })
            
            # Clean up
            del data, targets, outputs, loss
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        
        # Average all metrics
        epoch_metrics = {
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'dice': np.mean([m['dice'] for m in all_metrics]),
            'pixel_accuracy': np.mean([m['pixel_accuracy'] for m in all_metrics])
        }
        
        self.clear_memory()
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Memory-optimized validation epoch with simple metrics"""
        # Handle different autocast versions
        try:
            from torch.amp import autocast
            autocast_context = autocast('cuda')
        except ImportError:
            from torch.cuda.amp import autocast
            autocast_context = autocast()
        
        self.model.eval()
        running_loss = 0.0
        
        # Collect metrics
        all_metrics = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(pbar):
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast_context:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                batch_metrics = self.calculate_segmentation_metrics(outputs, targets)
                all_metrics.append(batch_metrics)
                
                running_loss += loss.item()
                
                # Clear memory periodically
                if batch_idx % 20 == 0:
                    self.clear_memory()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{batch_metrics["dice"]:.4f}',
                    'IoU': f'{batch_metrics["iou"]:.4f}'
                })
                
                # Clean up
                del data, targets, outputs, loss
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        
        # Average all metrics
        epoch_metrics = {
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'dice': np.mean([m['dice'] for m in all_metrics]),
            'pixel_accuracy': np.mean([m['pixel_accuracy'] for m in all_metrics])
        }
        
        self.clear_memory()
        return epoch_loss, epoch_metrics
    
def analyze_dataset_distribution(dataloader, num_batches=10):
    """Analyze the class distribution in your dataset"""
    print("=== DATASET ANALYSIS ===")
    
    total_pixels = 0
    tumor_pixels = 0
    samples_with_tumor = 0
    total_samples = 0
    
    for i, (_, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_tumor_pixels = (targets == 1).sum().item()
        batch_total_pixels = targets.numel()
        batch_samples_with_tumor = ((targets == 1).sum(dim=(1,2)) > 0).sum().item()
        
        total_pixels += batch_total_pixels
        tumor_pixels += batch_tumor_pixels
        samples_with_tumor += batch_samples_with_tumor
        total_samples += targets.shape[0]
        
        print(f"Batch {i}: {batch_tumor_pixels}/{batch_total_pixels} tumor pixels ({100*batch_tumor_pixels/batch_total_pixels:.1f}%)")
        print(f"         {batch_samples_with_tumor}/{targets.shape[0]} samples have tumors")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Tumor pixels: {tumor_pixels}/{total_pixels} ({100*tumor_pixels/total_pixels:.2f}%)")
    print(f"Samples with tumors: {samples_with_tumor}/{total_samples} ({100*samples_with_tumor/total_samples:.1f}%)")
    print("=" * 40)#

def main():
    """Example usage of the training module"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        print(f"Available GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(image_size=(256, 256))
    
    # Load and prepare data (replace with actual data path)
    try:
        # This would load real data
        data_path = r"/home/heet_bhalani/Downloads/archive/lgg-mri-segmentation/kaggle_3m"
        image_paths, mask_paths, labels = preprocessor.load_kaggle_lgg_dataset(data_path)
        
        # Create data splits
        data_splits = preprocessor.create_data_splits(image_paths, mask_paths, labels)
        
        # Create dataloaders
        seg_dataloaders = preprocessor.create_dataloaders(
            data_splits, batch_size=2, mode='segmentation', num_workers=1
        )

        cls_dataloaders = preprocessor.create_dataloaders(
            data_splits, batch_size=16, mode='classification'
        )
        
        # Train classification model
        print("\n=== Training Classification Model ===")
        """cls_model = get_model('resnet', num_classes=2, pretrained=True)
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
        )"""
        
        print("Analyzing dataset distribution...")
        analyze_dataset_distribution(seg_dataloaders['train'], num_batches=10)
        analyze_dataset_distribution(seg_dataloaders['val'], num_batches=5)

        # Train segmentation model
        print("\n=== Training Segmentation Model ===")
        seg_model = get_model('resunet', n_classes=2)

        param_count = sum(p.numel() for p in seg_model.parameters())
        print(f"Model parameters: {param_count:,}")

        seg_trainer = MemoryOptimizedTrainer(  # Use the new trainer class below
                                                model=seg_model,
                                                train_loader=seg_dataloaders['train'],
                                                val_loader=seg_dataloaders['val'],
                                                task='segmentation',
                                                device=device,
                                                log_dir='runs/segmentation',
                                                save_dir='checkpoints/segmentation'
                                            )
        
        best_seg_model = seg_trainer.train(
            num_epochs=30,  # Reduced epochs
            learning_rate=0.001,
            patience=8
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Please ensure the dataset path is correct and dependencies are installed.")


if __name__ == "__main__":
    main()