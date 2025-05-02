"""
Test and evaluation module for brain tumor detection models
Includes comprehensive evaluation metrics and visualization
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, 
                           average_precision_score)
from sklearn.preprocessing import label_binarize
import cv2
from typing import Dict, Tuple
from tqdm import tqdm
import json
import warnings
from models import get_model
from data_preprocessing import DataPreprocessor
warnings.filterwarnings("ignore")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda', 
                 task: str = 'classification'):
        """
        Initialize evaluator
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            task: 'classification' or 'segmentation'
        """
        self.model = model.to(device)
        self.device = device
        self.task = task
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")
        
        if 'val_metric' in checkpoint:
            print(f"Checkpoint validation metric: {checkpoint['val_metric']:.4f}")
    
    def predict_batch(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for a data loader
        
        Returns:
            predictions: Raw model outputs
            probabilities: Softmax probabilities
            targets: Ground truth labels
        """
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Generating predictions"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                
                if self.task == 'segmentation':
                    # For segmentation, flatten spatial dimensions
                    outputs = outputs.permute(0, 2, 3, 1).contiguous()
                    outputs = outputs.view(-1, outputs.size(-1))
                    probabilities = probabilities.permute(0, 2, 3, 1).contiguous()
                    probabilities = probabilities.view(-1, probabilities.size(-1))
                    targets = targets.view(-1)
                
                all_predictions.append(outputs.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        probabilities = np.concatenate(all_probabilities, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, probabilities, targets
    
    def evaluate_classification(self, data_loader, save_dir: str = 'evaluation_results'):
        """Comprehensive classification evaluation"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        predictions, probabilities, targets = self.predict_batch(data_loader)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Basic metrics
        report = classification_report(targets, predicted_classes, output_dict=True)
        print("Classification Report:")
        print(classification_report(targets, predicted_classes))
        
        # Confusion Matrix
        cm = confusion_matrix(targets, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Tumor', 'Tumor'], 
                   yticklabels=['No Tumor', 'Tumor'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        if len(np.unique(targets)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
            avg_precision = average_precision_score(targets, probabilities[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save metrics to JSON
        metrics = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall']
        }
        
        if len(np.unique(targets)) == 2:
            metrics['roc_auc'] = float(roc_auc)
            metrics['average_precision'] = float(avg_precision)
        
        # Convert numpy types before saving
        metrics_serializable = convert_numpy_types(metrics)
        
        with open(os.path.join(save_dir, 'classification_metrics.json'), 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        return metrics
    
    def calculate_segmentation_metrics(self, predictions: np.ndarray, 
                                     targets: np.ndarray) -> Dict[str, float]:
        """Calculate detailed segmentation metrics"""
        pred_classes = np.argmax(predictions, axis=1)
        
        # Overall metrics
        pixel_accuracy = np.mean(pred_classes == targets)
        
        # Per-class metrics
        metrics = {'pixel_accuracy': pixel_accuracy}
        
        for class_id in np.unique(targets):
            if class_id == 0:  # Background
                class_name = 'background'
            else:  # Tumor
                class_name = 'tumor'
            
            # Binary masks for current class
            pred_mask = (pred_classes == class_id).astype(np.float32)
            true_mask = (targets == class_id).astype(np.float32)
            
            # Intersection and Union
            intersection = np.sum(pred_mask * true_mask)
            union = np.sum(pred_mask) + np.sum(true_mask) - intersection
            
            # IoU
            iou = intersection / (union + 1e-8)
            metrics[f'{class_name}_iou'] = iou
            
            # Dice coefficient
            dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-8)
            metrics[f'{class_name}_dice'] = dice
            
            # Sensitivity and Specificity for tumor class
            if class_id == 1:  # Tumor class
                tp = intersection
                fn = np.sum(true_mask) - intersection
                fp = np.sum(pred_mask) - intersection
                tn = len(targets) - tp - fn - fp
                
                sensitivity = tp / (tp + fn + 1e-8)
                specificity = tn / (tn + fp + 1e-8)
                
                metrics['sensitivity'] = sensitivity
                metrics['specificity'] = specificity
        
        return metrics
    
    def evaluate_segmentation(self, data_loader, save_dir: str = 'evaluation_results'):
        """Comprehensive segmentation evaluation"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        predictions, probabilities, targets = self.predict_batch(data_loader)
        
        # Calculate metrics
        metrics = self.calculate_segmentation_metrics(predictions, targets)
        
        print("Segmentation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Visualize sample predictions
        self.visualize_segmentation_predictions(data_loader, save_dir, num_samples=8)
        
        # Convert numpy types before saving to JSON
        metrics_serializable = convert_numpy_types(metrics)
        
        # Save metrics
        with open(os.path.join(save_dir, 'segmentation_metrics.json'), 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        return metrics
    
    def visualize_segmentation_predictions(self, data_loader, save_dir: str, 
                                         num_samples: int = 8):
        """Visualize segmentation predictions"""
        self.model.eval()
        data_iter = iter(data_loader)
        
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
        
        samples_shown = 0
        
        with torch.no_grad():
            while samples_shown < num_samples:
                try:
                    data, targets = next(data_iter)
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(data)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    for i in range(min(data.size(0), num_samples - samples_shown)):
                        idx = samples_shown
                        
                        # Original image
                        img = data[i].cpu().permute(1, 2, 0).numpy()
                        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img = np.clip(img, 0, 1)
                        
                        axes[0, idx].imshow(img)
                        axes[0, idx].set_title('Original Image')
                        axes[0, idx].axis('off')
                        
                        # Ground truth mask
                        true_mask = targets[i].cpu().numpy()
                        axes[1, idx].imshow(true_mask, cmap='gray')
                        axes[1, idx].set_title('Ground Truth')
                        axes[1, idx].axis('off')
                        
                        # Predicted mask
                        pred_mask = predictions[i].cpu().numpy()
                        axes[2, idx].imshow(pred_mask, cmap='gray')
                        axes[2, idx].set_title('Prediction')
                        axes[2, idx].axis('off')
                        
                        samples_shown += 1
                        if samples_shown >= num_samples:
                            break
                    
                    if samples_shown >= num_samples:
                        break
                        
                except StopIteration:
                    break
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'segmentation_predictions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_classification_predictions(self, data_loader, save_dir: str,
                                           num_samples: int = 8):
        """Visualize classification predictions"""
        self.model.eval()
        data_iter = iter(data_loader)
        
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 8))
        axes = axes.flatten()
        
        samples_shown = 0
        
        with torch.no_grad():
            while samples_shown < num_samples:
                try:
                    data, targets = next(data_iter)
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(data)
                    probabilities = F.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    for i in range(min(data.size(0), num_samples - samples_shown)):
                        idx = samples_shown
                        
                        # Original image
                        img = data[i].cpu().permute(1, 2, 0).numpy()
                        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img = np.clip(img, 0, 1)
                        
                        axes[idx].imshow(img)
                        
                        true_label = "Tumor" if targets[i].item() == 1 else "No Tumor"
                        pred_label = "Tumor" if predictions[i].item() == 1 else "No Tumor"
                        confidence = probabilities[i, predictions[i]].item()
                        
                        color = 'green' if predictions[i] == targets[i] else 'red'
                        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})', 
                                          color=color)
                        axes[idx].axis('off')
                        
                        samples_shown += 1
                        if samples_shown >= num_samples:
                            break
                    
                    if samples_shown >= num_samples:
                        break
                        
                except StopIteration:
                    break
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'classification_predictions.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, data_loader, save_dir: str = 'evaluation_results'):
        """Main evaluation function"""
        print(f"Evaluating {self.task} model...")
        
        if self.task == 'classification':
            metrics = self.evaluate_classification(data_loader, save_dir)
            self.visualize_classification_predictions(data_loader, save_dir)
        else:  # segmentation
            metrics = self.evaluate_segmentation(data_loader, save_dir)
        
        return metrics


def main():
    """Example usage of the evaluation module"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data preprocessor and load test data
    preprocessor = DataPreprocessor(image_size=(256, 256))
    
    try:
        # Load test data (replace with actual paths)
        data_path = r"/home/heet_bhalani/Downloads/archive/lgg-mri-segmentation/kaggle_3m"
        image_paths, mask_paths, labels = preprocessor.load_kaggle_lgg_dataset(data_path)
        
        data_splits = preprocessor.create_data_splits(image_paths, mask_paths, labels)
        
        # Debug: Check what keys are available
        print("Available data split keys:", list(data_splits.keys()))
        print("Test split size:", len(data_splits.get('test', {}).get('images', [])))
        
        # Create test dataloaders - provide all required keys for create_dataloaders method
        test_data_splits = {
            'train': data_splits['test'],  # Use test data for all splits (dummy data)
            'val': data_splits['test'],
            'test': data_splits['test']
        }
        
        cls_dataloaders = preprocessor.create_dataloaders(
            test_data_splits, batch_size=16, mode='classification'
        )
        cls_test_loader = cls_dataloaders['test']
        
        seg_dataloaders = preprocessor.create_dataloaders(
            test_data_splits, batch_size=8, mode='segmentation'
        )
        seg_test_loader = seg_dataloaders['test']
        
        # Evaluate individual models
        print("\n=== Evaluating Classification Model ===")
        cls_model = get_model('resnet', num_classes=2, pretrained=True)
        cls_evaluator = ModelEvaluator(cls_model, device, 'classification')
        
        # Load best checkpoint if available
        cls_checkpoint_path = r"/home/heet_bhalani/Desktop/Projects/Brain-Tumor-Detection-Using-Deep-Learning/training_history/classification/best_model.pth"
        if os.path.exists(cls_checkpoint_path):
            cls_evaluator.load_checkpoint(cls_checkpoint_path)
        
        cls_metrics = cls_evaluator.evaluate_model(cls_test_loader, 'results/classification')
        
        print("\n=== Evaluating Segmentation Model ===")
        seg_model = get_model('resunet', n_classes=2)
        seg_evaluator = ModelEvaluator(seg_model, device, 'segmentation')
        
        # Load best checkpoint if available
        seg_checkpoint_path = r"/home/heet_bhalani/Desktop/Projects/Brain-Tumor-Detection-Using-Deep-Learning/training_history/segmentation/best_model.pth"
        if os.path.exists(seg_checkpoint_path):
            seg_evaluator.load_checkpoint(seg_checkpoint_path)
        
        seg_metrics = seg_evaluator.evaluate_model(seg_test_loader, 'results/segmentation')
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"Error in evaluation: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        print("Please ensure models are trained and checkpoints are available.")


if __name__ == "__main__":
    main()