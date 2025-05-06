"""
Main script for brain tumor detection project
Handles the complete pipeline from data preprocessing to model evaluation
"""

import os
import sys
import argparse
import torch
import json
from datetime import datetime
import logging

# Import project modules
from data_preprocessing import DataPreprocessor
from models import get_model
from train import Trainer
from test import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_tumor_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'checkpoints',
        'checkpoints/classification',
        'checkpoints/segmentation',
        'runs',
        'runs/classification', 
        'runs/segmentation',
        'results',
        'results/classification',
        'results/segmentation'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Created necessary directories")


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file"""
    default_config = {
        "data": {
            "dataset_path": "data/lgg-mri-segmentation",
            "image_size": [256, 256],
            "batch_size_classification": 16,
            "batch_size_segmentation": 8,
            "num_workers": 4,
            "test_size": 0.2,
            "val_size": 0.1
        },
        "training": {
            "num_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "patience": 15,
            "scheduler_type": "reduce_lr"
        },
        "models": {
            "classification": {
                "model_name": "resnet",
                "num_classes": 2,
                "pretrained": True
            },
            "segmentation": {
                "model_name": "resunet",
                "n_classes": 2,
                "bilinear": True
            }
        },
        "evaluation": {
            "save_predictions": True,
            "visualize_results": True,
            "num_samples_visualize": 8
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = default_config
        # Save default config
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Created default configuration file: config.json")
    
    return config


def preprocess_data(config: dict):
    """Handle data preprocessing"""
    logger.info("Starting data preprocessing...")
    
    # Initialize preprocessor
    image_size = tuple(config['data']['image_size'])
    preprocessor = DataPreprocessor(image_size=image_size)
    
    # Load dataset
    dataset_path = config['data']['dataset_path']
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        logger.info("Please download the LGG MRI Segmentation dataset from Kaggle:")
        logger.info("https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation")
        return None, None, None
    
    try:
        image_paths, mask_paths, labels = preprocessor.load_kaggle_lgg_dataset(dataset_path)
        logger.info(f"Loaded {len(image_paths)} images from dataset")
        
        # Get dataset statistics
        stats = preprocessor.get_dataset_statistics(labels)
        logger.info(f"Dataset statistics: {stats}")
        
        # Create data splits
        data_splits = preprocessor.create_data_splits(
            image_paths, mask_paths, labels,
            test_size=config['data']['test_size'],
            val_size=config['data']['val_size']
        )
        
        logger.info(f"Train samples: {len(data_splits['train']['images'])}")
        logger.info(f"Validation samples: {len(data_splits['val']['images'])}")
        logger.info(f"Test samples: {len(data_splits['test']['images'])}")
        
        # Create dataloaders
        cls_dataloaders = preprocessor.create_dataloaders(
            data_splits,
            batch_size=config['data']['batch_size_classification'],
            num_workers=config['data']['num_workers'],
            mode='classification'
        )
        
        seg_dataloaders = preprocessor.create_dataloaders(
            data_splits,
            batch_size=config['data']['batch_size_segmentation'],
            num_workers=config['data']['num_workers'],
            mode='segmentation'
        )
        
        logger.info("Data preprocessing completed successfully")
        return cls_dataloaders, seg_dataloaders, stats
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        return None, None, None


def train_classification_model(config: dict, dataloaders: dict, device: torch.device):
    """Train classification model"""
    logger.info("Starting classification model training...")
    
    # Create model
    model_config = config['models']['classification']
    model = get_model(
        model_config['model_name'],
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        task='classification',
        device=device,
        log_dir='runs/classification',
        save_dir='checkpoints/classification'
    )
    
    # Train model
    training_config = config['training']
    best_model_path = trainer.train(
        num_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['patience'],
        scheduler_type=training_config['scheduler_type']
    )
    
    logger.info(f"Classification training completed. Best model: {best_model_path}")
    return best_model_path


def train_segmentation_model(config: dict, dataloaders: dict, device: torch.device):
    """Train segmentation model"""
    logger.info("Starting segmentation model training...")
    
    # Create model
    model_config = config['models']['segmentation']
    model = get_model(
        model_config['model_name'],
        n_classes=model_config['n_classes'],
        bilinear=model_config.get('bilinear', True)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        task='segmentation',
        device=device,
        log_dir='runs/segmentation',
        save_dir='checkpoints/segmentation'
    )
    
    # Train model
    training_config = config['training']
    best_model_path = trainer.train(
        num_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['patience'],
        scheduler_type=training_config['scheduler_type']
    )
    
    logger.info(f"Segmentation training completed. Best model: {best_model_path}")
    return best_model_path


def evaluate_models(config: dict, cls_dataloader, seg_dataloader, 
                   cls_model_path: str, seg_model_path: str, device: torch.device):
    """Evaluate trained models"""
    logger.info("Starting model evaluation...")
    
    results = {}
    
    # Evaluate classification model
    if cls_model_path and os.path.exists(cls_model_path):
        logger.info("Evaluating classification model...")
        cls_model = get_model(
            config['models']['classification']['model_name'],
            num_classes=config['models']['classification']['num_classes'],
            pretrained=False
        )
        
        cls_evaluator = ModelEvaluator(cls_model, device, 'classification')
        cls_evaluator.load_checkpoint(cls_model_path)
        
        cls_metrics = cls_evaluator.evaluate_model(
            cls_dataloader['test'],
            'results/classification'
        )
        results['classification'] = cls_metrics
        logger.info(f"Classification evaluation completed. Accuracy: {cls_metrics.get('accuracy', 'N/A'):.4f}")
    
    # Evaluate segmentation model
    if seg_model_path and os.path.exists(seg_model_path):
        logger.info("Evaluating segmentation model...")
        seg_model = get_model(
            config['models']['segmentation']['model_name'],
            n_classes=config['models']['segmentation']['n_classes']
        )
        
        seg_evaluator = ModelEvaluator(seg_model, device, 'segmentation')
        seg_evaluator.load_checkpoint(seg_model_path)
        
        seg_metrics = seg_evaluator.evaluate_model(
            seg_dataloader['test'],
            'results/segmentation'
        )
        results['segmentation'] = seg_metrics
        logger.info(f"Segmentation evaluation completed. Dice: {seg_metrics.get('tumor_dice', 'N/A'):.4f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Brain Tumor Detection Pipeline')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'full'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation', 'both'],
                       default='both', help='Task to perform')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing (use existing dataloaders)')
    
    args = parser.parse_args()
    
    # Setup
    logger.info("="*50)
    logger.info("BRAIN TUMOR DETECTION PIPELINE")
    logger.info("="*50)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_directories()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Check CUDA info if available
    if device.type == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Data preprocessing
        if not args.skip_preprocessing:
            cls_dataloaders, seg_dataloaders, stats = preprocess_data(config)
            if cls_dataloaders is None:
                return 1
        else:
            logger.info("Skipping data preprocessing")
            cls_dataloaders, seg_dataloaders, stats = None, None, None
        
        cls_model_path = None
        seg_model_path = None
        
        # Training
        if args.mode in ['train', 'full']:
            if args.task in ['classification', 'both'] and cls_dataloaders:
                cls_model_path = train_classification_model(config, cls_dataloaders, device)
            
            if args.task in ['segmentation', 'both'] and seg_dataloaders:
                seg_model_path = train_segmentation_model(config, seg_dataloaders, device)
        
        # Evaluation
        if args.mode in ['evaluate', 'full']:
            # Use existing model paths if not training
            if args.mode == 'evaluate':
                cls_model_path = cls_model_path or 'checkpoints/classification/best_model.pth'
                seg_model_path = seg_model_path or 'checkpoints/segmentation/best_model.pth'
            
            if cls_dataloaders or seg_dataloaders:
                results = evaluate_models(
                    config, cls_dataloaders, seg_dataloaders,
                    cls_model_path, seg_model_path, device
                )
                
                # Save final results
                with open('results/final_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info("Final results saved to results/final_results.json")
        
        logger.info("="*50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error("Check the log file for detailed error information")
        return 1


if __name__ == "__main__":
    exit(main())