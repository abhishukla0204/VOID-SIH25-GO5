#!/usr/bin/env python3
"""
YOLOv8 Training Pipeline for Rockfall Detection
==============================================

This script trains a YOLOv8 model on the rockfall dataset.
Features:
- Configurable training parameters
- Model validation and evaluation
- Export to multiple formats (PyTorch, ONNX)
- Training metrics logging and visualization
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class RockfallTrainer:
    """YOLOv8 trainer for rockfall detection"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize trainer with configuration"""
        self.config_path = config_path
        self.load_config()
        self.setup_directories()
        
    def load_config(self):
        """Load training configuration"""
        # Default configuration
        self.config = {
            'model_size': 'yolov8n.pt',  # Start with nano for faster training
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.01,
            'patience': 20,  # Early stopping patience
            'workers': 4,
            'device': 'cpu',  # Force CPU since CUDA not available
            'project': 'rockfall_training',
            'name': f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'save_period': 10,  # Save model every N epochs
            'val_split': 0.2,
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        # Load custom config if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self.config.update(custom_config)
    
    def setup_directories(self):
        """Setup output directories"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data" / "rockfall_training_data"
        self.models_dir = self.project_root / "models"
        self.outputs_dir = self.project_root / "outputs"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"Data path: {self.data_path}")
        print(f"Models directory: {self.models_dir}")
    
    def validate_dataset(self):
        """Validate dataset structure and paths"""
        data_yaml_path = self.data_path / "data.yaml"
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
        
        # Load and validate data.yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print("Dataset configuration:")
        print(f"- Classes: {data_config['nc']}")
        print(f"- Class names: {data_config['names']}")
        
        # Check if paths exist
        for split in ['train', 'val', 'test']:
            if split in data_config:
                split_path = self.data_path / data_config[split]
                if split_path.exists():
                    image_count = len(list(split_path.glob("*.jpg"))) + len(list(split_path.glob("*.png")))
                    print(f"- {split.capitalize()}: {image_count} images found")
                else:
                    print(f"- Warning: {split} path not found: {split_path}")
        
        return str(data_yaml_path)
    
    def train_model(self):
        """Train the YOLOv8 model"""
        print("\n" + "="*60)
        print("Starting YOLOv8 Training for Rockfall Detection")
        print("="*60)
        
        # Validate dataset
        data_yaml_path = self.validate_dataset()
        
        # Initialize model
        print(f"\nInitializing YOLOv8 model: {self.config['model_size']}")
        model = YOLO(self.config['model_size'])
        
        # Print device information
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Training configuration
        train_args = {
            'data': data_yaml_path,
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['img_size'],
            'lr0': self.config['learning_rate'],
            'patience': self.config['patience'],
            'workers': self.config['workers'],
            'device': self.config['device'],
            'project': str(self.outputs_dir),
            'name': self.config['name'],
            'save_period': self.config['save_period'],
            'val': True,
            'plots': True,
            'verbose': True,
            'mosaic': self.config['mosaic'],
            'mixup': self.config['mixup'],
            'copy_paste': self.config['copy_paste'],
        }
        
        print(f"\nTraining configuration:")
        for key, value in train_args.items():
            print(f"- {key}: {value}")
        
        print(f"\nStarting training...")
        
        # Start training
        try:
            results = model.train(**train_args)
            
            # Training completed successfully
            print("\n" + "="*60)
            print("Training completed successfully!")
            print("="*60)
            
            return results, model
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise
    
    def evaluate_model(self, model):
        """Evaluate trained model"""
        print("\nEvaluating model...")
        
        # Validate on test set
        data_yaml_path = self.data_path / "data.yaml"
        results = model.val(data=str(data_yaml_path), split='test')
        
        # Print evaluation metrics
        print(f"\nEvaluation Results:")
        print(f"- mAP50: {results.box.map50:.4f}")
        print(f"- mAP50-95: {results.box.map:.4f}")
        print(f"- Precision: {results.box.mp:.4f}")
        print(f"- Recall: {results.box.mr:.4f}")
        
        return results
    
    def export_model(self, model, formats=['pt', 'onnx']):
        """Export model to different formats"""
        print(f"\nExporting model to formats: {formats}")
        
        export_paths = {}
        for format_type in formats:
            try:
                export_path = model.export(format=format_type)
                export_paths[format_type] = export_path
                print(f"- {format_type.upper()}: {export_path}")
                
                # Copy to models directory
                if export_path:
                    import shutil
                    filename = f"rockfall_yolov8_{self.config['model_size'].replace('.pt', '')}.{format_type}"
                    dest_path = self.models_dir / filename
                    shutil.copy2(export_path, dest_path)
                    print(f"  Copied to: {dest_path}")
                    
            except Exception as e:
                print(f"- Failed to export {format_type}: {str(e)}")
        
        return export_paths
    
    def save_training_summary(self, results, model):
        """Save training summary and metrics"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'model_info': {
                'architecture': self.config['model_size'],
                'parameters': sum(p.numel() for p in model.model.parameters()),
                'device': str(next(model.model.parameters()).device),
            },
            'performance': {
                'final_epoch': results.epoch if hasattr(results, 'epoch') else 'unknown',
                'best_fitness': float(results.fitness) if hasattr(results, 'fitness') else 'unknown',
            }
        }
        
        # Save summary
        summary_path = self.outputs_dir / f"training_summary_{self.config['name']}.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"\nTraining summary saved to: {summary_path}")
        return summary_path


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Rockfall Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--export', nargs='+', default=['pt', 'onnx'],
                       choices=['pt', 'onnx', 'torchscript', 'tflite'],
                       help='Export formats')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RockfallTrainer(args.config)
    
    # Override config with command line arguments
    trainer.config['model_size'] = args.model
    trainer.config['epochs'] = args.epochs
    trainer.config['batch_size'] = args.batch_size
    
    try:
        # Train model
        results, model = trainer.train_model()
        
        # Evaluate model
        eval_results = trainer.evaluate_model(model)
        
        # Export model
        export_paths = trainer.export_model(model, args.export)
        
        # Save training summary
        summary_path = trainer.save_training_summary(results, model)
        
        print("\n" + "="*60)
        print("🎉 Training pipeline completed successfully!")
        print("="*60)
        print(f"📁 Models saved in: {trainer.models_dir}")
        print(f"📊 Training outputs: {trainer.outputs_dir}")
        print(f"📋 Summary: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)