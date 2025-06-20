"""
Advanced Medical Image Classification System
==========================================
A production-ready deep learning system for classifying lung and colon cancer 
histopathological images using state-of-the-art computer vision techniques.

Author: [Your Name]
Version: 2.0.0
"""

import os
import gc
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, CSVLogger
)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import mlflow
import mlflow.tensorflow

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

@dataclass
class Config:
    """Configuration class for hyperparameters and settings."""
    # Data parameters
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    TEST_SPLIT: float = 0.1
    
    # Training parameters
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    PATIENCE: int = 10
    MIN_DELTA: float = 1e-4
    
    # Model parameters
    DROPOUT_RATE: float = 0.3
    L2_REG: float = 1e-4
    
    # Paths
    DATA_PATH: str = 'lung-and-colon-cancer-histopathological-images.zip'
    OUTPUT_DIR: str = 'outputs'
    MODEL_DIR: str = 'models'
    LOGS_DIR: str = 'logs'

class Logger:
    """Enhanced logging utility."""
    
    def __init__(self, name: str = __name__, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            # File handler
            file_handler = logging.FileHandler(f'{Config.LOGS_DIR}/training.log')
            file_handler.setLevel(level)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str): self.logger.info(message)
    def warning(self, message: str): self.logger.warning(message)
    def error(self, message: str): self.logger.error(message)

class DataManager:
    """Advanced data management and preprocessing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(__name__)
        self.classes = ['lung_aca', 'lung_n', 'lung_scc', 'colon_aca', 'colon_n']
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config.OUTPUT_DIR, self.config.MODEL_DIR, self.config.LOGS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def extract_data(self) -> str:
        """Extract dataset from zip file."""
        if not os.path.exists(self.config.DATA_PATH):
            raise FileNotFoundError(f"Dataset not found: {self.config.DATA_PATH}")
        
        extract_path = 'lung_colon_image_set'
        if not os.path.exists(extract_path):
            self.logger.info("Extracting dataset...")
            with ZipFile(self.config.DATA_PATH, 'r') as zip_file:
                zip_file.extractall()
            self.logger.info("Dataset extracted successfully.")
        
        return os.path.join(extract_path, 'lung_image_sets')
    
    def analyze_dataset(self, data_path: str) -> Dict:
        """Comprehensive dataset analysis."""
        analysis = {'classes': {}, 'total_images': 0}
        
        for class_name in self.classes:
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                images = glob(f"{class_path}/*.jpeg")
                analysis['classes'][class_name] = len(images)
                analysis['total_images'] += len(images)
        
        self.logger.info(f"Dataset Analysis: {analysis}")
        return analysis
    
    def create_data_generators(self, data_path: str) -> Tuple:
        """Create advanced data generators with augmentation."""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            seed=RANDOM_STATE
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            seed=RANDOM_STATE
        )
        
        return train_generator, validation_generator
    
    def visualize_samples(self, data_path: str, save_path: str = None):
        """Create comprehensive data visualization."""
        fig, axes = plt.subplots(len(self.classes), 3, figsize=(15, 4*len(self.classes)))
        fig.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
        
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(data_path, class_name)
            images = glob(f"{class_path}/*.jpeg")
            
            for j in range(3):
                if j < len(images):
                    img_path = np.random.choice(images)
                    img = Image.open(img_path)
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f'{class_name}', fontweight='bold')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ModelArchitect:
    """Advanced model architecture design."""
    
    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.logger = Logger(__name__)
    
    def build_efficientnet_model(self) -> keras.Model:
        """Build EfficientNet-based model with custom head."""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(
                512, 
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.L2_REG)
            ),
            layers.BatchNormalization(),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(
                256, 
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.L2_REG)
            ),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ], name='EfficientNet_Medical_Classifier')
        
        return model
    
    def build_custom_cnn(self) -> keras.Model:
        """Build custom CNN architecture."""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(self.config.L2_REG)),
            layers.BatchNormalization(),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(self.config.L2_REG)),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='Custom_CNN_Medical_Classifier')
        
        return model

class TrainingManager:
    """Advanced training management with MLflow integration."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(__name__)
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow for experiment tracking."""
        mlflow.set_experiment("Medical_Image_Classification")
    
    def get_callbacks(self, model_name: str) -> List:
        """Get comprehensive callbacks for training."""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'{self.config.MODEL_DIR}/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'{self.config.LOGS_DIR}/tensorboard',
                histogram_freq=1
            ),
            CSVLogger(f'{self.config.LOGS_DIR}/training_log.csv')
        ]
        return callbacks
    
    def train_model(self, model: keras.Model, train_gen, val_gen, 
                   model_name: str) -> keras.callbacks.History:
        """Train model with comprehensive logging."""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                'model_name': model_name,
                'img_size': self.config.IMG_SIZE,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'epochs': self.config.EPOCHS
            })
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.logger.info(f"Starting training for {model_name}")
            
            # Train model
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=self.config.EPOCHS,
                callbacks=self.get_callbacks(model_name),
                verbose=1
            )
            
            # Log metrics
            final_accuracy = max(history.history['val_accuracy'])
            mlflow.log_metric('final_val_accuracy', final_accuracy)
            
            # Log model
            mlflow.tensorflow.log_model(model, f"models/{model_name}")
            
            self.logger.info(f"Training completed. Best validation accuracy: {final_accuracy:.4f}")
            
            return history

class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self, config: Config, class_names: List[str]):
        self.config = config
        self.class_names = class_names
        self.logger = Logger(__name__)
    
    def plot_training_history(self, history: keras.callbacks.History, 
                            model_name: str, save_path: str = None):
        """Plot comprehensive training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str, 
                            save_path: str = None):
        """Plot confusion matrix with enhanced visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, model_name: str) -> Dict:
        """Generate comprehensive classification report."""
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        self.logger.info(f"Classification Report for {model_name}:")
        self.logger.info(f"Accuracy: {report['accuracy']:.4f}")
        
        return report

class MedicalImageClassifier:
    """Main class orchestrating the entire pipeline."""
    
    def __init__(self):
        self.config = Config()
        self.logger = Logger(__name__)
        self.data_manager = DataManager(self.config)
        
    def run_pipeline(self):
        """Execute the complete machine learning pipeline."""
        self.logger.info("Starting Medical Image Classification Pipeline")
        
        try:
            # Data preparation
            data_path = self.data_manager.extract_data()
            analysis = self.data_manager.analyze_dataset(data_path)
            
            # Visualize data
            self.data_manager.visualize_samples(
                data_path, 
                f'{self.config.OUTPUT_DIR}/sample_images.png'
            )
            
            # Create data generators
            train_gen, val_gen = self.data_manager.create_data_generators(data_path)
            
            # Model creation and training
            num_classes = len(self.data_manager.classes)
            model_architect = ModelArchitect(self.config, num_classes)
            trainer = TrainingManager(self.config)
            evaluator = ModelEvaluator(self.config, self.data_manager.classes)
            
            # Train EfficientNet model
            efficientnet_model = model_architect.build_efficientnet_model()
            efficientnet_history = trainer.train_model(
                efficientnet_model, train_gen, val_gen, "EfficientNet"
            )
            
            # Train Custom CNN model
            custom_model = model_architect.build_custom_cnn()
            custom_history = trainer.train_model(
                custom_model, train_gen, val_gen, "CustomCNN"
            )
            
            # Evaluation
            self.evaluate_models(evaluator, efficientnet_model, custom_model, 
                               efficientnet_history, custom_history, val_gen)
            
            self.logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def evaluate_models(self, evaluator, model1, model2, history1, history2, val_gen):
        """Evaluate and compare models."""
        # Plot training histories
        evaluator.plot_training_history(
            history1, "EfficientNet", 
            f'{self.config.OUTPUT_DIR}/efficientnet_history.png'
        )
        evaluator.plot_training_history(
            history2, "Custom CNN", 
            f'{self.config.OUTPUT_DIR}/custom_cnn_history.png'
        )
        
        # Generate predictions and evaluate
        val_gen.reset()
        y_true = val_gen.classes
        
        # EfficientNet evaluation
        pred1 = model1.predict(val_gen)
        y_pred1 = np.argmax(pred1, axis=1)
        evaluator.plot_confusion_matrix(
            y_true, y_pred1, "EfficientNet",
            f'{self.config.OUTPUT_DIR}/efficientnet_cm.png'
        )
        report1 = evaluator.generate_classification_report(y_true, y_pred1, "EfficientNet")
        
        # Custom CNN evaluation
        val_gen.reset()
        pred2 = model2.predict(val_gen)
        y_pred2 = np.argmax(pred2, axis=1)
        evaluator.plot_confusion_matrix(
            y_true, y_pred2, "Custom CNN",
            f'{self.config.OUTPUT_DIR}/custom_cnn_cm.png'
        )
        report2 = evaluator.generate_classification_report(y_true, y_pred2, "Custom CNN")
        
        # Save reports
        with open(f'{self.config.OUTPUT_DIR}/evaluation_reports.json', 'w') as f:
            json.dump({
                'EfficientNet': report1,
                'CustomCNN': report2
            }, f, indent=2)

if __name__ == "__main__":
    # Initialize and run the classifier
    classifier = MedicalImageClassifier()
    classifier.run_pipeline()
    
    # Memory cleanup
    gc.collect()
    tf.keras.backend.clear_session()
    
    print("\n" + "="*80)
    print("🎉 MEDICAL IMAGE CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
