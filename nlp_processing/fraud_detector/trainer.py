from typing import Optional, Tuple, Dict, List
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import os


class FraudDetectionTrainer:
    def __init__(self, model=None):
        self.model = model
        self._configure_tensorflow()
        
    def _configure_tensorflow(self):
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Enabled mixed precision training")
        except ValueError:
            pass
    
    def prepare_training_data(
        self,
        data_path: str
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Prepare numerical fraud detection training data.
        """
        df = pd.read_csv(data_path)
        
        # Use whichever fraud label column exists
        fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'isFraud'
        
        # Select numerical features only
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != fraud_col]
        
        X = df[numerical_cols].values
        y = df[fraud_col].values.astype(int)
        
        return X, y, X.shape[1]  # Return number of features
    
    def create_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        buffer_size: int = 10000
    ) -> tf.data.Dataset:
        """
        Create TensorFlow Dataset for efficient training.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
            
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def train(
        self,
        data_path: str,
        output_path: str,
        epochs: int = 10,
        batch_size: int = 64,
        val_split: float = 0.2
    ) -> tf.keras.callbacks.History:
        """
        Complete training pipeline for fraud detection.
        """
        try:
            # Prepare data
            X, y, num_features = self.prepare_training_data(data_path)
            
            # Split into train/validation
            val_size = int(len(X) * val_split)
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
            
            # Create datasets
            train_dataset = self.create_dataset(X_train, y_train, batch_size)
            val_dataset = self.create_dataset(X_val, y_val, batch_size, shuffle=False)
            
            # Initialize model if not provided
            if self.model is None:
                from .model import FraudDetectionModel
                self.model = FraudDetectionModel(num_features=num_features)
                self.model.build_model()
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(output_path, 'best_model.keras'),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            logging.info("Starting fraud detection model training...")
            history = self.model.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=2 if logging.getLogger().level == logging.INFO else 1
            )
            
            # Save final model
            self.model.model.save(os.path.join(output_path, 'fraud_model.keras'))
            logging.info(f"Training complete. Model saved to {output_path}")
            
            return history
            
        except Exception as e:
            logging.error(f"Fraud detection training failed: {str(e)}")
            raise