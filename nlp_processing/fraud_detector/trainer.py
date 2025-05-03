# banking_voice_assistant/nlp_processing/fraud_detector/trainer.py

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple 
import os
from .model import FraudDetectionModel  # Relative import


class FraudDetectionTrainer:
    """Handles training of the fraud detection model."""
    
    def __init__(self, model: FraudDetectionModel):
        self.model = model
    
    def prepare_training_data(
        self,
        data_path: str,
        tokenizer: Dict[str, int],
        max_sequence_length: Optional[int] = None,
        pad_token_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Prepare fraud detection training data.
        
        Args:
            data_path: Path to fraud data CSV
            tokenizer: Word to index mapping
            max_sequence_length: Maximum sequence length for padding
            pad_token_id: ID of padding token
            
        Returns:
            Tuple of (X, y, max_sequence_length)
        """
        df = pd.read_csv(data_path)
        texts = df["text"].values
        labels = df["is_fraud"].values
        
        # Convert texts to sequences
        tokenized = [
            [tokenizer.get(word.lower(), 0) for word in str(text).split()]
            for text in texts
        ]
        
        # Determine max sequence length
        if max_sequence_length is None:
            max_sequence_length = max(len(seq) for seq in tokenized)
        
        # Pad sequences
        X = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized,
            maxlen=max_sequence_length,
            padding='post',
            truncating='post',
            value=pad_token_id
        )
        
        return X, labels, max_sequence_length
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 64,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the fraud detection model.
        
        Args:
            X_train: Training input sequences
            y_train: Training labels
            X_val: Validation input sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model.model is None:
            raise ValueError("Model must be built before training")
            
        return self.model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=1
        )