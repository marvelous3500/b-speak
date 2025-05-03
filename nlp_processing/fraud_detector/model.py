# banking_voice_assistant/nlp_processing/fraud_detector/model.py
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict


class FraudDetectionModel:
    """Fraud detection model for banking conversations."""
    
    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize the fraud detection model.
        
        Args:
            model_config: Configuration dictionary for model parameters
        """
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_config = model_config or {
            "embedding_dim": 128,
            "lstm_units": 64,
            "dense_units": 32,
            "threshold": 0.5
        }
        
        self.model = None
    
    def build_model(self, vocab_size: int) -> None:
        """Build the fraud detection model architecture."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=self.model_config["embedding_dim"],
                mask_zero=True,
                name="embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.model_config["lstm_units"],
                    return_sequences=False,
                    name="lstm"
                ),
                name="bidirectional"
            ),
            tf.keras.layers.Dense(
                self.model_config["dense_units"],
                activation='relu',
                name="dense"
            ),
            tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                name="output"
            )
        ])
    
    def compile_model(self) -> None:
        """Compile the model with appropriate loss and metrics."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
            
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def predict(self, input_sequence: np.ndarray) -> Tuple[bool, float]:
        """
        Make a fraud prediction.
        
        Args:
            input_sequence: Padded and tokenized input sequence
            
        Returns:
            Tuple of (is_fraud, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        prediction = self.model.predict(input_sequence, verbose=0)
        confidence = float(prediction[0][0])
        return confidence > self.model_config["threshold"], confidence
    
    def save(self, model_path: str) -> None:
        """Save the model to disk."""
        if self.model:
            self.model.save(model_path)
    
    def load(self, model_path: str) -> None:
        """Load the model from disk."""
        self.model = tf.keras.models.load_model(model_path)