# banking_voice_assistant/nlp_processing/intent_classifier/model.py

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import os
import deepspeed
from deepspeed.accelerator import get_accelerator
import json

class IntentClassifierModel:
    """Intent classification model with fraud detection capabilities."""
    
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        deepspeed_config: Optional[Dict] = None
    ):
        """
        Initialize the intent classifier model.
        
        Args:
            model_config: Configuration dictionary for model parameters
            deepspeed_config: DeepSpeed configuration dictionary
        """

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model_config = model_config or {
            "unknown_token": "<UNK>",
            "sep_token": "[SEP]",
            "pad_token": "<PAD>",
            "context_window": 3,
            "fraud_threshold": 0.5,
            "embedding_dim": 256,
            "lstm_units": 128,
            "dense_units": 128,
            "dropout_rate": 0.3
        }
        
        self.ds_config = deepspeed_config or {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 2,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    "weight_decay": 1e-5
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                }
            }
        }
        
        self.accelerator = get_accelerator()
        self.intent_model = None
        self.fraud_model = None
        self.tokenizer = None
        self.intents = None
        self.intent_keywords = None
        
    def initialize_models(self, vocab_size: int, num_intents: int) -> None:
        """Initialize the intent and fraud detection models."""
        # Intent classification model
        self.intent_model = tf.keras.Sequential([
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
            tf.keras.layers.Dropout(
                self.model_config["dropout_rate"],
                name="dropout"
            ),
            tf.keras.layers.Dense(
                num_intents,
                activation='softmax',
                name="output"
            )
        ])
        
        # Fraud detection model
        self.fraud_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=128,
                mask_zero=True,
                name="fraud_embedding"
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=False),
                name="fraud_bidirectional"
            ),
            tf.keras.layers.Dense(32, activation='relu', name="fraud_dense"),
            tf.keras.layers.Dense(1, activation='sigmoid', name="fraud_output")
        ])
    
    def _init_deepspeed(self, model, train_dataset):
        """Initialize DeepSpeed engine."""
        return deepspeed.initialize(
            model=model,
            config_params=self.ds_config,
            training_data=train_dataset,
            optimizer=None,  # Defined in config
        )
    
    def compile_models(self):
        """Compile both models with appropriate loss functions and metrics."""
        if self.intent_model is None or self.fraud_model is None:
            raise ValueError("Models must be initialized before compilation")
            
        self.intent_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.fraud_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def predict_intent(self, input_sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predict intent from input sequence.
        
        Args:
            input_sequence: Padded and tokenized input sequence
            
        Returns:
            Tuple of (predicted intent, confidence score)
        """
        if self.intent_model is None:
            raise ValueError("Intent model not loaded")
        if self.intents is None:
            raise ValueError("Intents not loaded")
            
        predictions = self.intent_model.predict(input_sequence, verbose=0)
        intent_idx = np.argmax(predictions, axis=1)[0]
        return self.intents[intent_idx], float(np.max(predictions, axis=1)[0])
    
    def detect_fraud(self, input_sequence: np.ndarray) -> Tuple[bool, float]:
        """
        Detect fraud from input sequence.
        
        Args:
            input_sequence: Padded and tokenized input sequence
            
        Returns:
            Tuple of (is_fraud, confidence score)
        """
        if self.fraud_model is None:
            return False, 0.0
            
        prediction = self.fraud_model.predict(input_sequence, verbose=0)
        confidence = float(prediction[0][0])
        return confidence > self.model_config["fraud_threshold"], confidence
    
    def save_models(self, intent_model_path: str, fraud_model_path: str) -> None:
        """Save both models to disk."""
        if self.intent_model:
            self.intent_model.save(intent_model_path)
        if self.fraud_model:
            self.fraud_model.save(fraud_model_path)
    
    def load_models(self, intent_model_path: str, fraud_model_path: str) -> None:
        """Load models from disk."""
        if os.path.exists(intent_model_path):
            self.intent_model = tf.keras.models.load_model(intent_model_path)
        if os.path.exists(fraud_model_path):
            self.fraud_model = tf.keras.models.load_model(fraud_model_path)
    
    def load_intents(self, intents: List[str]) -> None:
        """Load the list of supported intents."""
        self.intents = intents
        self._create_intent_keyword_map()
    
    def _create_intent_keyword_map(self) -> None:
        """Create mapping of keywords to intents for automatic labeling."""
        self.intent_keywords = defaultdict(list)
        
        # Basic mappings (should be expanded based on your specific data)
        self.intent_keywords['transfer'] = ['transfer', 'send money', 'move funds']
        self.intent_keywords['payment'] = ['pay', 'payment', 'bill']
        self.intent_keywords['balance_check'] = ['balance', 'how much', 'account status']
        # Add mappings for all other intents...
    
    def auto_label_intent(self, text: str) -> str:
        """Auto-detect intent from text using keyword matching."""
        text = text.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        return 'default_intent'