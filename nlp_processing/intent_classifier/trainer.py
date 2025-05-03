import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import deepspeed
from transformers import AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.basicConfig(level=logging.INFO)

class IntentClassifierTrainer:
    def __init__(self, intents: List[str], data_path: str = 'data/conversations/raw/banking_conversations.csv'):
        self.intents = intents
        self.tokenizer = {
            '<UNK>': 0,  # Unknown token
            '<PAD>': 1,  # Padding token
            '[SEP]': 2   # Separator token
        }
        self.data_path = data_path
        self.model = None
        self.unknown_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.sep_token = '[SEP]'
        self.next_token_id = 3
        self._create_intent_keyword_map()

        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def _create_intent_keyword_map(self):
        """Create mapping of keywords to intents for automatic labeling"""
        self.intent_keywords = {
            'transfer': ['transfer', 'send money', 'move funds'],
            'payment': ['pay', 'payment', 'bill'],
            'balance_check': ['balance', 'how much', 'account status'],
            'card_arrival': ['card arrive', 'when card', 'delivery date'],
            'default_intent': ['none', 'others', 'none']
        }

    def _determine_intent(self, text: str) -> str:
        """Auto-detect intent from text using keyword matching"""
        if not isinstance(text, str):
            return 'default_intent'
        
        text = text.lower().strip()
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        return 'default_intent'

    def _load_training_data(self) -> Tuple[List[str], List[str]]:
        """Load and validate training data"""
        try:
            df = pd.read_csv(self.data_path)
            
            # Validate required columns
            required_cols = {'text', 'category'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Missing columns: {missing}")
            
            # Clean data
            df = df.dropna(subset=list(required_cols))
            df = df[df['text'].str.strip().astype(bool)]
            
            texts = df['text'].astype(str).tolist()
            labels = df['category'].astype(str).tolist()
            
            if not texts:
                raise ValueError("No valid texts found after cleaning")
                
            return texts, labels
            
        except Exception as e:
            raise ValueError(f"Failed to load training data: {str(e)}")

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text and build vocabulary incrementally"""
        tokens = []
        for word in text.lower().split():
            if word not in self.tokenizer:
                self.tokenizer[word] = self.next_token_id
                self.next_token_id += 1
            tokens.append(self.tokenizer[word])
        return tokens

    def _prepare_dataset(self, texts: List[str], labels: List[str]):
        """Prepare TensorFlow dataset with proper padding"""
        # Tokenize texts
        tokenized = [self._tokenize(text) for text in texts]
        
        # Convert labels to indices
        label_indices = [self.intents.index(label) for label in labels]
        
        # Pad sequences
        max_len = max(len(seq) for seq in tokenized)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized,
            maxlen=max_len,
            padding='post',
            value=self.tokenizer[self.pad_token]
        )
        
        return tf.data.Dataset.from_tensor_slices(
            (padded_sequences, tf.convert_to_tensor(label_indices)))



    def _build_model(self, max_len: int, num_intents: int):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_intents
            )
            
            # Simplified CPU-compatible config
            ds_config = {
                "train_batch_size": 8,
                "optimizer": {
                    "type": "AdamW",
                    "params": {"lr": 5e-5}
                },
                "zero_optimization": {
                    "stage": 2
                }
            }
            
            # Initialize DeepSpeed
            model_engine = deepspeed.initialize(
                model=model,
                config_params=ds_config,
                model_parameters=model.parameters()
            )[0]
            
            logging.info("Model successfully initialized with DeepSpeed")
            return model_engine
            
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            raise

   

    def train(self, data_path: str, vocab_path: str, output_path: str, epochs: int = 10, batch_size: int = 32):
        """Complete training pipeline"""
        try:
            # Load and validate data
            texts, labels = self._load_training_data()
            logging.info(f"Loaded {len(texts)} training examples")
            
            # Initialize model if needed
            if self.model is None:
                tokenized = [self._tokenize(text) for text in texts]
                max_len = max(len(seq) for seq in tokenized)
                self.model = self._build_model(max_len, len(self.intents))
                logging.info("Model initialized")
            
            # Prepare dataset
            dataset = self._prepare_dataset(texts, labels)
            dataset = dataset.shuffle(1000).batch(batch_size)
            
            # Training loop
            logging.info("Starting training...")
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for batch_X, batch_y in dataset:
                    outputs = self.model(batch_X, training=True)
                    loss = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(batch_y, outputs)
                    )
                    
                    self.model.backward(loss)
                    self.model.step()
                    
                    total_loss += loss.numpy()
                    num_batches += 1
                    
                    if num_batches % 10 == 0:
                        logging.info(
                            f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.numpy():.4f}"
                        )
                
                logging.info(
                    f"Epoch {epoch+1} completed, Avg Loss: {total_loss/num_batches:.4f}"
                )
            
            # Save model
            self.model.save_checkpoint(output_path)
            logging.info(f"Model saved to {output_path}")
            
            # Save vocabulary
            self._save_vocabulary(vocab_path)
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def _save_vocabulary(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w', encoding='utf-8') as f:
            for word, idx in sorted(self.tokenizer.items(), key=lambda x: x[1]):
                f.write(f"{word} {idx}\n")

    def _load_vocabulary(self, path: str) -> Dict[str, int]:
        """Load vocabulary from file"""
        vocab = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        word = ' '.join(parts[:-1])
                        idx = int(parts[-1])
                        vocab[word] = idx
                    except ValueError:
                        continue
        return vocab