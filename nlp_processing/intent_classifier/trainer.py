import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple
from pathlib import Path

class IntentClassifierTrainer:
    def __init__(self, intents: List[str]):
        self.intents = intents
        self.model = None
        self.tokenizer = None
        self.label_map = {label: idx for idx, label in enumerate(intents)}
        self._configure_tensorflow()

    def _configure_tensorflow(self):
        """Configure TensorFlow settings for optimal performance"""
        # Enable mixed precision if available
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Enabled mixed precision training")
        except ValueError:
            pass

    # def _load_training_data(self, data_path: str) -> Tuple[List[str], List[int]]:
    #     """Load and preprocess training data"""
    #     df = pd.read_csv(data_path)
    #     texts = df['text'].astype(str).tolist()
    #     labels = [self.label_map[label] for label in df['category']]
    #     return texts, labels


    def _load_training_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        try:
            # Load data with explicit dtype and NaN handling
            df = pd.read_csv(
                data_path,
                dtype={'text': str, 'category': str},
                usecols=['text', 'category'],  # Only load needed columns
                na_values=['', ' ', 'nan', 'NaN', 'N/A', 'null'],
                keep_default_na=True
            )
            
            # Drop rows with missing values in either column
            df = df.dropna(subset=['text', 'category'])
            
            # Validate we have data remaining
            if len(df) == 0:
                raise ValueError("No valid training data after NaN removal")
                
            # Convert texts
            texts = df['text'].str.strip().tolist()
            
            # Convert labels with validation
            valid_labels = []
            valid_texts = []
            missing_labels = set()
            
            for text, label in zip(df['text'], df['category']):
                if label in self.label_map:
                    valid_labels.append(self.label_map[label])
                    valid_texts.append(text)
                else:
                    missing_labels.add(label)
            
            # Log warnings about missing labels
            if missing_labels:
                logger.warning(f"{len(missing_labels)} unknown labels encountered: {missing_labels}")
                logger.warning(f"Dropped {len(df) - len(valid_labels)} samples with invalid labels")
            
            if not valid_labels:
                raise ValueError("No valid labels found after filtering")
                
            return valid_texts, valid_labels
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise

    def _create_tokenizer(self, texts: List[str]) -> tf.keras.preprocessing.text.Tokenizer:
        """Create and fit tokenizer on texts"""
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        return tokenizer

    def _prepare_dataset(self, texts: List[str], labels: List[int], batch_size: int = 32) -> tf.data.Dataset:
        """Create TensorFlow dataset"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        max_len = max(len(seq) for seq in sequences)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_len, padding='post'
        )
        
        dataset = tf.data.Dataset.from_tensor_slices((
            padded,
            np.array(labels)
        ))  # Properly closed parentheses
        return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def _build_model(self, vocab_size: int, max_len: int) -> tf.keras.Model:
        """Build intent classification model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=vocab_size + 1,
                output_dim=256,
                input_length=max_len,
                mask_zero=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.intents), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, data_path: str, vocab_path: str, output_path: str, 
              epochs: int = 10, batch_size: int = 32):
        """Complete training pipeline"""
        try:
            # 1. Load and prepare data
            texts, labels = self._load_training_data(data_path)
            logging.info(f"Loaded {len(texts)} training examples")
            
            # 2. Initialize tokenizer and vocabulary
            self.tokenizer = self._create_tokenizer(texts)
            vocab_size = len(self.tokenizer.word_index)
            logging.info(f"Vocabulary size: {vocab_size}")
            
            # 3. Prepare dataset
            dataset = self._prepare_dataset(texts, labels, batch_size)
            
            # 4. Build and train model
            sample_input = next(iter(dataset))[0]
            max_len = sample_input.shape[1]
            
            self.model = self._build_model(vocab_size, max_len)
            logging.info("Model architecture:")
            self.model.summary(print_fn=logging.info)
            
            # 5. Train model
            logging.info("Starting training...")
            history = self.model.fit(
                dataset,
                epochs=epochs,
                verbose=2 if logging.getLogger().level == logging.INFO else 1
            )
            
            # 6. Save final artifacts
            self._save_vocabulary(vocab_path)
            self.model.save(output_path)
            logging.info(f"Training complete. Model saved to {output_path}")
            
            return history
        
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def _save_vocabulary(self, path: str):
        """Save vocabulary to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for word, idx in self.tokenizer.word_index.items():
                f.write(f"{word}\n")