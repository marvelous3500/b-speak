# banking_voice_assistant/nlp_processing/dialog_utils.py

from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf


class DialogProcessor:
    
    def __init__(self, intent_model, fraud_model, tokenizer, intents):
        """
        Initialize the dialog processor.
        
        Args:
            intent_model: Loaded intent classification model
            fraud_model: Loaded fraud detection model
            tokenizer: Word to index mapping
            intents: List of supported intents
        """
        self.intent_model = intent_model
        self.fraud_model = fraud_model
        self.tokenizer = tokenizer
        self.intents = intents
        self.unknown_token = '<UNK>'
        self.sep_token = '[SEP]'
        self.pad_token = '<PAD>'
    
    def process_conversation(
        self,
        conversation: List[Dict[str, str]],
        context_window: int = 3
    ) -> Dict:
        """
        Process a conversation and return predictions.
        
        Args:
            conversation: List of {'speaker': str, 'text': str} dictionaries
            context_window: Number of previous turns to consider as context
            
        Returns:
            Dictionary with prediction results
        """
        # Get last customer utterance and context
        customer_utts = [t['text'] for t in conversation if t['speaker'] == 'client']
        if not customer_utts:
            return self._empty_response()
            
        last_utt = customer_utts[-1]
        context = [
            t['text'] for t in conversation[-context_window*2:] 
            if t['speaker'] == 'agent'
        ]
        
        # Prepare input
        input_text = self._create_input_text(last_utt, context)
        tokens = [self.tokenizer.get(word.lower(), 0) for word in input_text.split()]
        input_tensor = tf.convert_to_tensor([tokens], dtype=tf.int32)
        
        # Get predictions
        intent, confidence = self._predict_intent(input_tensor)
        is_fraud, fraud_confidence = self._predict_fraud(input_tensor)
        
        return {
            "intent": intent,
            "confidence": float(confidence),
            "is_fraud": is_fraud,
            "fraud_confidence": float(fraud_confidence),
            "context_used": context
        }
    
    def _create_input_text(self, utterance: str, context: List[str]) -> str:
        """Format conversation context for model input."""
        context_str = f" {self.sep_token} ".join(context)
        return f"{context_str} {self.sep_token} {utterance}" if context else utterance
    
    def _predict_intent(self, input_tensor: tf.Tensor) -> Tuple[str, float]:
        """Predict intent from input tensor."""
        intent_pred = self.intent_model.predict(input_tensor, verbose=0)
        intent_idx = np.argmax(intent_pred, axis=1)[0]
        return self.intents[intent_idx], float(np.max(intent_pred, axis=1)[0])
    
    def _predict_fraud(self, input_tensor: tf.Tensor) -> Tuple[bool, float]:
        """Predict fraud from input tensor."""
        if self.fraud_model is None:
            return False, 0.0
            
        pred = self.fraud_model.predict(input_tensor, verbose=0)
        confidence = float(pred[0][0])
        return confidence > 0.5, confidence
    
    def _empty_response(self) -> Dict:
        """Return empty response when no customer input is found."""
        return {
            "intent": "no_customer_input",
            "confidence": 0.0,
            "is_fraud": False,
            "fraud_confidence": 0.0,
            "context_used": []
        }


class Vocabulary:
    """Handles vocabulary creation and management."""
    
    def __init__(self):
        self.tokenizer = {}
        self.special_tokens = {
            '<UNK>': 0,
            '[SEP]': 1,
            '<PAD>': 2
        }
    
    def create_from_texts(self, texts: List[str]) -> None:
        """
        Create vocabulary from list of texts.
        
        Args:
            texts: List of text strings
        """
        self.tokenizer = self.special_tokens.copy()
        idx = len(self.special_tokens)
        
        for text in texts:
            for word in str(text).lower().split():
                if word not in self.tokenizer:
                    self.tokenizer[word] = idx
                    idx += 1
    
    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            for word, idx in sorted(self.tokenizer.items(), key=lambda x: x[1]):
                f.write(f"{word} {idx}\n")
    
    def load(self, path: str) -> None:
        """Load vocabulary from file."""
        self.tokenizer = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        word = ' '.join(parts[:-1])
                        idx = int(parts[-1])
                        self.tokenizer[word] = idx
                    except ValueError:
                        continue