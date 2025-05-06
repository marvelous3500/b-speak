import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
INTENTS = ['transfer', 'payment', 'balance_check', 'card_arrival', 'card_linking', 'exchange_rate']

class ModelTester:
    def __init__(self):
        try:
            # Load models
            self.intent_model = load_model(os.path.join(MODEL_DIR, "intent_model.keras"), compile=False)
            self.fraud_model = load_model(os.path.join(MODEL_DIR, "fraud_model.keras"), compile=False)
            
            # Load vectorizer with dictionary support
            self.vectorizer = self._load_vectorizer()
            self._test_vectorizer()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _load_vectorizer(self):
        """Loads vectorizer handling dictionary outputs"""
        vectorizer_dir = os.path.join(MODEL_DIR, "text_vectorizer")
        
        # Verify files exist
        required_files = [
            os.path.join(vectorizer_dir, "saved_model.pb"),
            os.path.join(vectorizer_dir, "variables/variables.index")
        ]
        for f in required_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing vectorizer file: {f}")
        
        # Load with signature
        loaded = tf.saved_model.load(vectorizer_dir)
        if 'serving_default' in loaded.signatures:
            return loaded.signatures['serving_default']
        return loaded

    def _test_vectorizer(self):
        """Tests vectorizer with dictionary output support"""
        try:
            test_output = self.vectorize_text(["test input"])
            if len(test_output.shape) != 2:
                raise ValueError("Vectorizer output has wrong shape")
            logger.info("Vectorizer test passed")
        except Exception as e:
            raise RuntimeError(f"Vectorizer test failed: {str(e)}")

    def vectorize_text(self, texts):
        """Handles both direct and dictionary outputs"""
        result = self.vectorizer(tf.constant(texts))
        if isinstance(result, dict):
            return result['output']  # Handle dictionary output
        return result  # Handle direct tensor output

    def predict(self, text):
        try:
            # 1. Vectorize the input text
            vectorized = self.vectorize_text([text])
            logger.debug(f"Vectorized output shape: {vectorized.shape}")

            print(f"Model input shape: {self.intent_model.input_shape}")
            
            # # 2. Verify the output shape matches model expectations
            expected_shape = self.intent_model.input_shape[1]  # Get (None, 379) -> 379
            
            # 3. Make predictions
            intent_pred = self.intent_model.predict(vectorized, verbose=0)
            fraud_pred = self.fraud_model.predict(vectorized, verbose=0)
            
            # 4. Process results
            intent = INTENTS[np.argmax(intent_pred[0])]
            is_fraud = bool(fraud_pred[0][0] > 0.5)
            confidence = float(np.max(intent_pred[0]))
            
            return {
                "intent": intent,
                "is_fraud": is_fraud,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            
            # Diagnostic information
            logger.debug("Model input shapes:")
            logger.debug(f"Intent model: {self.intent_model.input_shape}")
            logger.debug(f"Fraud model: {self.fraud_model.input_shape}")
            
            # Suggest specific fixes
            if "Shape mismatch" in str(e):
                logger.info("\nTroubleshooting steps:")
                logger.info("1. Check vectorizer configuration in train_nlp.py:")
                logger.info(f"   output_sequence_length={expected_shape}")
                logger.info("2. Delete and retrain models:")
                logger.info("   rm -rf models/text_vectorizer models/*.keras")
                logger.info("   python train_nlp.py")
            
            raise RuntimeError("Prediction aborted due to errors") from e

if __name__ == "__main__":
    try:
        tester = ModelTester()
        result = tester.predict(" I did not get my card yet, is it lost? ")
        print(result)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Delete vectorizer: rm -rf models/text_vectorizer")
        logger.info("2. Retrain: python train_nlp.py")
        logger.info("3. Verify: ls -l models/text_vectorizer/")