import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
INTENTS = ['transfer', 'payment', 'balance_check', 'card_arrival', 'card_linking', 'exchange_rate']

class ModelTester:
    def __init__(self):
        self.vectorizer = tf.saved_model.load(f"{MODEL_DIR}/text_vectorizer")
        self.intent_model = load_model(f"{MODEL_DIR}/intent_model.keras", compile=False)
        self.fraud_model = load_model(f"{MODEL_DIR}/fraud_model.keras", compile=False)
    
    def predict(self, text):
        # Vectorize text
        vectorized = self.vectorizer(tf.constant([text])).numpy()
        
        # Predict intent
        intent_pred = self.intent_model.predict(vectorized)
        intent = INTENTS[np.argmax(intent_pred[0])]
        
        # Check fraud
        fraud_pred = self.fraud_model.predict(vectorized)
        is_fraud = fraud_pred[0][0] > 0.5
        
        return {
            "intent": intent,
            "is_fraud": bool(is_fraud),
            "confidence": float(np.max(intent_pred))
        }

if __name__ == "__main__":
    tester = ModelTester()
    result = tester.predict("I need to transfer $1000")
    print(result)