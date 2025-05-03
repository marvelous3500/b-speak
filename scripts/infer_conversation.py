from nlp_processing.intent_classifier.model import IntentClassifierModel
from nlp_processing.fraud_detector.model import FraudDetectorModel

INTENTS = [
    # Your complete list of intents
]

class BankingAssistant:
    def __init__(self):
        self.intent_classifier = IntentClassifierModel(
            INTENTS,
            vocab_path="data/vocab/nlp_vocab.txt",
            model_path="models/intent_model.h5"
        )
        self.fraud_detector = FraudDetectorModel(
            vocab_path="data/vocab/nlp_vocab.txt",
            model_path="models/fraud_model.h5"
        )
    
    def process_conversation(self, conversation: List[Dict]) -> Dict:
        """End-to-end processing of conversation"""
        intent_result = self.intent_classifier.predict(conversation)
        fraud_result = self.fraud_detector.detect(conversation)
        
        return {
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "is_fraud": fraud_result["is_fraud"],
            "fraud_confidence": fraud_result["confidence"],
            "context_used": intent_result["context_used"]
        }

if __name__ == "__main__":
    assistant = BankingAssistant()
    conversation = [
        # Example conversation
    ]
    result = assistant.process_conversation(conversation)
    print(result)