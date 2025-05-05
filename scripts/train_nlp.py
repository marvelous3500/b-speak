import os
import warnings
from datetime import datetime
import logging
import torch

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# Third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Local application imports
from nlp_processing.intent_classifier.trainer import IntentClassifierTrainer
from nlp_processing.fraud_detector.trainer import FraudDetectionTrainer
from nlp_processing.dialog_utils import Vocabulary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INTENTS = [
    'transfer', 'payment', 'balance_check', "card_arrival", "card_linking",
    "exchange_rate",
    # ... (keep all your existing intents)
]

DATA_DIR = "data/conversations"
VOCAB_PATH = "data/vocab/nlp_vocab.txt"
MODEL_DIR = "models"

def timer_decorator(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"Completed {func.__name__} in {end_time - start_time}")
        return result
    return wrapper

@timer_decorator
def prepare_data():
    """Prepare and split datasets (handles both formats automatically)."""
    try:
        logger.info("Creating directories...")
        os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Process intent data
        logger.info("Processing intent data...")
        intent_df = pd.read_csv(os.path.join(DATA_DIR, "raw/banking_conversations.csv"))
        logger.info(f"Loaded {len(intent_df)} intent samples")
        
        train_intent, test_intent = train_test_split(intent_df, test_size=0.2, random_state=42)
        train_intent.to_csv(os.path.join(DATA_DIR, "processed/intent_train.csv"), index=False)
        test_intent.to_csv(os.path.join(DATA_DIR, "processed/intent_test.csv"), index=False)
        logger.info(f"Split intent data: {len(train_intent)} train, {len(test_intent)} test")
        
        # Process fraud data
        logger.info("Processing fraud data...")
        fraud_df = pd.read_csv(os.path.join(DATA_DIR, "raw/fraud_processed.csv"))
        logger.info(f"Loaded {len(fraud_df)} fraud samples")
        
        train_fraud, test_fraud = train_test_split(fraud_df, test_size=0.2, random_state=42)
        train_fraud.to_csv(os.path.join(DATA_DIR, "processed/fraud_train.csv"), index=False)
        test_fraud.to_csv(os.path.join(DATA_DIR, "processed/fraud_test.csv"), index=False)
        logger.info(f"Split fraud data: {len(train_fraud)} train, {len(test_fraud)} test")
        
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        raise

@timer_decorator
def train_models():
    try:
        # Initialize vocabulary
        logger.info("Initializing vocabulary...")
        vocab = Vocabulary()
        intent_train_df = pd.read_csv(os.path.join(DATA_DIR, "processed/intent_train.csv"))
        
        # Extract text based on format
        if 'text' in intent_train_df.columns:
            texts = intent_train_df['text'].tolist()
        elif 'conversation_id' in intent_train_df.columns:
            texts = intent_train_df[intent_train_df['speaker'] == 'client']['text'].tolist()
        
        logger.info(f"Creating vocabulary from {len(texts)} samples...")
        vocab.create_from_texts(texts)
        vocab.save(VOCAB_PATH)
        logger.info(f"Vocabulary saved with {len(vocab.tokenizer)} tokens")
        
        # Set device
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Train intent classifier
        logger.info("Training intent classifier...")
        intent_trainer = IntentClassifierTrainer(INTENTS)
        
        # Modified training call without DeepSpeed
        intent_trainer.train(
            data_path=os.path.join(DATA_DIR, "processed/intent_train.csv"),
            vocab_path=VOCAB_PATH,
            output_path=os.path.join(MODEL_DIR, "intent_model.keras"),
            epochs=15,
            batch_size=64,
        )

        logger.info("Training fraud detection model...")
        fraud_trainer = FraudDetectionTrainer()
        fraud_trainer.train(
            data_path=os.path.join(DATA_DIR, "processed/fraud_train.csv"),
            output_path=os.path.join(MODEL_DIR, "fraud_model.keras")
        )
        logger.info("Fraud detection model training completed")
        
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("=== Starting NLP Training Pipeline ===")
    try:
        # Verify torch setup
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        prepare_data()
        train_models()
        logger.info("=== Training completed successfully ===")
    except Exception as e:
        logger.error(f"!!! Training failed: {str(e)}")
        raise
