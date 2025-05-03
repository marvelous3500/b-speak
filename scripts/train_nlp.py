# banking_voice_assistant/scripts/train_nlp.py

import os
import warnings
from datetime import datetime
import logging

# Environment configuration
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# Third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split
import deepspeed
from deepspeed.accelerator import get_accelerator

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
            "exchange_rate", "card_payment_wrong_exchange_rate", "extra_charge_on_statement",
            "pending_cash_withdrawal", "fiat_currency_support", "card_delivery_estimate",
            "automatic_top_up", "card_not_working", "exchange_via_app", "lost_or_stolen_card",
            "age_limit", "pin_blocked", "contactless_not_working", "top_up_by_bank_transfer_charge",
            "pending_top_up", "cancel_transfer", "top_up_limits", "wrong_amount_of_cash_received",
            "card_payment_fee_charged", "transfer_not_received_by_recipient", 
            "supported_cards_and_currencies", "getting_virtual_card", "card_acceptance",
            "top_up_reverted", "balance_not_updated_after_cheque_or_cash_deposit",
            "card_payment_not_recognised", "edit_personal_details", "why_verify_identity",
            "unable_to_verify_identity", "get_physical_card", "visa_or_mastercard",
            "topping_up_by_card", "disposable_card_limits", "compromised_card", "atm_support",
            "direct_debit_payment_not_recognised", "passcode_forgotten", "declined_cash_withdrawal",
            "pending_card_payment", "lost_or_stolen_phone", "request_refund", "declined_transfer",
            "Refund_not_showing_up", "declined_card_payment", "pending_transfer", "terminate_account",
            "card_swallowed", "transaction_charged_twice", "verify_source_of_funds", "transfer_timing",
            "reverted_card_payment", "change_pin", "beneficiary_not_allowed", "transfer_fee_charged",
            "receiving_money", "failed_transfer", "transfer_into_account", "verify_top_up",
            "getting_spare_card", "top_up_by_cash_or_cheque", "order_physical_card",
            "virtual_card_not_working", "wrong_exchange_rate_for_cash_withdrawal",
            "get_disposable_virtual_card", "top_up_failed", "balance_not_updated_after_bank_transfer",
            "cash_withdrawal_not_recognised", "exchange_charge", "top_up_by_card_charge",
            "activate_my_card", "cash_withdrawal_charge", "card_about_to_expire",
            "apple_pay_or_google_pay", "verify_my_identity", "country_support", "declined_card_payment",
            "card_not_working","lost_or_stolen_card", "default_intent", "reverted_card_payment?"
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
    """Train both intent and fraud detection models."""
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
        
        # Train intent classifier
        logger.info("Training intent classifier...")
        intent_trainer = IntentClassifierTrainer(INTENTS)
        intent_trainer.train(
            data_path=os.path.join(DATA_DIR, "processed/intent_train.csv"),
            vocab_path=VOCAB_PATH,
            output_path=os.path.join(MODEL_DIR, "intent_model"),
            epochs=15,
            batch_size=64
        )

        logger.info("Intent classifier training completed")
        
        # Train fraud detector
        logger.info("Training fraud detection model...")
        fraud_trainer = FraudDetectorTrainer()
        fraud_trainer.train(
            data_path=os.path.join(DATA_DIR, "processed/fraud_train.csv"),
            vocab_path=VOCAB_PATH,
            output_path=os.path.join(MODEL_DIR, "fraud_model.h5")
        )
        logger.info("Fraud detection model training completed")
        
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("=== Starting NLP Training Pipeline ===")
    try:
        prepare_data()
        train_models()
        logger.info("=== Training completed successfully ===")
    except Exception as e:
        logger.error(f"!!! Training failed: {str(e)}")
        raise