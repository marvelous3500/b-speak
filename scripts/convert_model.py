import os
from pathlib import Path
from tensorflow.keras.models import load_model, save_model
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def convert_model():
    """Convert and optimize the fraud detection model format."""
    try:
        # Get absolute project paths
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        model_path = models_dir / "fraud_model.keras"
        temp_output = models_dir / "fraud_model_converted.keras"
        
        logger.info(f"Looking for model at: {model_path}")

        # Verify model exists
        if not model_path.exists():
            available_models = list(models_dir.glob("*.keras"))
            raise FileNotFoundError(
                f"Model file not found at {model_path}\n"
                f"Available models: {[m.name for m in available_models]}"
            )

        logger.info("Loading existing model...")
        model = load_model(model_path)
        
        logger.info("Saving converted model...")
        save_model(model, temp_output)
        
        logger.info("Replacing old model file...")
        model_path.unlink(missing_ok=True)  # Remove old file if exists
        temp_output.rename(model_path)  # Rename new file
        
        logger.info(f"✅ Successfully converted model at {model_path}")
        logger.info("Verification:")
        for model_file in sorted(models_dir.glob("fraud_model*")):
            size_kb = model_file.stat().st_size / 1024
            logger.info(f" - {model_file.name} ({size_kb:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {str(e)}")
        if 'temp_output' in locals() and temp_output.exists():
            temp_output.unlink()
        return False

if __name__ == "__main__":
    logger.info("=== Starting model conversion ===")
    success = convert_model()
    if success:
        logger.info("=== Conversion completed successfully ===")
    else:
        logger.error("!!! Conversion failed")