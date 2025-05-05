# scripts/convert_fraud_model.py
import os
import shutil
from tensorflow.keras.models import load_model, save_model

def convert_model():
    # Absolute path to the model file (adjusted for your specific location)
    model_path = "/Users/macbookm1/Desktop/B-assistent/models/fraud_model.keras"
    
    # Temporary output path
    temp_output = "/Users/macbookm1/Desktop/B-assistent/models/fraud_model_converted.keras"
    
    try:
        # 1. Load the existing model
        model = load_model(model_path)
        
        # 2. Save with proper .keras format
        save_model(model, temp_output)
        
        # 3. Replace the old file
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rename(temp_output, model_path)
        
        print(f"✅ Successfully converted model at {model_path}")
        print("Verification:")
        os.system(f"ls -lh {os.path.dirname(model_path)}/fraud_model*")
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)

if __name__ == "__main__":
    convert_model()