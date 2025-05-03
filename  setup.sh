python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Install dependencies
pip install transformers torch huggingface-hub

# Download model
python scripts/download_deepseek.py --output_dir ../audio_processing/stt_models/deepseek

