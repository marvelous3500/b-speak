#!/usr/bin/env python3
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from huggingface_hub import snapshot_download , login 
import argparse



def download_model(model_id: str, output_dir: str):
    try:
        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            resume_download=True,
            token=True  # Required for gated models
        )
        print(f"✅ Model saved to {output_dir}")
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./models/deepseek")
    args = parser.parse_args()

    # Try DeepSeek's model first, fallback to Whisper
    try:
        download_model("deepseek-ai/deepseek-whisper", args.output_dir)
    except:
        print("DeepSeek model not found, using OpenAI Whisper")
        download_model("openai/whisper-base", args.output_dir)