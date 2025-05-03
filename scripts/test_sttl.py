from audio_processing.stt_inference import STTEngine

stt = STTEngine()  # Should print loading messages
audio = ...  # Your numpy array
print(stt.transcribe(audio))  # Test transcription