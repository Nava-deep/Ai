import logging
import speech_recognition as sr
from transformers import pipeline
import sys
import os
from typing import Optional
import librosa
import soundfile as sf

# Configure logging for tracking execution and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        # Initialize speech recognizer and Whisper model
        self.recognizer = sr.Recognizer()
        try:
            self.whisper = pipeline("automatic-speech-recognition", model=model_name)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            sys.exit(1)

    def preprocess_audio(self, audio_path: str) -> str:
        # Preprocess audio to ensure compatibility with Whisper model
        try:
            output_path = "processed_audio.wav"
            y, sr = librosa.load(audio_path, sr=16000)
            sf.write(output_path, y, sr)
            logger.info(f"Audio preprocessed and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            return ""

    def transcribe_with_speechrecognition(self, audio_path: str) -> str:
        # Transcribe audio using SpeechRecognition with Google API
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            transcription = self.recognizer.recognize_google(audio)
            logger.info("SpeechRecognition transcription successful")
            return transcription
        except Exception as e:
            logger.error(f"SpeechRecognition transcription failed: {str(e)}")
            return f"Error: {str(e)}"

    def transcribe_with_whisper(self, audio_path: str) -> str:
        # Transcribe audio using Whisper model
        try:
            processed_audio = self.preprocess_audio(audio_path)
            if not processed_audio:
                return "Error: Audio preprocessing failed"
            result = self.whisper(processed_audio)
            os.remove(processed_audio)
            logger.info("Whisper transcription successful")
            return result["text"]
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return f"Error: {str(e)}"

    def transcribe(self, audio_path: str) -> dict:
        # Perform transcription using both methods and return results
        if not os.path.exists(audio_path):
            logger.error("Audio file not found")
            return {"speech_recognition": "Error: Audio file not found", "whisper": "Error: Audio file not found"}
        return {
            "speech_recognition": self.transcribe_with_speechrecognition(audio_path),
            "whisper": self.transcribe_with_whisper(audio_path)
        }

def main():
    # Initialize speech-to-text system and process sample audio
    stt = SpeechToText()
    sample_audio = "sample_audio.wav"
    if not os.path.exists(sample_audio):
        logger.info("Creating sample audio file")
        import numpy as np
        sample_rate = 16000
        t = np.linspace(0, 2, 2 * sample_rate)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(sample_audio, audio, sample_rate)
    result = stt.transcribe(sample_audio)
    print(f"\n{'='*50}\nTranscription Results\n{'='*50}")
    print(f"SpeechRecognition: {result['speech_recognition']}")
    print(f"Whisper: {result['whisper']}")

if __name__ == "__main__":
    main()
