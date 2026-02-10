import os
import shutil
import logging
import subprocess
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from faster_whisper import WhisperModel
from tqdm import tqdm
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

CONFIG = {
    "SAMPLE_RATE": 22050,          
    "CHANNELS": 1,                 # Mono [cite: 7, 45]
    "BIT_DEPTH": "PCM_16",         # 16-bit PCM [cite: 7, 46]
    "MIN_SEGMENT_LEN": 2000,       # 2 seconds (ms) [cite: 12, 46]
    "MAX_SEGMENT_LEN": 15000,      # 15 seconds (ms) [cite: 12, 46]
    "MIN_SILENCE_LEN": 400,        # 400 ms [cite: 46]
    "SILENCE_THRESH": -38,        
    "OUTPUT_DIR": "wavs",
    "METADATA_FILE": "metadata.csv"
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioPipeline:
    def __init__(self, output_dir, metadata_file):
        self.output_dir = output_dir
        self.metadata_file = metadata_file
        self.metadata = []
        
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
        print(f"{Fore.CYAN} Loading Whisper Large-v3 model on GPU...{Style.RESET_ALL}")
        try:
            self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            print(f"{Fore.GREEN} Model Loaded Successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED} GPU not found or error. Falling back to CPU.{Style.RESET_ALL}")
            self.model = WhisperModel("medium", device="cpu", compute_type="int8")

    def download_audio(self, url, filename="raw_audio"):
        """Downloads audio using yt-dlp strictly in WAV format [cite: 7]"""
        print(f"{Fore.YELLOW} Downloading: {url}{Style.RESET_ALL}")
        
        output_template = f"{filename}.%(ext)s"

        command = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"ffmpeg:-ac {CONFIG['CHANNELS']} -ar {CONFIG['SAMPLE_RATE']}", # Mono, 22050Hz
            "-o", output_template,
            url
        ]
        
        subprocess.run(command, check=True)
        return f"{filename}.wav"

    def split_and_process(self, raw_audio_path):
        """Splits audio on silence and ensures 2-15s constraints [cite: 10, 11, 12, 13]"""
        print(f"{Fore.YELLOW} Processing & Splitting Audio...{Style.RESET_ALL}")
        
        audio = AudioSegment.from_wav(raw_audio_path)
        
        # Step 1: Split strictly on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=CONFIG['MIN_SILENCE_LEN'],
            silence_thresh=CONFIG['SILENCE_THRESH'],
            keep_silence=100 # Keep 100ms padding for natural sound
        )

        processed_chunks = []
        current_chunk = AudioSegment.empty()

        # Step 2: Merge logic (Constraint: Min 2s) [cite: 13]
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < CONFIG['MAX_SEGMENT_LEN']:
                current_chunk += chunk
            else:
                # If adding chunk exceeds max, save current and start new
                if len(current_chunk) >= CONFIG['MIN_SEGMENT_LEN']:
                    processed_chunks.append(current_chunk)
                current_chunk = chunk
        
        # Append last chunk if valid
        if len(current_chunk) >= CONFIG['MIN_SEGMENT_LEN']:
            processed_chunks.append(current_chunk)

        return processed_chunks

    def clean_audio(self, audio_segment):
        """Applies Noise Reduction & Normalization [cite: 15, 16, 20]"""
        
        # Convert Pydub to Numpy for Noise Reduce
        samples = np.array(audio_segment.get_array_of_samples())
        

        try:
            reduced_noise = nr.reduce_noise(y=samples, sr=CONFIG['SAMPLE_RATE'], prop_decrease=0.6, stationary=True)
            return reduced_noise
        except Exception:
            return samples 

    def run(self, youtube_url, speaker_name="speaker1"):
        raw_file = self.download_audio(youtube_url)
        
        chunks = self.split_and_process(raw_file)
        
        print(f"{Fore.CYAN}Processing {len(chunks)} segments...{Style.RESET_ALL}")

        for i, chunk in enumerate(tqdm(chunks, desc="Denoising & Transcribing")):
            # 1. Denoise [cite: 16]
            clean_samples = self.clean_audio(chunk)
            
            # 2. Save cleaned audio temporarily for Whisper
            filename = f"audio_{i:06d}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            # Export strictly as 16-bit PCM WAV [cite: 21, 46]
            sf.write(filepath, clean_samples, CONFIG['SAMPLE_RATE'], subtype=CONFIG['BIT_DEPTH'])
            
            # 3. Transcribe [cite: 23]
            try:
                segments, _ = self.model.transcribe(filepath, language="bn") # 'bn' for Bengali
                text = " ".join([segment.text for segment in segments]).strip()
                

                if text:
                    self.metadata.append(f"{filename}|{text}|{speaker_name}")
            except Exception as e:
                print(f"{Fore.RED}Transcription Failed for {filename}: {e}{Style.RESET_ALL}")

        # Save Metadata
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            for line in self.metadata:
                f.write(line + "\n")
        
        # Cleanup
        if os.path.exists(raw_file):
            os.remove(raw_file)
            
        print(f"{Fore.GREEN} Pipeline Complete! Metadata saved to {self.metadata_file}{Style.RESET_ALL}")

# --- EXECUTION ---
if __name__ == "__main__":

    YOUTUBE_URL = "https://youtu.be/OZUk5sVnvbQ?si=Wxc3fsfwSS5l-PwD" 
    
    pipeline = AudioPipeline(CONFIG['OUTPUT_DIR'], CONFIG['METADATA_FILE'])
    
    
    url = input("Enter YouTube URL: ")
    if url:
        pipeline.run(url, speaker_name="Habibur_Sample")