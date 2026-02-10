import os
import shutil
import logging
import subprocess
import time
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from faster_whisper import WhisperModel
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configuration parameters
CONFIG = {
    "SAMPLE_RATE": 22050,
    "CHANNELS": 1,
    "BIT_DEPTH": "PCM_16",
    "MIN_SEGMENT_LEN": 2000,   # Minimum segment length in milliseconds (2 seconds)
    "MAX_SEGMENT_LEN": 15000,  # Maximum segment length in milliseconds (15 seconds - Hard Limit)
    "MIN_SILENCE_LEN": 500,    # Minimum silence length to detect for splitting (500ms)
    "SILENCE_THRESH": -40,     # Silence threshold in dBFS (-40dBFS)
    "KEEP_SILENCE": 300,       # Silence to keep at segment boundaries (300ms)
    "OUTPUT_DIR": "wavs",
    "METADATA_FILE": "metadata.csv"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioPipeline:
    def __init__(self, output_dir, metadata_file):
        self.output_dir = output_dir
        self.metadata_file = metadata_file
        self.metadata = []
        self.total_audio_duration = 0
        self.start_time = 0
        
        # Setup output directory
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                print(f"{Fore.RED}Warning: Could not clear wavs folder. {e}{Style.RESET_ALL}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Whisper model with GPU fallback to CPU
        print(f"{Fore.CYAN}Loading Whisper Large-v3 on GPU...{Style.RESET_ALL}")
        try:
            self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            print(f"{Fore.GREEN}Model loaded successfully on GPU!{Style.RESET_ALL}")
        except:
            print(f"{Fore.RED}GPU failed, falling back to CPU.{Style.RESET_ALL}")
            self.model = WhisperModel("medium", device="cpu", compute_type="int8")

    def download_audio(self, url, filename="raw_audio"):
        """Download audio from YouTube URL using yt-dlp"""
        print(f"{Fore.YELLOW}Downloading audio...{Style.RESET_ALL}")
        
        # Clean previous raw file if exists
        if os.path.exists(f"{filename}.wav"):
            os.remove(f"{filename}.wav")
            
        # Download using yt-dlp
        command = [
            "yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
            "--postprocessor-args", f"ffmpeg:-ac {CONFIG['CHANNELS']} -ar {CONFIG['SAMPLE_RATE']}",
            "-o", f"{filename}.%(ext)s", url
        ]
        subprocess.run(command, check=True)
        return f"{filename}.wav"

    def strip_silence(self, audio_segment, silence_thresh=-45, padding=50):
        """Trim leading and trailing silence from audio segment"""
        if len(audio_segment) < padding:
            return audio_segment
            
        # Trim start silence
        start_trim = 0
        while start_trim < len(audio_segment) and audio_segment[start_trim:start_trim+10].dBFS < silence_thresh:
            start_trim += 10
        
        # Trim end silence
        end_trim = len(audio_segment)
        while end_trim > start_trim and audio_segment[end_trim-10:end_trim].dBFS < silence_thresh:
            end_trim -= 10
            
        # Apply padding and return trimmed segment
        return audio_segment[max(0, start_trim-padding):min(len(audio_segment), end_trim+padding)]

    def smart_split_long_chunk(self, chunk):
        """Recursively split chunks exceeding maximum length at quietest points"""
        if len(chunk) <= CONFIG['MAX_SEGMENT_LEN']:
            return [chunk]
        
        # Search for split point between 10s and 15s
        search_start = 10000  # 10 seconds
        search_end = CONFIG['MAX_SEGMENT_LEN']  # 15 seconds
        
        # Safety check for chunks shorter than search_end
        if len(chunk) < search_end:
            search_end = len(chunk)

        # Analyze volume in the search window to find quietest point
        window = chunk[search_start:search_end]
        step = 100  # Check every 100ms
        min_loudness = 0
        split_point = search_end  # Default to hard cut at max length
        
        best_split_found = False
        
        # Find lowest dBFS (quietest point) in the window
        for i in range(0, len(window) - step, step):
            loudness = window[i:i+step].dBFS
            if loudness < min_loudness:
                min_loudness = loudness
                split_point = search_start + i
                best_split_found = True
        
        # If no suitable quiet point found, cut at 14.5s
        if not best_split_found or min_loudness > -10:
            split_point = 14500
        
        # Split chunk and recursively process second part
        first_part = chunk[:split_point]
        second_part = chunk[split_point:]
        
        return [first_part] + self.smart_split_long_chunk(second_part)

    def split_and_process(self, raw_audio_path):
        """Split audio into segments using silence detection and force splitting for long segments"""
        print(f"{Fore.YELLOW}Splitting audio (with force splitting for long segments)...{Style.RESET_ALL}")
        audio = AudioSegment.from_wav(raw_audio_path)
        
        # Initial split based on silence detection
        initial_chunks = split_on_silence(
            audio,
            min_silence_len=CONFIG['MIN_SILENCE_LEN'],
            silence_thresh=CONFIG['SILENCE_THRESH'],
            keep_silence=CONFIG['KEEP_SILENCE']
        )

        final_chunks = []
        current_chunk = AudioSegment.empty()

        for chunk in initial_chunks:
            # Clean silence from chunk boundaries
            chunk = self.strip_silence(chunk)
            
            # Skip very small chunks (likely noise)
            if len(chunk) < 500:
                continue

            # Check if adding chunk exceeds maximum length
            if len(current_chunk) + len(chunk) < CONFIG['MAX_SEGMENT_LEN']:
                current_chunk += chunk
            else:
                # Save current chunk if it meets minimum length
                if len(current_chunk) >= CONFIG['MIN_SEGMENT_LEN']:
                    final_chunks.append(current_chunk)
                
                # Handle new chunk - check if it's too long
                if len(chunk) > CONFIG['MAX_SEGMENT_LEN']:
                    # Force split the long chunk
                    split_pieces = self.smart_split_long_chunk(chunk)
                    # Add all but last piece to final list
                    final_chunks.extend(split_pieces[:-1])
                    # Last piece becomes new current chunk for potential merging
                    current_chunk = split_pieces[-1]
                else:
                    current_chunk = chunk
        
        # Handle final chunk
        if len(current_chunk) >= CONFIG['MIN_SEGMENT_LEN']:
            if len(current_chunk) > CONFIG['MAX_SEGMENT_LEN']:
                final_chunks.extend(self.smart_split_long_chunk(current_chunk))
            else:
                final_chunks.append(current_chunk)

        return final_chunks

    def clean_audio(self, audio_segment):
        """Apply noise reduction to audio segment"""
        samples = np.array(audio_segment.get_array_of_samples())
        # Apply moderate noise reduction to preserve voice quality
        try:
            return nr.reduce_noise(y=samples, sr=CONFIG['SAMPLE_RATE'], prop_decrease=0.6, stationary=True)
        except:
            return samples

    def run(self, youtube_url, speaker_name="speaker1"):
        """Main pipeline execution"""
        self.start_time = time.time()
        
        # Reset metadata list for new run
        self.metadata = []
        
        # Download and process audio
        raw_file = self.download_audio(youtube_url)
        chunks = self.split_and_process(raw_file)
        
        print(f"{Fore.CYAN}Processing {len(chunks)} segments...{Style.RESET_ALL}")

        # Process each chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Transcribing")):
            # Verify chunk doesn't exceed length limit (with 100ms tolerance)
            if len(chunk) > CONFIG['MAX_SEGMENT_LEN'] + 100:
                print(f"{Fore.RED}Skipping segment {i} (Length {len(chunk)}ms > 15s limit){Style.RESET_ALL}")
                continue
                
            self.total_audio_duration += len(chunk) / 1000.0
            
            # Clean and save audio
            clean_samples = self.clean_audio(chunk)
            filename = f"audio_{i:06d}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            sf.write(filepath, clean_samples, CONFIG['SAMPLE_RATE'], subtype=CONFIG['BIT_DEPTH'])
            
            # Transcribe using Whisper
            try:
                segments, _ = self.model.transcribe(filepath, language="bn")
                text = " ".join([segment.text for segment in segments]).strip()
                
                if text:
                    self.metadata.append(f"{filename}|{text}|{speaker_name}")
            except Exception as e:
                print(f"{Fore.RED}Transcription error for {filename}: {e}{Style.RESET_ALL}")

        # Save metadata file with BOM for Excel compatibility
        with open(self.metadata_file, "w", encoding="utf-8-sig") as f:
            for line in self.metadata:
                f.write(line + "\n")
        
        # Clean up raw audio file
        if os.path.exists(raw_file):
            try:
                os.remove(raw_file)
            except:
                pass
        
        # Print performance metrics
        total_time = time.time() - self.start_time
        rtf = total_time / self.total_audio_duration if self.total_audio_duration > 0 else 0
        
        print(f"\n{Fore.GREEN}Pipeline Complete!{Style.RESET_ALL}")
        print(f"Metadata saved to: {self.metadata_file}")
        print(f"{Fore.YELLOW}Performance Metrics:{Style.RESET_ALL}")
        print(f"   - Total Audio Processed: {self.total_audio_duration:.2f} sec")
        print(f"   - Real-Time Factor (RTF): {rtf:.2f}")

if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    if url:
        pipeline = AudioPipeline(CONFIG['OUTPUT_DIR'], CONFIG['METADATA_FILE'])
        pipeline.run(url, speaker_name="Habibur_Sample")