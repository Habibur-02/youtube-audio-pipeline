# youtube-audio-pipeline
this repo contains youtube audio processing pipeline using whisper large v3 model and desinged with companies requirements

# YouTube Audio Processing Pipeline 🎧

A high-performance Python pipeline designed to download, process, denoise, and transcribe YouTube audio for dataset creation. Built with **Faster-Whisper (Large-v3)** and **GPU Acceleration** for state-of-the-art accuracy and speed.

## 🚀 Key Features
* **Smart Downloading:** Extracts high-quality audio (WAV, 22050Hz, Mono) directly using `yt-dlp` and `FFmpeg`.
* **Precision Splitting:** Segmenting based on silence detection with strictly enforced constraints (2s - 15s duration).
* **Advanced Denoising:** Utilizes stationary noise reduction to remove background noise while preserving natural speech characteristics (non-robotic).
* **SOTA Transcription:** Powered by **Whisper Large-v3** running on GPU (CUDA) for high-accuracy Bengali speech recognition.
* **Standardized Output:** Generates LJSpeech-style metadata (`filename|text|speaker`).

## 🛠 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Habibur-02/youtube-audio-pipeline.git](https://github.com/Habibur-02/youtube-audio-pipeline.git)
    cd youtube-audio-pipeline
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.8+ and FFmpeg installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **FFmpeg Setup**
    Ensure `ffmpeg` is added to your system PATH or placed in the project directory.

## 🏃 Usage

Run the pipeline using the simple command-line interface:

```bash
python pipeline.py

```

When prompted, enter the YouTube URL. The pipeline will handle the rest!

# ⚙️ Technical Specifications
```
Parameter	Value
Sample Rate	22050 Hz
Bit Depth	16-bit PCM
Channels	Mono
Silence Threshold	-38 dBFS
Min Silence Length	400 ms


```