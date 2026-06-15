# Whisper Turbo Transcriber 🎙️⚡

Whisper Turbo Transcriber is a high-speed, GPU-accelerated local web dashboard and command-line interface (CLI) for transcribing video and audio files. It uses OpenAI's highly optimized **Whisper Large-v3-Turbo** model via the `faster-whisper` library.

By leveraging OpenAI's turbo variant (pruned to 4 decoder layers instead of 32), the application achieves up to **8x faster transcription speeds** on GPUs compared to the standard Whisper Large-v3 model, while retaining comparable multilingual accuracy.

---

## 🌟 Key Features

*   **OpenAI Whisper Large-v3-Turbo**: Performs high-accuracy speech-to-text inference locally.
*   **GPU-Accelerated (CUDA)**: Runs with `float16` precision on NVIDIA graphics cards. Automatically falls back to CPU if no compatible GPU is available.
*   **Dual Usage Modes**: 
    *   **Web Dashboard**: An interactive, dark-themed Flask dashboard featuring drag-and-drop uploads, queue management, progress bars, and direct transcript views.
    *   **CLI Mode**: A standalone command-line script for batch processing specific files directly.
*   **Automatic Audio Processing**: Automatically extracts audio tracks from input videos and normalizes them to a standard 16 kHz mono PCM WAV format for optimal transcription accuracy.
*   **Structured Outputs**: Generates timestamped `.txt` transcripts alongside the extracted `.wav` audio.

---

## 📁 Repository Directory Structure

```text
whisper-turbo-transcriber/
  ├── run.bat                  # Primary launcher script (starts server & opens browser)
  ├── setup_env.bat            # One-click environment setup script (creates local venv & pip installs)
  ├── .gitignore               # Excludes python virtual environments and video binary files
  ├── README.md                # Main documentation guide
  ├── transcribe.py            # Standalone CLI transcription script
  ├── Outputs/                 # Output folder for WAV extracts and transcripts
  │     └── .gitkeep
  └── Code/                    # Backend server and dashboard assets
        ├── server.py          # Flask backend server and Whisper worker thread
        ├── requirements.txt   # Python dependency manifest
        ├── static/            # Static files for the web interface
        │     └── style.css    # Dark mode layout styling
        └── templates/         # HTML template files
              └── index.html   # Main dashboard layout
```

---

## 🛠️ Click-to-Install Setup & Installation

Whisper Turbo Transcriber is designed to support a simple, automated environment setup.

### Prerequisites
1.  **Python 3.10+**: Download and install from [python.org](https://python.org/). Ensure you check **"Add Python to PATH"** during installation.
2.  **NVIDIA GPU**: A dedicated NVIDIA graphics card with updated graphics drivers is required for GPU acceleration (requires CUDA 12 support).
3.  **Space**: Ensure you have ~1.6 GB of free disk space for the Whisper Large-v3-Turbo model cache (downloaded on the first run).

### One-Click Setup
1.  Double-click [setup_env.bat](file:///C:/Users/Manish/Desktop/development/whisper-turbo-transcriber/setup_env.bat) in the root directory.
2.  The script will automatically:
    *   Verify Python is in your PATH.
    *   Create a local virtual environment (`Code\venv`).
    *   Upgrade `pip` and install all required python dependencies, including `faster-whisper`, `imageio-ffmpeg`, and NVIDIA CUDA/cuDNN packages.

---

## 🚀 How to Run

### Web Dashboard Mode
1.  Double-click [run.bat](file:///C:/Users/Manish/Desktop/development/whisper-turbo-transcriber/run.bat) in the root directory.
2.  The dashboard will automatically open in your default browser at `http://127.0.0.1:5000` (it will auto-bind to another port if 5000 is occupied).
3.  Drag and drop your audio or video files directly onto the drop zone to queue them.
4.  Once processed, click the file name in the table to view the raw transcript inside the browser.
5.  All outputs are stored locally in the `Outputs/` directory.

### Command Line Mode (Alternative)
For batch-scripting or simple CLI execution:
1.  Open PowerShell in the project directory.
2.  Run the standalone transcription script:
    ```powershell
    Code\venv\Scripts\python.exe transcribe.py
    ```
    *Note: Edit `transcribe.py` directly to adjust the `VIDEO_PATH` variable to point to any video or audio file on your machine.*

---

## ⚙️ Configuration & Customization

Key parameters can be customized directly in the code files:

*   **Model Selection**: If you need to fallback to lightweight models (e.g. `base.en`, `tiny`), change the `WHISPER_MODEL_NAME` variable inside [Code/server.py](file:///C:/Users/Manish/Desktop/development/whisper-turbo-transcriber/Code/server.py) and `transcribe.py`.
*   **Port Selection**: The web dashboard is configured to start on port `5000` by default. If that port is busy, the Flask app automatically increments and binds to the next free port.

---

## 📜 Technical Details

### CUDA DLL Pre-Loading
PyTorch and CTranslate2 require access to native NVIDIA DLLs (cuBLAS, cuDNN, CUDA RT) to accelerate inference on Windows. The system handles this by:
1.  Locating the local `nvidia/` subpackages in your environment's `site-packages` directory.
2.  Adding these binary directories to the system PATH and DLL search path via `os.add_dll_directory`.
3.  Using `ctypes.CDLL` to preload the required DLLs (`cudart64_12.dll`, `cublasLt64_12.dll`, `cublas64_12.dll`, `cudnn64_9.dll`) directly into process memory.
