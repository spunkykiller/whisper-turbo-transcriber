# /// script
# dependencies = [
#   "faster-whisper",
#   "imageio-ffmpeg",
#   "nvidia-cublas-cu12",
#   "nvidia-cudnn-cu12",
#   "nvidia-cuda-runtime-cu12",
#   "nvidia-cuda-nvrtc-cu12",
# ]
# ///

import os
import sys

# Force stdout/stderr to use UTF-8 to prevent encoding errors on Windows when printing non-ASCII characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import subprocess
import time
import ctypes

def setup_cuda_dlls():
    # Find site-packages
    for path in sys.path:
        if 'site-packages' in path:
            nvidia_path = os.path.join(path, 'nvidia')
            if os.path.isdir(nvidia_path):
                # Add bin folders to DLL search path AND system PATH
                for root, dirs, files in os.walk(nvidia_path):
                    if 'bin' in dirs:
                        bin_dir = os.path.join(root, 'bin')
                        if os.path.isdir(bin_dir):
                            try:
                                os.add_dll_directory(bin_dir)
                                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                                print(f"Added to DLL search path & PATH: {bin_dir}")
                            except Exception as e:
                                print(f"Error adding {bin_dir} to path: {e}")

setup_cuda_dlls()

# Explicitly load the DLLs via ctypes to force them into process memory
try:
    print("Pre-loading CUDA and cuDNN DLLs via ctypes...")
    for path in sys.path:
        if 'site-packages' in path:
            nvidia_path = os.path.join(path, 'nvidia')
            if os.path.isdir(nvidia_path):
                # Order of loading matters because of dependencies:
                # 1. cuda_runtime
                # 2. cublas
                # 3. cudnn
                dlls_to_load = [
                    ("cuda_runtime", "cudart64_12.dll"),
                    ("cublas", "cublasLt64_12.dll"),
                    ("cublas", "cublas64_12.dll"),
                    ("cudnn", "cudnn64_9.dll"),
                ]
                for pkg, dll_name in dlls_to_load:
                    dll_path = os.path.join(nvidia_path, pkg, "bin", dll_name)
                    if os.path.exists(dll_path):
                        try:
                            ctypes.CDLL(dll_path)
                            print(f"Successfully pre-loaded {dll_name}")
                        except Exception as e:
                            print(f"Failed to load {dll_name} from {dll_path}: {e}")
                    else:
                        # Fallback recursive search
                        for root, dirs, files in os.walk(os.path.join(nvidia_path, pkg)):
                            if dll_name in files:
                                full_path = os.path.join(root, dll_name)
                                try:
                                    ctypes.CDLL(full_path)
                                    print(f"Successfully pre-loaded {dll_name} from {full_path}")
                                    break
                                except Exception as e:
                                    print(f"Failed to load {dll_name} from {full_path}: {e}")
except Exception as e:
    print(f"Ctypes pre-loading failed: {e}")


import imageio_ffmpeg
from faster_whisper import WhisperModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(SCRIPT_DIR, "Outputs")
VIDEO_PATH = os.path.join(SCRIPT_DIR, "test_video.mp4")

if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]

base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
AUDIO_PATH = os.path.join(TRANSCRIPT_DIR, f"{base_name}.wav")
TXT_OUTPUT_PATH = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.txt")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

def main():
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
        
    print(f"--- Audio Extraction ---")
    if os.path.exists(AUDIO_PATH):
        print(f"Audio file already exists at: {AUDIO_PATH}. Skipping extraction.")
    else:
        if not os.path.exists(VIDEO_PATH):
            print(f"Error: Video file not found at {VIDEO_PATH}")
            sys.exit(1)
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Extracting audio using ffmpeg: {ffmpeg_exe}...")
        # Extract audio track to 16kHz mono wav for optimal Whisper performance
        cmd = [
            ffmpeg_exe, "-y",
            "-i", VIDEO_PATH,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            AUDIO_PATH
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("FFmpeg extraction failed!")
            print(result.stderr.decode('utf-8', errors='replace'))
            sys.exit(1)
        print(f"Audio successfully extracted to {AUDIO_PATH} in {time.time() - start_time:.2f} seconds.")

    print(f"\n--- Loading Whisper Model ---")
    model_name = "large-v3-turbo"
    print(f"Loading Whisper model '{model_name}' on GPU (CUDA) with float16 precision...")
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        print("Model loaded successfully on GPU.")
    except Exception as e:
        print(f"GPU loading failed: {e}")
        print("Falling back to CPU with float32 precision...")
        model = WhisperModel(model_name, device="cpu", compute_type="float32")
        print("Model loaded successfully on CPU.")

    print(f"\n--- Transcribing Audio ---")
    start_time = time.time()
    
    # Run transcription (beam_size=5 is standard for Whisper)
    segments, info = model.transcribe(AUDIO_PATH, beam_size=5)
    
    print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    print(f"Total audio duration: {info.duration:.2f} seconds")
    print(f"Transcribing... (writing results to {TXT_OUTPUT_PATH})")
    
    with open(TXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        # Write header
        header = f"File: {VIDEO_PATH}\nLanguage: {info.language} ({info.language_probability:.2%})\nDuration: {info.duration:.2f}s\n\nTranscript:\n"
        f.write(header)
        print(header, end="")
        
        for segment in segments:
            start_str = format_time(segment.start)
            end_str = format_time(segment.end)
            line = f"[{start_str} -> {end_str}] {segment.text}\n"
            f.write(line)
            # Flush to file so we don't lose data if interrupted
            f.flush()
            # Print to stdout in real-time
            print(line, end="")
            
    print(f"\nTranscription completed in {time.time() - start_time:.2f} seconds.")
    print(f"Transcript saved to: {TXT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
