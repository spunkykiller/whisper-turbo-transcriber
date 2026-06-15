import os
import sys
import time
import queue
import ctypes
import threading
import subprocess
import webbrowser
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, abort


# ---------------------------------------------------------------------------
# CUDA DLL setup – must run before importing faster-whisper / ctranslate2
# ---------------------------------------------------------------------------
def _setup_cuda_dlls():
    for path in sys.path:
        if 'site-packages' not in path:
            continue
        nv = os.path.join(path, 'nvidia')
        if not os.path.isdir(nv):
            continue
        for root, dirs, _ in os.walk(nv):
            if 'bin' not in dirs:
                continue
            bin_dir = os.path.join(root, 'bin')
            if not os.path.isdir(bin_dir):
                continue
            try:
                os.add_dll_directory(bin_dir)
                os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
            except Exception:
                pass


def _preload_cuda_dlls():
    try:
        dll_specs = [
            ('cuda_runtime', 'cudart64_12.dll'),
            ('cublas', 'cublasLt64_12.dll'),
            ('cublas', 'cublas64_12.dll'),
            ('cudnn', 'cudnn64_9.dll'),
        ]
        for path in sys.path:
            if 'site-packages' not in path:
                continue
            nv = os.path.join(path, 'nvidia')
            if not os.path.isdir(nv):
                continue
            for pkg, dll_name in dll_specs:
                dll_path = os.path.join(nv, pkg, 'bin', dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except Exception:
                        pass
    except Exception:
        pass


_setup_cuda_dlls()
_preload_cuda_dlls()

import imageio_ffmpeg
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
OUTPUTS_DIR = BASE_DIR / 'Outputs'
UPLOADS_DIR = CODE_DIR / 'uploads'

ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv', '.flv', '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.wma'}
WHISPER_MODEL_NAME = 'large-v3-turbo'

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Clean stale uploads
for f in UPLOADS_DIR.iterdir():
    if f.is_file():
        f.unlink()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024  # 50 GB

# ---------------------------------------------------------------------------
# Job manager (thread-safe queue + state)
# ---------------------------------------------------------------------------
jobs_lock = threading.Lock()
jobs: dict[str, dict] = {}
job_queue: queue.Queue = queue.Queue()
_ffmpeg_path: str | None = None


def _get_ffmpeg() -> str:
    global _ffmpeg_path
    if _ffmpeg_path is None:
        try:
            _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            _ffmpeg_path = r"C:\Users\Manish\AppData\Local\Programs\Stremio\ffmpeg.exe"
    return _ffmpeg_path


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _safe_name(original: str) -> str:
    stem = Path(original).stem
    wav = OUTPUTS_DIR / f'{stem}.wav'
    txt = OUTPUTS_DIR / f'{stem}_transcript.txt'
    if not wav.exists() and not txt.exists():
        return stem
    return f'{stem}_{int(time.time())}'


def _worker(model: WhisperModel):
    ffmpeg = _get_ffmpeg()

    while True:
        item = job_queue.get()
        if item is None:
            job_queue.task_done()
            break

        original_name, safe_name, video_path = item
        out_wav = OUTPUTS_DIR / f'{safe_name}.wav'
        out_txt = OUTPUTS_DIR / f'{safe_name}_transcript.txt'

        try:
            # --- extract audio --------------------------------------------------
            with jobs_lock:
                jobs[original_name] = {'status': 'extracting', 'progress': 0, 'transcript_path': None, 'error': None}

            cmd = [
                ffmpeg, '-y',
                '-i', str(video_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                str(out_wav),
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=7200)
            if result.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
                raise RuntimeError('Audio extraction failed (no audio track or corrupted file)')

            # --- transcribe -----------------------------------------------------
            with jobs_lock:
                jobs[original_name] = {'status': 'transcribing', 'progress': 0, 'transcript_path': None, 'error': None}

            segments, info = model.transcribe(str(out_wav), beam_size=5)
            total = max(1, int(info.duration / 30))

            with open(out_txt, 'w', encoding='utf-8') as f:
                f.write(f'File: {original_name}\n')
                f.write(f'Language: {info.language} ({info.language_probability:.2%})\n')
                f.write(f'Duration: {info.duration:.2f}s\n\nTranscript:\n')

                for i, seg in enumerate(segments):
                    line = f'[{_fmt_time(seg.start)} -> {_fmt_time(seg.end)}] {seg.text}\n'
                    f.write(line)
                    f.flush()

                    if i % 5 == 0:
                        with jobs_lock:
                            pct = min(99, int((i + 1) / total * 100))
                            jobs[original_name] = {'status': 'transcribing', 'progress': pct, 'transcript_path': None, 'error': None}

            with jobs_lock:
                jobs[original_name] = {'status': 'done', 'progress': 100, 'transcript_path': str(out_txt), 'error': None}

        except Exception as exc:
            with jobs_lock:
                jobs[original_name] = {'status': 'error', 'progress': 0, 'transcript_path': None, 'error': str(exc)}
        finally:
            if video_path.exists():
                video_path.unlink()
            job_queue.task_done()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    uploaded = request.files.getlist('files')
    accepted = []

    for f in uploaded:
        name = Path(f.filename).name if f.filename else ''
        if not name or not _is_video(name):
            continue

        safe = _safe_name(name)
        dest = UPLOADS_DIR / f'{safe}{Path(name).suffix}'
        f.save(str(dest))

        with jobs_lock:
            jobs[name] = {'status': 'queued', 'progress': 0, 'transcript_path': None, 'error': None}

        job_queue.put((name, safe, dest))
        accepted.append({'filename': name, 'status': 'queued'})

    return jsonify({'files': accepted})


@app.route('/api/status')
def api_status():
    with jobs_lock:
        return jsonify(dict(jobs))


@app.route('/transcript/<path:filename>')
def serve_transcript(filename):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists():
        abort(404)
    return send_from_directory(str(OUTPUTS_DIR), safe, mimetype='text/plain; charset=utf-8')


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting Whisper Transcription Server...')
    print(f'Loading model {WHISPER_MODEL_NAME!r}...')

    model = None
    try:
        model = WhisperModel(WHISPER_MODEL_NAME, device='cuda', compute_type='float16')
        print('  [OK] Loaded on GPU (CUDA float16)')
    except Exception as exc:
        print(f'  [FAIL] GPU failed: {exc}')
        print('  -> Falling back to CPU (float32) - this will be slow.')
        try:
            model = WhisperModel(WHISPER_MODEL_NAME, device='cpu', compute_type='float32')
            print('  [OK] Loaded on CPU')
        except Exception as exc2:
            print(f'  [FAIL] CPU also failed: {exc2}')
            sys.exit(1)

    worker = threading.Thread(target=_worker, args=(model,), daemon=True)
    worker.start()

    # Find a free port
    port = 5000
    for _ in range(10):
        try:
            sock = __import__('socket').socket(__import__('socket').AF_INET, __import__('socket').SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            break
        except OSError:
            port += 1

    url = f'http://127.0.0.1:{port}'
    print(f'\n  Dashboard : {url}')
    print(f'  Outputs   : {OUTPUTS_DIR}')
    print('  Drop video files on the dashboard to transcribe them.\n')

    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, threaded=True, debug=False)
