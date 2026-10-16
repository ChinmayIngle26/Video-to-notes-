import shutil
import subprocess
import uuid
from pathlib import Path

# Detect ffmpeg – installed via `brew install ffmpeg` on macOS
FFMPEG = (
    shutil.which('ffmpeg')
    or ('/opt/homebrew/bin/ffmpeg' if Path('/opt/homebrew/bin/ffmpeg').exists() else None)
    or ('/usr/local/bin/ffmpeg' if Path('/usr/local/bin/ffmpeg').exists() else None)
)
if not FFMPEG:
    raise RuntimeError("ffmpeg not found. Install it with: brew install ffmpeg")

import numpy as np  # Ensure NumPy is available for torch
import whisper
import markdown2
from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import pytesseract
from pytesseract import image_to_string
from PIL import Image
from werkzeug.utils import secure_filename

# Detect tesseract – installed via `brew install tesseract` on macOS
_TESS = (
    shutil.which('tesseract')
    or '/opt/homebrew/bin/tesseract'
    or '/usr/local/bin/tesseract'
)
pytesseract.pytesseract.tesseract_cmd = _TESS

FRAME_INTERVAL = 5.0  # seconds between extracted frames

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['NOTE_FOLDER'] = Path(__file__).parent / 'notes'
app.secret_key = 'replace-with-a-secure-key'

# Ensure folders exist
for folder in (app.config['UPLOAD_FOLDER'], app.config['NOTE_FOLDER']):
    folder.mkdir(parents=True, exist_ok=True)

# Load Whisper model once (large model may take time; adjust as needed)
model = whisper.load_model('base')  # change to 'large' for higher quality


def extract_audio(video_path: Path, audio_path: Path):
    """Extract audio (mp3) from video using ffmpeg."""
    cmd = [
        FFMPEG, '-y', '-i', str(video_path),
        '-vn', '-q:a', '0', '-map', 'a', str(audio_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_audio(audio_path: Path):
    """Run Whisper transcription and return list of segments.
    Each segment dict contains 'start', 'end', 'text'."""
    result = model.transcribe(str(audio_path))
    return result.get('segments', [])


def extract_frames(video_path: Path, frames_dir: Path, interval_seconds: float = FRAME_INTERVAL):
    """Extract frames every `interval_seconds` using ffmpeg.
    Returns list of frame file paths ordered by time."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg treats the whole output path as a printf-style pattern, so a literal
    # '%' anywhere in the path (e.g. a folder named "NLP 50%") breaks it.
    # Run inside frames_dir and use a bare relative filename pattern instead.
    cmd = [
        FFMPEG, '-y', '-i', str(video_path.resolve()),
        '-vf', f'fps=1/{interval_seconds}',
        '-q:v', '2',
        'frame_%05d.jpg'
    ]
    subprocess.run(
        cmd, check=True, cwd=str(frames_dir),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    return sorted(frames_dir.glob('frame_*.jpg'))


def ocr_frames(frame_paths, interval_seconds: float = FRAME_INTERVAL):
    """Run OCR on a list of frame image paths.
    Returns dict mapping frame timestamp (seconds) -> extracted text.
    The timestamp is inferred from filename ordering (ffmpeg names sequentially)."""
    ocr_results = {}
    for i, path in enumerate(frame_paths):
        timestamp = i * interval_seconds
        with Image.open(path) as img:
            text = image_to_string(img)
        ocr_results[timestamp] = text.strip()
    return ocr_results


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def combine_notes(segments, ocr_results):
    """Merge transcription segments with nearest OCR text.
    Returns markdown string with timestamps and combined content."""
    lines = []
    for seg in segments:
        start = seg['start']
        text = seg['text'].strip()

        # Find nearest OCR timestamp (guard against no frames extracted)
        ocr_text = ''
        if ocr_results:
            nearest_ts = min(ocr_results.keys(), key=lambda t: abs(t - start))
            ocr_text = ocr_results[nearest_ts]

        lines.append(f"### {format_timestamp(start)}\n")
        lines.append(f"**Transcription:** {text}\n\n")
        if ocr_text:
            lines.append(f"**Slide OCR:** {ocr_text}\n\n")
    return "\n".join(lines)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        unique_id = uuid.uuid4().hex
        safe_name = secure_filename(file.filename) or 'video'
        video_path = app.config['UPLOAD_FOLDER'] / f"{unique_id}_{safe_name}"
        file.save(video_path)

        try:
            # 1. Extract audio
            audio_path = app.config['UPLOAD_FOLDER'] / f"{unique_id}.mp3"
            extract_audio(video_path, audio_path)

            # 2. Transcribe audio
            segments = transcribe_audio(audio_path)

            # 3. Extract frames for OCR (5-second interval)
            frames_dir = app.config['UPLOAD_FOLDER'] / f"frames_{unique_id}"
            frame_paths = extract_frames(video_path, frames_dir, interval_seconds=FRAME_INTERVAL)

            # 4. OCR on frames
            ocr_results = ocr_frames(frame_paths)

            # 5. Combine into markdown notes
            notes_md = combine_notes(segments, ocr_results)
            notes_path = app.config['NOTE_FOLDER'] / f"{unique_id}.md"
            notes_path.write_text(notes_md, encoding='utf-8')

            notes_html = markdown2.markdown(notes_md)
            return render_template(
                'result.html',
                notes_html=notes_html,
                download_url=url_for('download_note', note_id=unique_id),
            )
        except subprocess.CalledProcessError as e:
            flash(f'Processing error: {e}')
            return redirect(request.url)
        except Exception as e:
            flash(f'Unexpected error: {e}')
            return redirect(request.url)
    return render_template('index.html')


@app.route('/download/<note_id>')
def download_note(note_id):
    note_path = app.config['NOTE_FOLDER'] / f"{note_id}.md"
    if not note_path.exists():
        flash('Note not found')
        return redirect(url_for('index'))
    return send_file(str(note_path), as_attachment=True)


if __name__ == '__main__':
    # Run in development mode – for production use a WSGI server.
    app.run(host='0.0.0.0', port=5001, debug=True)