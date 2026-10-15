# Lecture Note Generator

## Overview
This project provides a **simple web interface** to upload video lectures and automatically generate **structured markdown notes** using:
- **OpenAI Whisper** for speech‑to‑text transcription.
- **Tesseract OCR** to extract any slide text from video frames.
- **Flask** as a lightweight Python backend.
- A **premium glassmorphic UI** with vibrant colors and smooth micro‑animations.

The final notes combine timestamps, transcribed speech, and OCR‑extracted slide text, all rendered as markdown (and downloadable as a `.md` file).

---
## Project Structure
```
video_notes_project/
├─ app.py                     # Flask backend
├─ requirements.txt           # Python dependencies
├─ README.md                  # This file
├─ .gitignore                 # Ignored files/folders
├─ static/
│   └─ styles.css            # Premium CSS (glassmorphism)
├─ templates/
│   ├─ index.html            # Upload page
│   └─ result.html           # Result page
├─ uploads/                   # Temporary uploaded videos & intermediate files (auto‑created)
└─ notes/                     # Generated markdown notes (auto‑created)
```

---
## Prerequisites
1. **Python 3.9+**
2. **ffmpeg** – for audio extraction and frame capture.
   ```bash
   # macOS (Homebrew)
   brew install ffmpeg
   ```
3. **Tesseract OCR** – for slide text extraction.
   ```bash
   # macOS (Homebrew)
   brew install tesseract
   ```
4. (Optional) **GPU** – Whisper can use CUDA if you have a compatible GPU. Install `torch` with CUDA support if desired.

---
## Setup & Run
```bash
# 1️⃣ Clone the repository (or copy the folder you have)
cd /Users/chinmayingle/Desktop/NLP/video_notes_project

# 2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# source venv\Scripts\activate   # On Windows

# 3️⃣ Install Python dependencies
pip install -r requirements.txt

# 4️⃣ Start the Flask app
python app.py
```
The server will start on **http://localhost:5001**. Open this URL in your browser, upload a lecture video, and wait for the notes to be generated.

---
## How It Works
1. **Upload video** – saved to `uploads/` with a unique identifier.
2. **Audio extraction** – `ffmpeg` extracts an MP3 file.
3. **Whisper transcription** – the audio is fed to Whisper (base model by default). Change to `large` for higher quality.
4. **Frame extraction** – every 5 seconds a frame is saved.
5. **OCR** – each frame is run through Tesseract; extracted text is linked to the nearest transcription timestamp.
6. **Combine** – timestamps, transcription, and OCR text are merged into markdown and saved under `notes/`.
7. **Result page** – renders the markdown as HTML and offers a download link.

---
## Customisation
- **Model size** – edit `model = whisper.load_model('base')` in `app.py` to `large`, `medium`, etc.
- **OCR interval** – change the `interval_seconds` argument in `extract_frames` (default 5 s).
- **Styling** – modify `static/styles.css` for colors, fonts, or glass‑morphism intensity.

---
## Known Limitations
- The current implementation keeps intermediate audio, frames, and notes on disk for simplicity. They are stored in `uploads/` and `notes/`. You can clean them manually or add a background task to purge old files.
- Accuracy depends on the Whisper model and the quality of the video/audio.
- OCR works best on clear, high‑contrast slides.

---
## License
This sample project is provided under the MIT License. Feel free to adapt and extend it for your own workflows.
