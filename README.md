# Incredible India: Monuments & Stories

Discover Indian heritage one landmark at a time. Upload a photo to identify a
monument, explore a visitor-friendly overview of its history and architecture,
and ask an interactive multilingual guide follow-up questions by text or voice.

## Features

- Classifies photos across 24 Indian monuments with a trained CNN.
- Shows the uploaded image, prediction confidence, and other possible matches.
- Generates an AI-written landmark overview and answers visitor questions using
  the monument prediction as context.
- Supports typed questions and recorded questions transcribed with Groq Whisper.
- Offers answers in English and the 22 scheduled Indian languages.
- Reads answers aloud using the speech voices available in the visitor's browser
  or device. Voice availability varies by device and language.
- Includes an animated guide with selectable actions and outfits.
- Accepts image uploads up to 20 MB.

## Project contents

```text
.
├── app.py
├── class_names.json
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml
└── models/
    └── indian_monuments_cnn.keras
```

The included model and `class_names.json` must come from the same training run.

## Run locally on Windows

Python 3.12 is recommended for compatibility with the TensorFlow dependency.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
Copy-Item .env.example .env
```

Add your Groq API key to `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Then start the app:

```powershell
python -m streamlit run .\app.py --server.port 8502 --server.maxUploadSize 20
```

Open [http://localhost:8502](http://localhost:8502).

The monument classifier works with the included model. Groq-powered landmark
stories, question answering, and voice transcription require a valid Groq API
key. Never publish your real `.env` file or API key.

## Model details

The CNN accepts RGB images resized to `160 × 160` pixels and predicts one of 24
class labels. The saved model contains its own `Rescaling(1./255)` layer, so the
app passes pixel values in the `0–255` range without scaling them a second time.

## Notes

AI-generated heritage information may be incomplete or simplified. Verify
important historical details with trusted sources. Speech synthesis depends on
browser and device support for the selected language.
