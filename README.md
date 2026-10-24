# YouTube Comments Sentiment Analysis

A Python project that collects up to 2,000 top-level comments from one YouTube video, classifies each comment as **positive**, **neutral**, or **negative**, and generates CSV/JSON outputs plus bar and pie charts.

**Target video:** https://youtu.be/kAgAQ24_Jso  
**Environment:** VS Code  
**Model:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

> Important: results are not included as precomputed data because the project must be run with a valid API key and the live comments. The code writes real results after it runs. Do not claim the target of 2,000 was reached unless the output confirms it.

## Project structure

```text
youtube-sentiment-analysis/
├── src/
│   └── main.py
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── reports/
│   ├── project_report.md
│   ├── project_report.docx
│   └── project_report.pdf
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Requirements

- Python 3.10 or 3.11 recommended
- Internet access for YouTube API requests and the first model download
- Google account to create a YouTube Data API v3 key
- Around 1–2 GB of free disk space recommended for the model and Python packages

## 2. Set up YouTube Data API v3

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. Open **APIs & Services → Library**, search for **YouTube Data API v3**, and enable it.
4. Open **APIs & Services → Credentials → Create credentials → API key**.
5. Restrict the key where practical, and do not publish it.
6. In this project folder, copy `.env.example` to a new file named `.env`.
7. Paste your key after `YOUTUBE_API_KEY=`. Keep `.env` private; it is excluded by `.gitignore`.

The YouTube API may return fewer than 2,000 comments if fewer are available, comments are disabled, the API key/quota is invalid, or access is restricted.

## 3. Run in VS Code

Open this folder in VS Code. In the integrated terminal:

### Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your API key, then run:
```powershell
python src/main.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your API key, then run:
```bash
python src/main.py
```

On the first run, Hugging Face downloads the sentiment model, so it may take a few minutes. CPU inference is supported; a GPU is optional.

## 4. Outputs

After a successful run, the `outputs/` directory contains:

- `sentiment_results.csv` — each comment, publication date, like count, predicted sentiment, and model confidence score
- `summary.json` — counts and percentages by sentiment, model name, and video URL
- `sentiment_bar_chart.png` — sentiment counts
- `sentiment_pie_chart.png` — sentiment proportions

Use the actual `summary.json` and charts to complete the results section in the report.

## 5. Methodology

1. Parse the video ID from the URL.
2. Request published top-level comment threads using YouTube Data API v3 (`maxResults` up to 100 per page).
3. Follow `nextPageToken` until the target is reached or no further pages are available.
4. Remove duplicate comment IDs.
5. Apply a multilingual pretrained transformer classifier.
6. Export predictions and visualizations.

## 6. Limitations

- This is automated classification, not ground-truth annotation; confidence scores are not accuracy scores.
- The model card lists 12 languages, but **Urdu is not among the listed languages**. English and some supported languages may work better than Urdu/Roman Urdu, code-switching, sarcasm, slang, and emoji-heavy comments. Interpret mixed-language results cautiously and manually review a sample.
- The API collects top-level comments only; replies are not included.
- The API's returned order is `time`, so this is not a random or representative sample of all audience opinions.
- The video may have fewer than 2,000 retrievable comments.

## 7. GitHub upload

1. Create a new empty repository on GitHub.
2. Make sure `.env` is not staged or uploaded. Never commit API keys.
3. Commit the project files and push them.
4. Replace the repository placeholder in `reports/project_report.md` and the report files with your actual GitHub URL.
5. Do not upload the comments CSV unless you have checked platform terms and privacy considerations; the code and charts may be sufficient for submission.

## References

- YouTube Data API v3, `commentThreads.list`: https://developers.google.com/youtube/v3/docs/commentThreads/list
- Model card: https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student

## Troubleshooting

- **`commentsDisabled`**: comments are disabled for the video; this project cannot collect them through the API.
- **`quotaExceeded`**: check the Google Cloud project quota and try again when available.
- **`API key not valid`**: confirm the key, enabled API, and key restrictions.
- **No 2,000 comments**: the script processes up to the target, not a guarantee of exactly 2,000.
- **Model download or memory issues**: use Python 3.10/3.11 and ensure sufficient free disk/RAM. If needed, lower the pipeline batch size from 16 in `src/main.py`.
