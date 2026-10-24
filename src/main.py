"""
YouTube Comments Sentiment Analysis
Fetches up to 2,000 top-level comments using YouTube Data API v3,
classifies sentiment with a multilingual Hugging Face model, and writes
CSV, JSON summary, and charts.

Run from the project root:
    python src/main.py
"""
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
ENV_FILE = ROOT / ".env"
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
DEFAULT_VIDEO_URL = "https://youtu.be/kAgAQ24_Jso"
TARGET_COMMENTS = 2000
API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def get_video_id(url: str) -> str:
    """Accept youtube.com/watch?v=..., youtu.be/..., and shorts URLs."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().replace("www.", "")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            video_id = parts[1] if len(parts) > 1 and parts[0] in {"shorts", "embed", "live"} else ""
    else:
        video_id = ""
    if not video_id:
        raise ValueError("Could not read a YouTube video ID from VIDEO_URL in .env")
    return video_id


def fetch_comments(api_key: str, video_id: str, target: int = TARGET_COMMENTS) -> pd.DataFrame:
    """Fetch top-level published comments in time order; API pages contain at most 100."""
    rows, page_token = [], None
    while len(rows) < target:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, target - len(rows)),
            "order": "time",
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        response = requests.get(API_URL, params=params, timeout=30)
        if response.status_code != 200:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"YouTube API request failed ({response.status_code}): {detail}\n"
                "Check the API key, YouTube Data API v3 enablement, video ID, and whether comments are enabled."
            )
        payload = response.json()
        items = payload.get("items", [])
        if not items:
            break
        for item in items:
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snippet.get("textDisplay", "").strip()
            if not text:
                continue
            rows.append({
                "comment_id": item.get("snippet", {}).get("topLevelComment", {}).get("id", ""),
                "published_at": snippet.get("publishedAt", ""),
                "like_count": snippet.get("likeCount", 0),
                "comment_text": text,
            })
            if len(rows) >= target:
                break
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
        print(f"Fetched {len(rows)} comments...")
    if not rows:
        raise RuntimeError("No comments were returned. The video may have comments disabled or none available.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["comment_id"]).head(target)
    return df


def classify_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Classify comments with a multilingual three-class model."""
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,  # CPU works on a normal VS Code setup; change to 0 if CUDA is configured.
    )
    texts = df["comment_text"].fillna("").astype(str).tolist()
    # Pipeline truncation prevents very long comments from exceeding the model's input limit.
    predictions = classifier(texts, batch_size=16, truncation=True, max_length=512)
    df = df.copy()
    df["sentiment"] = [str(p["label"]).lower() for p in predictions]
    df["sentiment_score"] = [round(float(p["score"]), 4) for p in predictions]
    # Normalize possible model label formatting.
    df["sentiment"] = df["sentiment"].str.replace("label_", "", regex=False)
    return df


def save_outputs(df: pd.DataFrame, video_id: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "sentiment_results.csv"
    df.to_csv(results_path, index=False, encoding="utf-8-sig")

    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    percentages = (counts / len(df) * 100).round(2)
    summary = {
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "comments_analyzed": int(len(df)),
        "model": MODEL_NAME,
        "sentiment_counts": {str(k): int(v) for k, v in counts.items()},
        "sentiment_percentages": {str(k): float(v) for k, v in percentages.items()},
        "note": "Model predictions are estimates, not human-verified ground truth. Mixed-language Roman Urdu may be less reliable."
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("YouTube Comment Sentiment Counts")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of comments")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sentiment_bar_chart.png", dpi=160)
    plt.close()

    nonzero = counts[counts > 0]
    plt.figure(figsize=(6, 6))
    nonzero.plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of Comment Sentiments")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sentiment_pie_chart.png", dpi=160)
    plt.close()

    print("\nAnalysis complete.")
    print(f"Comments analyzed: {len(df)}")
    print(f"Results CSV: {results_path}")
    print(f"Summary JSON: {OUT_DIR / 'summary.json'}")
    print(f"Charts: {OUT_DIR / 'sentiment_bar_chart.png'}, {OUT_DIR / 'sentiment_pie_chart.png'}")
    print("\nSentiment counts:")
    print(counts.to_string())
    print("\nSentiment percentages:")
    print(percentages.astype(str).add("%").to_string())


def main():
    load_dotenv(ENV_FILE)
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    video_url = os.getenv("VIDEO_URL", DEFAULT_VIDEO_URL).strip()
    target = int(os.getenv("TARGET_COMMENTS", str(TARGET_COMMENTS)))
    if not api_key or api_key == "paste_your_api_key_here":
        sys.exit(
            "Missing YOUTUBE_API_KEY. Copy .env.example to .env and add your Google Cloud API key."
        )
    if target < 1:
        sys.exit("TARGET_COMMENTS must be at least 1.")
    video_id = get_video_id(video_url)
    print(f"Video ID: {video_id}")
    print(f"Target: up to {target} top-level comments")
    comments = fetch_comments(api_key, video_id, target)
    print(f"Collected {len(comments)} unique comments. Loading sentiment model (first run downloads it)...")
    results = classify_sentiment(comments)
    save_outputs(results, video_id)


if __name__ == "__main__":
    main()
