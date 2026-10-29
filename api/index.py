import os
import re
import tempfile
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

app = FastAPI(title="YouTube Transcript API")

PROXY_URL = os.getenv("PROXY_URL")  # optional, e.g. http://user:pass@host:port
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_AUDIO_BYTES = 24 * 1024 * 1024  # Whisper endpoints reject files > 25MB


class TranscriptRequest(BaseModel):
    url: str
    languages: Optional[list[str]] = None


def extract_video_id(url: str) -> str:
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    parsed = urlparse(url)
    host = (parsed.hostname or "").replace("www.", "").replace("m.", "")
    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
        else:
            m = re.match(r"^/(?:shorts|embed|live|v)/([\w-]{11})", parsed.path)
            vid = m.group(1) if m else ""
    else:
        vid = ""
    if not re.fullmatch(r"[\w-]{11}", vid):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return vid


def fetch_captions(video_id: str, languages: list[str]) -> dict:
    proxy = (
        GenericProxyConfig(http_url=PROXY_URL, https_url=PROXY_URL)
        if PROXY_URL
        else None
    )
    api = YouTubeTranscriptApi(proxy_config=proxy)
    fetched = api.fetch(video_id, languages=languages)
    segments = fetched.to_raw_data()
    return {
        "language": fetched.language_code,
        "is_generated": fetched.is_generated,
        "text": " ".join(s["text"].replace("\n", " ") for s in segments),
        "segments": segments,
    }


def transcribe_audio(video_id: str) -> dict:
    if GROQ_API_KEY:
        endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"
        key, model = GROQ_API_KEY, "whisper-large-v3-turbo"
    elif OPENAI_API_KEY:
        endpoint = "https://api.openai.com/v1/audio/transcriptions"
        key, model = OPENAI_API_KEY, "whisper-1"
    else:
        raise RuntimeError("Set GROQ_API_KEY or OPENAI_API_KEY for audio fallback")

    with tempfile.TemporaryDirectory(dir="/tmp" if os.path.isdir("/tmp") else None) as tmp:
        opts = {
            # smallest audio-only stream; m4a/webm need no ffmpeg
            "format": "worstaudio[ext=m4a]/worstaudio",
            "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "max_filesize": MAX_AUDIO_BYTES,
        }
        if PROXY_URL:
            opts["proxy"] = PROXY_URL
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        files = os.listdir(tmp)
        if not files:
            raise RuntimeError("Audio too large (>24MB) or could not be downloaded")
        path = os.path.join(tmp, files[0])
        if os.path.getsize(path) > MAX_AUDIO_BYTES:
            raise RuntimeError("Audio too large (>24MB) for transcription")

        with open(path, "rb") as f:
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (files[0], f)},
                data={"model": model, "response_format": "verbose_json"},
                timeout=240,
            )
        if not resp.ok:
            raise RuntimeError(f"Transcription failed: {resp.status_code} {resp.text[:300]}")
        body = resp.json()

    segments = [
        {"text": s["text"].strip(), "start": s["start"], "duration": s["end"] - s["start"]}
        for s in body.get("segments", [])
    ]
    return {
        "language": body.get("language"),
        "is_generated": True,
        "text": body.get("text", "").strip(),
        "segments": segments,
    }


@app.exception_handler(404)
def not_found(request: Request, exc):
    # include the path the function actually received, to debug Vercel routing
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found", "received_path": request.url.path},
    )


@app.get("/")
def health():
    return {"status": "ok", "usage": "POST /api/transcript {\"url\": \"<youtube url>\"}"}


@app.api_route("/api/transcript", methods=["GET", "POST"])
@app.api_route("/transcript", methods=["GET", "POST"])
@app.api_route("/api/index", methods=["GET", "POST"])
@app.api_route("/api/index/transcript", methods=["GET", "POST"])
def transcript(url: Optional[str] = None, body: Optional[TranscriptRequest] = None):
    target = url or (body.url if body else None)
    if not target:
        raise HTTPException(status_code=400, detail="Missing 'url'")
    languages = (body.languages if body and body.languages else None) or ["en"]
    video_id = extract_video_id(target)

    try:
        result = fetch_captions(video_id, languages)
        source = "youtube_captions"
    except Exception as caption_err:
        try:
            result = transcribe_audio(video_id)
            source = "audio_transcription"
        except Exception as audio_err:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "No captions found and audio transcription failed",
                    "caption_error": str(caption_err)[:300],
                    "audio_error": str(audio_err)[:300],
                },
            )

    return {"video_id": video_id, "source": source, **result}
