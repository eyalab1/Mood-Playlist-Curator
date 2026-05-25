"""FastAPI backend for the Mood Playlist Curator web UI.

Serves the static page and exposes one endpoint that runs the real
four-agent pipeline. Replaces the Streamlit prototype (app.py).

Run:
    uvicorn server:app --reload --port 8000
Then open http://localhost:8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from db import init_db
from orchestrator import generate_playlist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mood Playlist Curator")

STATIC_DIR = Path(__file__).parent / "static"

init_db()


class GenerateRequest(BaseModel):
    mood_text: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    """Run the full pipeline for a mood and return the playlist JSON.

    Defined as a normal (sync) function so FastAPI runs it in a worker
    thread, keeping the server responsive during the ~30-60s agent run.
    """
    mood = req.mood_text.strip()
    if not mood:
        return JSONResponse({"detail": "Empty mood_text"}, status_code=400)
    try:
        result = generate_playlist(mood)
    except Exception as e:  # noqa: BLE001 - report failure to the frontend
        logger.exception("Pipeline failed for mood: %s", mood)
        return JSONResponse({"detail": str(e)}, status_code=500)
    return JSONResponse(result)
