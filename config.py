import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501/callback"
)
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# Default to the cheap model during development. Switch to sonnet for the
# final eval run.
MODEL_DEV = "claude-haiku-4-5-20251001"
MODEL_PROD = "claude-sonnet-4-6"
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", MODEL_DEV)

EMBEDDING_MODEL = "voyage-3.5"

DB_PATH = os.getenv("DB_PATH", "data/app.db")
