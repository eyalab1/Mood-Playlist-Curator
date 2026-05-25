import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spotify_user_id TEXT UNIQUE,
    display_name TEXT,
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    mood_prompt TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    spotify_playlist_id TEXT,
    arc_summary TEXT
);

CREATE TABLE IF NOT EXISTS playlist_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    spotify_track_id TEXT NOT NULL,
    track_name TEXT,
    artist TEXT,
    explanation TEXT,
    FOREIGN KEY (playlist_id) REFERENCES playlists(id)
);

CREATE TABLE IF NOT EXISTS tracks (
    spotify_track_id TEXT PRIMARY KEY,
    name TEXT,
    artist TEXT,
    album TEXT,
    audio_features_json TEXT,
    lyrics_text TEXT,
    lyrics_embedding BLOB
);

CREATE TABLE IF NOT EXISTS agent_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id INTEGER,
    agent_name TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_traces_lookup
    ON agent_traces (agent_name, input_hash);

CREATE TABLE IF NOT EXISTS taste_profile (
    user_id INTEGER PRIMARY KEY,
    profile_json TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    playlist_id INTEGER,
    track_id TEXT,
    action TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    prompt_id TEXT,
    metrics_json TEXT,
    playlist_json TEXT,
    FOREIGN KEY (run_id) REFERENCES eval_runs(id)
);
"""


def init_db(path: str = DB_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_conn(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def find_cached_trace(agent_name: str, input_hash: str) -> dict | None:
    """Return the most recent successful output for this agent + input hash,
    or None if no cache hit."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT output_json FROM agent_traces
            WHERE agent_name = ? AND input_hash = ? AND output_json IS NOT NULL
            ORDER BY id DESC LIMIT 1
            """,
            (agent_name, input_hash),
        ).fetchone()
    if row and row["output_json"]:
        return json.loads(row["output_json"])
    return None


def save_trace(
    agent_name: str,
    input_hash: str,
    input_json: str,
    output_json: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
    latency_ms: int | None,
    playlist_id: int | None = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO agent_traces
                (playlist_id, agent_name, input_hash, input_json, output_json,
                 tokens_in, tokens_out, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                playlist_id,
                agent_name,
                input_hash,
                input_json,
                output_json,
                tokens_in,
                tokens_out,
                latency_ms,
            ),
        )
        return cur.lastrowid


def upsert_track(
    spotify_track_id: str,
    name: str,
    artist: str,
    album: str | None,
    audio_features_json: str | None,
    lyrics_text: str | None,
    lyrics_embedding: bytes | None = None,
) -> None:
    """Insert or replace a corpus track."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO tracks
                (spotify_track_id, name, artist, album,
                 audio_features_json, lyrics_text, lyrics_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(spotify_track_id) DO UPDATE SET
                name = excluded.name,
                artist = excluded.artist,
                album = excluded.album,
                audio_features_json = excluded.audio_features_json,
                lyrics_text = excluded.lyrics_text,
                lyrics_embedding = COALESCE(excluded.lyrics_embedding,
                                            tracks.lyrics_embedding)
            """,
            (
                spotify_track_id,
                name,
                artist,
                album,
                audio_features_json,
                lyrics_text,
                lyrics_embedding,
            ),
        )


def get_tracks_needing_embedding() -> list[dict]:
    """Tracks that have lyrics but no embedding yet."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT spotify_track_id, lyrics_text FROM tracks
            WHERE lyrics_text IS NOT NULL AND lyrics_embedding IS NULL
            """
        ).fetchall()
    return [dict(r) for r in rows]


def set_track_embedding(spotify_track_id: str, embedding: bytes) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE tracks SET lyrics_embedding = ? WHERE spotify_track_id = ?",
            (embedding, spotify_track_id),
        )


def get_embedded_tracks() -> list[dict]:
    """All tracks that have an embedding, for vector search."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT spotify_track_id, name, artist, album,
                   audio_features_json, lyrics_embedding
            FROM tracks
            WHERE lyrics_embedding IS NOT NULL
            """
        ).fetchall()
    return [dict(r) for r in rows]


def corpus_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        with_lyrics = conn.execute(
            "SELECT COUNT(*) FROM tracks WHERE lyrics_text IS NOT NULL"
        ).fetchone()[0]
        with_emb = conn.execute(
            "SELECT COUNT(*) FROM tracks WHERE lyrics_embedding IS NOT NULL"
        ).fetchone()[0]
    return {"total": total, "with_lyrics": with_lyrics, "with_embedding": with_emb}


def save_playlist(
    mood_prompt: str,
    arc_summary: str,
    tracks: list[dict],
    user_id: int | None = None,
) -> int:
    """tracks: list of {spotify_track_id, track_name, artist, explanation}."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO playlists (user_id, mood_prompt, arc_summary) "
            "VALUES (?, ?, ?)",
            (user_id, mood_prompt, arc_summary),
        )
        playlist_id = cur.lastrowid
        for i, t in enumerate(tracks):
            conn.execute(
                """
                INSERT INTO playlist_tracks
                    (playlist_id, position, spotify_track_id,
                     track_name, artist, explanation)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    playlist_id,
                    i + 1,
                    t["spotify_track_id"],
                    t.get("track_name"),
                    t.get("artist"),
                    t.get("explanation"),
                ),
            )
        return playlist_id


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
