"""One-time corpus builder for the RAG step.

Two passes:
  Pass 1 (slow): for each seed track, look it up on Spotify (id + genres),
                 fetch lyrics from Genius, store in the tracks table.
  Pass 2 (fast): embed every track that has lyrics but no embedding yet.

Resumable: pass 1 records each attempted seed in a checkpoint file, so a
re-run skips seeds it already tried (whether they succeeded or failed).
Delete data/corpus_checkpoint.json to force a full rebuild.

Run from the project root:
    python -m rag.build_corpus
"""

import json
import logging
import time
from pathlib import Path

from config import DB_PATH
from db import (
    corpus_stats,
    get_tracks_needing_embedding,
    init_db,
    set_track_embedding,
    upsert_track,
)
from embeddings import iter_embed_documents, to_blob
from genius_client import get_client as get_genius
from spotify_client import get_client as get_spotify
from rag.seed_tracks import all_tracks

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(DB_PATH).parent / "corpus_checkpoint.json"


def _load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        return set(json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")))
    return set()


def _save_checkpoint(done: set[str]) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8"
    )


def _seed_key(name: str, artist: str) -> str:
    return f"{name}|||{artist}"


def fetch_pass(limit: int | None = None) -> None:
    """Pass 1: Spotify lookup + Genius lyrics for every seed track."""
    spotify = get_spotify()
    genius = get_genius()
    seeds = all_tracks()
    if limit is not None:
        seeds = seeds[:limit]
    done = _load_checkpoint()

    total = len(seeds)
    skipped = 0
    stored = 0
    no_track = 0
    no_lyrics = 0

    for i, (name, artist, mood) in enumerate(seeds, start=1):
        key = _seed_key(name, artist)
        if key in done:
            skipped += 1
            continue

        prefix = f"[{i}/{total}] {name} - {artist}"
        try:
            track = spotify.search_best(name, artist)
        except Exception as e:
            logger.warning("%s :: Spotify error, will retry next run: %s", prefix, e)
            continue

        if track is None:
            logger.info("%s :: no Spotify match, skipping", prefix)
            no_track += 1
            done.add(key)
            _save_checkpoint(done)
            continue

        lyrics = genius.fetch_lyrics(name, artist)
        if not lyrics:
            logger.info("%s :: no lyrics, skipping", prefix)
            no_lyrics += 1
            done.add(key)
            _save_checkpoint(done)
            continue

        meta = json.dumps(
            {"mood": mood, "genres": track["genres"]}, ensure_ascii=False
        )
        upsert_track(
            spotify_track_id=track["spotify_track_id"],
            name=track["name"],
            artist=track["artist"],
            album=track["album"],
            audio_features_json=meta,
            lyrics_text=lyrics,
            lyrics_embedding=None,
        )
        stored += 1
        logger.info("%s :: stored (%d chars of lyrics)", prefix, len(lyrics))
        done.add(key)
        _save_checkpoint(done)
        # Be polite to the APIs.
        time.sleep(0.3)

    logger.info(
        "\nFetch pass done. stored=%d skipped(prev runs)=%d "
        "no_track=%d no_lyrics=%d",
        stored,
        skipped,
        no_track,
        no_lyrics,
    )


def embed_pass() -> None:
    """Pass 2: embed every track that has lyrics but no embedding.

    Saves each batch to the DB as it completes, so a crash partway through
    keeps the work already done. On the free tier this pass takes roughly
    25 to 30 minutes due to the 10K-tokens/min rate limit.
    """
    pending = get_tracks_needing_embedding()
    if not pending:
        logger.info("Nothing to embed.")
        return

    logger.info(
        "Embedding %d tracks. On the free tier this takes ~25-30 min "
        "(pausing between batches to respect the rate limit).",
        len(pending),
    )
    ids = [t["spotify_track_id"] for t in pending]
    texts = [t["lyrics_text"] for t in pending]

    saved = 0
    for start, batch_embeddings in iter_embed_documents(texts):
        for offset, vec in enumerate(batch_embeddings):
            set_track_embedding(ids[start + offset], to_blob(vec))
            saved += 1
        logger.info("Saved %d/%d embeddings.", saved, len(ids))

    logger.info("Embedding pass done.")


def main() -> None:
    import sys

    limit = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])

    init_db()
    if limit:
        logger.info("Smoke test: building only the first %d seed tracks.\n", limit)
    else:
        logger.info("Starting corpus build. This can take 15 to 25 minutes.\n")
    fetch_pass(limit=limit)
    embed_pass()
    stats = corpus_stats()
    logger.info(
        "\nCorpus ready: %d tracks total, %d with lyrics, %d with embeddings.",
        stats["total"],
        stats["with_lyrics"],
        stats["with_embedding"],
    )


if __name__ == "__main__":
    main()
