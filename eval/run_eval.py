"""Evaluation runner: real playlists, real scores, real ablations.

Runs the pipeline on every prompt in eval/test_set.json for two
configurations (with_rag, without_rag), scores each generated playlist with
four metrics (artist_diversity, theme_coverage, explanation_quality,
mood_fit), and stores results in eval_runs / eval_results.

Two genre-based metrics from the original spec (genre_diversity,
forbidden_genre_rate) were dropped because Spotify silently restricted the
artist genres field for new apps, leaving no reliable per-track genre source.

Modes:
    python -m eval.run_eval              # dry-run: 1 prompt x 2 configs (~$0.05)
    python -m eval.run_eval --full       # full run: 15 prompts x 2 configs (~$0.60-1.50)
"""

import json
import logging
import sys
import time
from pathlib import Path

from db import get_conn, get_embedded_tracks, init_db
from embeddings import embed_documents, embed_query, from_blob
from eval.metrics import (
    artist_diversity,
    explanation_quality,
    mood_fit,
    theme_coverage,
)
from genius_client import get_client as get_genius
from orchestrator import generate_playlist

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


CONFIGS = [
    {"name": "with_rag", "use_rag": True},
    {"name": "without_rag", "use_rag": False},
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_corpus_index() -> dict[str, dict]:
    """Return {spotify_track_id: {"lyrics_embedding": ndarray}}.

    Loaded once so the main loop is dict lookups, not DB hits.
    """
    rows = get_embedded_tracks()
    index: dict[str, dict] = {}
    for r in rows:
        index[r["spotify_track_id"]] = {
            "lyrics_embedding": from_blob(r["lyrics_embedding"]),
        }
    return index


def enrich_track(
    track: dict,
    corpus_index: dict[str, dict],
    genius=None,
    embed_cache: dict | None = None,
) -> dict:
    """Attach lyrics_embedding (or None) to a track in place.

    Order of attempts:
      1. If the track is in the corpus, use the stored embedding.
      2. If a genius client and per-run cache are provided, fetch lyrics from
         Genius and embed them with Voyage. Cache the result so duplicate
         tracks across configs are not refetched.
      3. Otherwise leave lyrics_embedding as None (mood_fit will skip the
         track).
    """
    tid = track.get("spotify_track_id")
    if tid in corpus_index:
        track["lyrics_embedding"] = corpus_index[tid]["lyrics_embedding"]
        return track

    if embed_cache is not None and tid in embed_cache:
        track["lyrics_embedding"] = embed_cache[tid]
        return track

    embedding = None
    if genius is not None:
        try:
            lyrics = genius.fetch_lyrics(
                track.get("track_name", ""), track.get("artist", "")
            )
            if lyrics:
                embedding = embed_documents([lyrics])[0]
        except Exception as e:  # noqa: BLE001 - never abort eval over one track
            logger.debug(
                "lyric fetch/embed failed for %s by %s: %s",
                track.get("track_name"),
                track.get("artist"),
                e,
            )

    track["lyrics_embedding"] = embedding
    if embed_cache is not None and tid:
        embed_cache[tid] = embedding
    return track


def score_playlist(result: dict, mood_embedding, prompt: dict) -> dict:
    """Run the four metrics on a generated playlist."""
    tracks = result.get("tracks", [])
    return {
        "artist_diversity": artist_diversity(tracks),
        "theme_coverage": theme_coverage(
            tracks, prompt.get("expected_themes", [])
        ),
        "explanation_quality": explanation_quality(tracks),
        "mood_fit": mood_fit(mood_embedding, tracks),
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_eval_run(config_json: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO eval_runs (config_json) VALUES (?)",
            (json.dumps(config_json, ensure_ascii=False),),
        )
        return cur.lastrowid


def save_eval_result(
    run_id: int,
    prompt_id: str,
    config_name: str,
    scores: dict,
    playlist: dict,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO eval_results (run_id, prompt_id, metrics_json, playlist_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                run_id,
                f"{config_name}:{prompt_id}",
                json.dumps(scores, ensure_ascii=False),
                json.dumps(_compact_playlist(playlist), ensure_ascii=False),
            ),
        )


def _compact_playlist(result: dict) -> dict:
    """Strip embeddings before storing in JSON (they are big and binary)."""
    compact_tracks = []
    for t in result.get("tracks", []):
        copy = {k: v for k, v in t.items() if k != "lyrics_embedding"}
        compact_tracks.append(copy)
    return {
        "mood_text": result.get("mood_text"),
        "arc_summary": result.get("arc_summary"),
        "tracks": compact_tracks,
    }


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------
def main() -> None:
    full = "--full" in sys.argv

    init_db()
    test_set_path = Path(__file__).parent / "test_set.json"
    prompts = json.loads(test_set_path.read_text(encoding="utf-8"))["prompts"]
    if not full:
        prompts = prompts[:1]
        logger.info("DRY RUN: %d prompt x %d configs (use --full for all 15 prompts).",
                    len(prompts), len(CONFIGS))
    else:
        logger.info("FULL RUN: %d prompts x %d configs.", len(prompts), len(CONFIGS))

    logger.info("Loading corpus index...")
    corpus_index = load_corpus_index()
    logger.info("Corpus index: %d embedded tracks.", len(corpus_index))

    logger.info("Embedding mood texts (%d)...", len(prompts))
    mood_embeds = {p["id"]: embed_query(p["mood_text"]) for p in prompts}

    run_id = save_eval_run(
        {
            "mode": "full" if full else "dry",
            "configs": [c["name"] for c in CONFIGS],
            "n_prompts": len(prompts),
        }
    )
    logger.info("eval_runs row id = %d", run_id)

    all_scores: dict[str, list[dict]] = {c["name"]: [] for c in CONFIGS}
    all_tracks: dict[str, list[tuple[str, str]]] = {c["name"]: [] for c in CONFIGS}

    # Per-run cache so non-corpus tracks are fetched + embedded at most once.
    genius = get_genius()
    embed_cache: dict = {}

    for config in CONFIGS:
        logger.info("\n=== Config: %s ===", config["name"])
        for i, prompt in enumerate(prompts, start=1):
            label = prompt["mood_text"][:60]
            logger.info("[%d/%d] %s :: %s", i, len(prompts), prompt["id"], label)
            try:
                t0 = time.time()
                result = generate_playlist(
                    prompt["mood_text"], use_rag=config["use_rag"]
                )
                elapsed = round(time.time() - t0, 1)
            except Exception as e:  # noqa: BLE001 - never abort the whole eval
                logger.warning("  generation failed: %s", e)
                continue

            for t in result.get("tracks", []):
                enrich_track(t, corpus_index, genius, embed_cache)

            scores = score_playlist(result, mood_embeds[prompt["id"]], prompt)
            scores["_elapsed_s"] = elapsed
            all_scores[config["name"]].append({"prompt_id": prompt["id"], **scores})
            all_tracks[config["name"]].extend(
                (t.get("track_name", ""), t.get("artist", ""))
                for t in result.get("tracks", [])
            )
            save_eval_result(run_id, prompt["id"], config["name"], scores, result)
            logger.info(
                "  artist=%.2f theme=%.2f expl=%.2f mood=%.2f  (%.1fs)",
                scores["artist_diversity"],
                scores["theme_coverage"],
                scores["explanation_quality"],
                scores["mood_fit"],
                elapsed,
            )

    print_summary(all_scores, all_tracks)


def print_summary(
    all_scores: dict[str, list[dict]],
    all_tracks: dict[str, list[tuple[str, str]]] | None = None,
) -> None:
    print()
    print("=" * 70)
    print("SUMMARY (averaged across prompts)")
    print("=" * 70)
    metrics = ["artist_diversity", "theme_coverage", "explanation_quality", "mood_fit"]
    print(f"{'config':<14} | " + " | ".join(f"{m[:9]:>9}" for m in metrics) + " | runs")
    print("-" * 70)
    averaged: dict[str, dict[str, float]] = {}
    for name, rows in all_scores.items():
        if not rows:
            print(f"{name:<14} | (no runs)")
            continue
        avg = {m: sum(r[m] for r in rows) / len(rows) for m in metrics}
        averaged[name] = avg
        print(
            f"{name:<14} | "
            + " | ".join(f"{avg[m]:>9.3f}" for m in metrics)
            + f" | {len(rows):>4}"
        )

    if "with_rag" in averaged and "without_rag" in averaged:
        print("\nDelta (with_rag minus without_rag):")
        for m in metrics:
            d = averaged["with_rag"][m] - averaged["without_rag"][m]
            sign = "+" if d >= 0 else ""
            print(f"  {m:<22} {sign}{d:.3f}")

    if all_tracks:
        print("\nUnique tracks across all prompts (cross-prompt diversity):")
        for name, items in all_tracks.items():
            if not items:
                print(f"  {name:<14} : (no tracks)")
                continue
            keys = {(n.strip().lower(), a.strip().lower()) for n, a in items}
            total = len(items)
            unique = len(keys)
            pct = (unique / total * 100) if total else 0.0
            print(
                f"  {name:<14} : {unique} unique / {total} total ({pct:.0f}% unique)"
            )


if __name__ == "__main__":
    main()
