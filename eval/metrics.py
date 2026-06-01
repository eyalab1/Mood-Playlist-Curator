"""Evaluation metrics for a generated playlist.

Each metric takes a playlist (a list of track dicts as produced by the
orchestrator, i.e. with keys like track_name, artist, explanation) and
returns a single number. Some metrics also take reference data from the
test set (expected_themes, should_avoid_genres).

Built one metric at a time. Run `python -m eval.metrics` for a small demo.
"""

import numpy as np


def artist_diversity(tracks: list[dict]) -> float:
    """unique_artists / total_tracks. Higher means more variety (0 to 1).

    Artist names are lowercased and trimmed so the same artist is not
    double-counted. A multi-artist string is treated as one unit.
    """
    if not tracks:
        return 0.0
    artists = {t.get("artist", "").strip().lower() for t in tracks}
    artists.discard("")  # ignore tracks with a missing artist
    return len(artists) / len(tracks)


def genre_diversity(tracks: list[dict]) -> float:
    """unique primary genres / tracks that have a genre. 0 to 1, higher = broader.

    Each track is expected to carry a "genres" list (Spotify artist genre
    tags); the eval runner fills these in before scoring. We use the first
    tag as the track's primary genre so the score stays in 0 to 1. Tracks
    with no genre data are left out of the denominator so missing data does
    not unfairly lower the score.
    """
    primary = []
    for t in tracks:
        genres = t.get("genres") or []
        if genres:
            primary.append(genres[0].strip().lower())
    if not primary:
        return 0.0
    return len(set(primary)) / len(primary)


def forbidden_genre_rate(
    tracks: list[dict], should_avoid_genres: list[str]
) -> float:
    """Fraction of tracks in a banned genre. 0 to 1, LOWER is better.

    Matching is case-insensitive substring: a track is flagged if any avoid
    token appears inside any of the track's genre tags, so 'metal' flags
    'death metal' and 'heavy metal'. Tracks with no genre data cannot be
    judged and are counted as non-violations. The denominator is the whole
    playlist, so the score is "fraction of the playlist that broke the rule".
    """
    if not tracks:
        return 0.0
    avoid = [a.strip().lower() for a in (should_avoid_genres or []) if a.strip()]
    if not avoid:
        return 0.0
    violations = 0
    for t in tracks:
        genres = [g.strip().lower() for g in (t.get("genres") or [])]
        if any(tok in g for g in genres for tok in avoid):
            violations += 1
    return violations / len(tracks)


def theme_coverage(tracks: list[dict], expected_themes: list[str]) -> float:
    """Fraction of expected_themes that appear in the track explanations.

    All explanations are concatenated, then each theme is checked as a
    case-insensitive substring. So 'hope' matches 'hopeful'. If
    expected_themes is empty the metric is undefined; we return 1.0 since
    nothing was required.
    """
    if not expected_themes:
        return 1.0
    blob = " ".join((t.get("explanation") or "") for t in tracks).lower()
    hits = sum(1 for theme in expected_themes if theme.strip().lower() in blob)
    return hits / len(expected_themes)


def explanation_quality(tracks: list[dict]) -> float:
    """Heuristic 0 to 1 score for the per-track explanations.

    Averages three signals:
      length: average explanation length divided by a target of 80 chars,
              capped at 1.0
      lyric mentions: fraction of explanations that reference lyrics/vocals
      audio mentions: fraction that reference tempo/beat/instruments/etc.

    A proxy, not ground truth, but it catches obviously thin explanations.
    """
    if not tracks:
        return 0.0
    explanations = [(t.get("explanation") or "") for t in tracks]
    nonempty = [e for e in explanations if e.strip()]
    if not nonempty:
        return 0.0

    target = 80
    avg_len = sum(len(e) for e in nonempty) / len(nonempty)
    length_score = min(1.0, avg_len / target)

    lyric_kw = {
        "lyric", "lyrics", "words", "verse", "chorus", "line", "sings",
        "vocal", "vocals",
    }
    audio_kw = {
        "tempo", "beat", "rhythm", "energy", "melody", "harmony",
        "instrument", "guitar", "piano", "synth", "drum", "bass",
        "production", "groove", "acoustic", "electronic",
    }

    def mentions(e: str, vocab: set[str]) -> bool:
        low = e.lower()
        return any(w in low for w in vocab)

    lyric_score = sum(1 for e in nonempty if mentions(e, lyric_kw)) / len(nonempty)
    audio_score = sum(1 for e in nonempty if mentions(e, audio_kw)) / len(nonempty)

    return (length_score + lyric_score + audio_score) / 3.0


def mood_fit(mood_embedding, tracks: list[dict]) -> float:
    """Average cosine similarity between the mood embedding and each track's
    lyric embedding. Higher = playlist lyrics align better with the mood.

    Substitute for the original "audio feature distance" metric, which used
    the Spotify audio_features endpoint (deprecated for new apps). Tracks
    without a "lyrics_embedding" are skipped; the eval runner attaches
    embeddings from the corpus DB before scoring.

    Cosine similarity is technically in [-1, 1]; for text embeddings the
    values are nearly always positive, so we report the average as a
    0-to-1-ish score.
    """
    if not tracks:
        return 0.0
    q = np.asarray(mood_embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims: list[float] = []
    for t in tracks:
        emb = t.get("lyrics_embedding")
        if emb is None:
            continue
        v = np.asarray(emb, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        sims.append(float(np.dot(q, v)))
    if not sims:
        return 0.0
    return sum(sims) / len(sims)


if __name__ == "__main__":
    sample = [
        {"artist": "Bon Iver", "track_name": "Skinny Love",
         "genres": ["indie folk", "folk"],
         "explanation": "Fragile and aching, sets the melancholy tone honestly."},
        {"artist": "Bon Iver", "track_name": "Holocene",
         "genres": ["indie folk"],
         "explanation": "Expands the space, adds the first hint of warmth."},
        {"artist": "The Beatles", "track_name": "Here Comes the Sun",
         "genres": ["rock", "classic rock"],
         "explanation": "The hopeful turn the mood was reaching for."},
        {"artist": "Adele", "track_name": "Someone Like You",
         "genres": ["pop", "soul"],
         "explanation": "A song about heartbreak and accepting loss."},
        {"artist": "Radiohead", "track_name": "Creep",
         "genres": ["alternative rock", "rock"],
         "explanation": "Loneliness and self-rejection in plain words."},
    ]
    print("Sample playlist (5 tracks, Bon Iver appears twice):")
    for t in sample:
        print(f"  - {t['track_name']} by {t['artist']}  {t['genres']}")

    print(f"\nartist_diversity = {artist_diversity(sample):.2f}")
    print("(4 unique artists / 5 tracks = 0.80)")

    print(f"\ngenre_diversity = {genre_diversity(sample):.2f}")
    print("(primary genres: indie folk, indie folk, rock, pop, alternative rock")
    print(" -> 4 unique / 5 = 0.80)")

    avoid = ["edm", "metal", "pop"]
    rate = forbidden_genre_rate(sample, avoid)
    print(f"\nforbidden_genre_rate (avoid={avoid}) = {rate:.2f}")
    print("(only Adele's 'pop' track is flagged -> 1 / 5 = 0.20; lower is better)")

    themes = ["melancholy", "hope", "heartbreak", "loneliness", "anger"]
    cov = theme_coverage(sample, themes)
    print(f"\ntheme_coverage (themes={themes}) = {cov:.2f}")
    print("(melancholy, hope, heartbreak, loneliness all appear; anger does not")
    print(" -> 4 / 5 = 0.80)")

    quality = explanation_quality(sample)
    print(f"\nexplanation_quality = {quality:.2f}")
    print("(short demo explanations: avg length ~47/80 = 0.58, 1/5 mention lyrics,")
    print(" 0/5 mention audio -> overall ~0.26; real LLM explanations score higher)")

    # mood_fit demo with fake 2D embeddings so the math is readable
    mood_vec = [1.0, 0.0]
    fake_tracks = [
        {"track_name": "Skinny Love",       "lyrics_embedding": [1.0, 0.0]},  # cos 1.00
        {"track_name": "Holocene",          "lyrics_embedding": [0.8, 0.6]},  # cos 0.80
        {"track_name": "Here Comes the Sun","lyrics_embedding": [0.5, 0.866]},  # cos 0.50
        {"track_name": "Someone Like You",  "lyrics_embedding": [0.0, 1.0]},  # cos 0.00
        {"track_name": "Creep",             "lyrics_embedding": [0.6, 0.8]},  # cos 0.60
    ]
    score = mood_fit(mood_vec, fake_tracks)
    print(f"\nmood_fit (fake 2D vectors, mood=[1,0]) = {score:.2f}")
    print("(cosines: 1.00, 0.80, 0.50, 0.00, 0.60 -> mean = 0.58)")
