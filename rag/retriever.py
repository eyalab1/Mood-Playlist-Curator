"""Vector search over the lyrics corpus.

vector_search(query_text, k) embeds the query, then returns the k corpus
tracks whose lyric embeddings are most similar by cosine similarity.

For a few hundred vectors a brute-force numpy computation is instant and far
simpler than a dedicated vector index, so that is what we use.
"""

import json

import numpy as np

from db import get_embedded_tracks
from embeddings import embed_query, from_blob


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and each row of matrix."""
    q = query / (np.linalg.norm(query) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ q


def vector_search(query_text: str, k: int = 10) -> list[dict]:
    """Return the k corpus tracks most similar in lyric meaning to query_text."""
    rows = get_embedded_tracks()
    if not rows:
        return []

    query_vec = np.asarray(embed_query(query_text), dtype=np.float32)
    matrix = np.vstack([from_blob(r["lyrics_embedding"]) for r in rows])
    sims = _cosine(query_vec, matrix)

    top_idx = np.argsort(sims)[::-1][:k]
    results = []
    for idx in top_idx:
        r = rows[idx]
        meta = (
            json.loads(r["audio_features_json"])
            if r["audio_features_json"]
            else {}
        )
        results.append(
            {
                "spotify_track_id": r["spotify_track_id"],
                "name": r["name"],
                "artist": r["artist"],
                "album": r["album"],
                "genres": meta.get("genres", []),
                "mood": meta.get("mood"),
                "similarity": float(sims[idx]),
            }
        )
    return results


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "can't sleep, racing thoughts at 3am"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"Query: {q!r}\n")
    hits = vector_search(q, k)
    if not hits:
        print("Corpus is empty. Run: python -m rag.build_corpus")
    for r in hits:
        mood = r["mood"] or "?"
        print(f"  {r['similarity']:.3f}  {r['name']} - {r['artist']}  [{mood}]")
