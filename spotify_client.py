import logging

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

logger = logging.getLogger(__name__)


class SpotifyClient:
    def __init__(self):
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise RuntimeError(
                "Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in .env"
            )
        auth = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
        )
        self.sp = spotipy.Spotify(auth_manager=auth)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search Spotify and return a list of track summaries.

        Spotify rejects most limit values for newer apps with "Invalid limit".
        Empirically only limit=10 works reliably with this client credentials
        setup, so we ignore the requested limit and always send 10. The
        Curator can compensate by issuing more searches.
        """
        results = self.sp.search(q=query, type="track", limit=10)
        tracks = []
        for item in results.get("tracks", {}).get("items", []):
            tracks.append(
                {
                    "spotify_track_id": item["id"],
                    "name": item["name"],
                    "artist": ", ".join(a["name"] for a in item["artists"]),
                    "album": item["album"]["name"],
                    "popularity": item.get("popularity"),
                    "preview_url": item.get("preview_url"),
                }
            )
        return tracks

    def search_best(self, track_name: str, artist: str) -> dict | None:
        """Return the single best-matching track plus its artist genre tags,
        or None if nothing matched. Used by the corpus builder.

        Genres live on the artist object, not the track, so we do one extra
        lookup. If that lookup fails we still return the track with empty
        genres rather than dropping it.
        """
        query = f"{track_name} {artist}"
        results = self.sp.search(q=query, type="track", limit=10)
        items = results.get("tracks", {}).get("items", [])
        if not items:
            return None
        item = items[0]
        genres: list[str] = []
        try:
            artist_id = item["artists"][0]["id"]
            genres = self.sp.artist(artist_id).get("genres", [])
        except spotipy.SpotifyException as e:
            logger.warning("artist genres lookup failed for '%s': %s", artist, e)
        return {
            "spotify_track_id": item["id"],
            "name": item["name"],
            "artist": ", ".join(a["name"] for a in item["artists"]),
            "album": item["album"]["name"],
            "genres": genres,
        }

    def audio_features(self, track_ids: list[str]) -> list[dict | None]:
        """Return audio features per track, or None for tracks where the
        endpoint failed. Spotify deprecated this endpoint for new apps in
        late 2024; we degrade gracefully so the pipeline still runs."""
        if not track_ids:
            return []
        try:
            features = self.sp.audio_features(track_ids)
            return features if features else [None] * len(track_ids)
        except spotipy.SpotifyException as e:
            logger.warning("audio_features failed: %s", e)
            return [None] * len(track_ids)


_client: SpotifyClient | None = None


def get_client() -> SpotifyClient:
    """Lazy module-level singleton."""
    global _client
    if _client is None:
        _client = SpotifyClient()
    return _client


if __name__ == "__main__":
    client = get_client()
    sample = client.search("rainy day melancholy", limit=5)
    for t in sample:
        print(f"- {t['name']} by {t['artist']} (id: {t['spotify_track_id']})")
