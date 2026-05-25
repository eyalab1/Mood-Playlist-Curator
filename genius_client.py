"""Genius API wrapper.

One job: fetch_lyrics(track_name, artist) -> str | None.

The corpus builder calls this ~400 times so we keep it tolerant of failures.
A missing or broken song should return None, not raise. We strip the
"EmbedShare" footer that lyricsgenius leaves on the end of every response.
"""

import logging
import re

import lyricsgenius
from requests.exceptions import RequestException, Timeout

from config import GENIUS_ACCESS_TOKEN

logger = logging.getLogger(__name__)


_EMBED_FOOTER = re.compile(r"\d*Embed$")


class GeniusClient:
    def __init__(self):
        if not GENIUS_ACCESS_TOKEN:
            raise RuntimeError("Missing GENIUS_ACCESS_TOKEN in .env")
        self.g = lyricsgenius.Genius(
            GENIUS_ACCESS_TOKEN,
            timeout=10,
            sleep_time=0.5,
            retries=2,
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)"],
        )
        # Silence the library's own print output.
        self.g.verbose = False

    def fetch_lyrics(self, track_name: str, artist: str) -> str | None:
        try:
            song = self.g.search_song(track_name, artist)
        except (Timeout, RequestException) as e:
            logger.warning("Genius network error for '%s' by %s: %s", track_name, artist, e)
            return None
        except Exception as e:
            logger.warning("Genius unexpected error for '%s' by %s: %s", track_name, artist, e)
            return None

        if song is None or not song.lyrics:
            return None

        text = song.lyrics
        # Strip the title prefix Genius prepends, e.g. "Hurt Lyrics\n..."
        if text.lower().startswith(track_name.lower()):
            newline = text.find("\n")
            if newline > -1:
                text = text[newline + 1:]
        # Strip the "123Embed" footer
        text = _EMBED_FOOTER.sub("", text).strip()
        return text or None


_client: GeniusClient | None = None


def get_client() -> GeniusClient:
    global _client
    if _client is None:
        _client = GeniusClient()
    return _client


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    track = sys.argv[1] if len(sys.argv) > 1 else "Hurt"
    artist = sys.argv[2] if len(sys.argv) > 2 else "Johnny Cash"
    print(f"Fetching '{track}' by {artist}...")
    lyrics = get_client().fetch_lyrics(track, artist)
    if lyrics is None:
        print("No lyrics found.")
    else:
        print(f"\n--- Lyrics ({len(lyrics)} chars) ---\n")
        print(lyrics[:800])
        if len(lyrics) > 800:
            print(f"\n... ({len(lyrics) - 800} more chars)")
