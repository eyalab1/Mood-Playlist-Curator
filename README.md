# Mood-Aware Playlist Curator

Turn a natural-language mood into a Spotify playlist using a multi-agent LLM
pipeline with lyrics-based RAG. You describe a feeling ("rainy sunday,
melancholy but hopeful") and four AI agents read the mood, curate candidate
tracks, critique the selection, and sequence them into a playlist with a
per-track explanation and an emotional-arc summary.

Course project for **LLM-Augmented Software Practice** (instructor: Mikael
Gorsky), by Eyal Abisdris.

## Live demo

Deployed link: _add your Render URL here once it is live._

No login or registration is needed. Type a mood, press Generate. The first
load after a period of inactivity can take up to a minute while the free host
wakes up.

## What it does

1. You enter a mood in plain language.
2. The **Mood Interpreter** turns it into a structured profile (emotions,
   energy, valence, themes, favored and avoided genres, listening context).
3. The **Curator** builds a candidate pool of 30 to 50 tracks using two
   tools: keyword search over Spotify and lyric-meaning search over a curated
   RAG corpus.
4. The **Critic** filters the pool down to a strong final selection and can
   send feedback back to the Curator for one retry.
5. The **Sequencer** orders the kept tracks into an emotional arc and writes a
   short summary plus a transition note for each track.
6. The web UI renders the playlist with live Spotify players, the reasoning
   for every track, and a refinement box to adjust the result.

## How RAG fits in

A one-time build fetches lyrics for a curated set of songs (Genius), embeds
them with Voyage AI, and stores the vectors in SQLite. At query time the
Curator can call `vector_search`, which embeds the query and returns the
closest songs by lyric meaning. This surfaces songs that match the feeling of
a mood even when their titles do not contain the mood words, something plain
keyword search cannot do.

## Architecture

```
  user mood text
        |
        v
  [Mood Interpreter] -> structured mood profile
        |
        v
  [Curator] (tools: spotify_search, vector_search over the lyrics corpus)
        |
        v  (candidate pool of ~30 to 50 tracks)
  [Critic] -> filtered list, may request one Curator re-run with feedback
        |
        v  (final ~15 to 25 tracks)
  [Sequencer] -> ordered playlist with arc summary
        |
        v
  rendered in the web UI with Spotify embed players
```

Orchestration is plain Python (no LangGraph). Every agent input and output is
validated with Pydantic and logged to the `agent_traces` table, which doubles
as a content-addressed cache.

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Backend / API | FastAPI + Uvicorn |
| Frontend | Single static page, HTML + Tailwind (CDN) + vanilla JS |
| Database | SQLite |
| LLM provider | Anthropic Claude (Haiku for dev, Sonnet for the final eval) |
| Embeddings | Voyage AI (`voyage-3.5`) |
| Vector search | numpy cosine similarity over stored embeddings |
| External APIs | Spotify Web API (search), Genius API (corpus lyrics) |
| Deployment | Render (free tier) |

## Running locally

1. Python 3.11+.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your keys:
   - `ANTHROPIC_API_KEY` (agents)
   - `VOYAGE_API_KEY` (embeddings)
   - `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` (catalog search)
   - `GENIUS_ACCESS_TOKEN` (only needed to build the corpus)
4. The lyrics corpus ships prebuilt in `data/app.db`. To rebuild it from
   scratch:
   ```
   python -m rag.build_corpus
   ```
5. Run the web app:
   ```
   uvicorn server:app --port 8000
   ```
   Open http://localhost:8000.
6. Or run the pipeline from the CLI:
   ```
   python -m orchestrator "rainy sunday, melancholy but hopeful"
   ```

## Project structure

```
mood-playlist-curator/
  server.py            FastAPI backend (serves the page, runs the pipeline)
  static/index.html    the web UI (HTML + Tailwind + vanilla JS)
  orchestrator.py      runs the 4-agent pipeline, handles the Critic retry
  config.py            env vars and model names
  db.py                SQLite schema and helpers
  spotify_client.py    Spotify Web API wrapper (client credentials)
  genius_client.py     Genius lyrics fetcher (corpus build only)
  embeddings.py        Voyage embedding wrapper + serialization
  agents/
    base.py            shared agent runner: tool loop, JSON parsing, tracing, cache
    mood_interpreter.py
    curator.py
    critic.py
    sequencer.py
  rag/
    seed_tracks.py     the curated seed list, grouped by mood
    build_corpus.py    one-time: fetch lyrics, embed, store
    retriever.py       vector_search over the corpus
  eval/                evaluation harness (in progress)
  data/app.db          SQLite database incl. the prebuilt corpus
```

## Notes and limitations

- The corpus is ~350 songs and English-skewed (many Hebrew songs had no
  lyrics available on Genius and were skipped).
- Spotify deprecated its audio-features endpoint for new apps, so the system
  does not rely on numeric audio features; the client degrades gracefully.
- Spotify embed players need the viewer to be logged in to Spotify to play
  full tracks. Without a login, tracks play a 30-second preview where one is
  available.
- The deployed link runs on a free host that sleeps when idle, so the first
  request after a pause is slow.

## License

Course project, not for production use.
