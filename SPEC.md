# Mood-Aware Playlist Curator, Specification

**Author:** Eyal Abisdris
**Course:** LLM-Augmented Software Practice
**Instructor:** Mikael Gorsky
**Date:** May 2026

## 1. Overview

This project is a web application that takes a natural-language description
of a mood (for example, "rainy Sunday, melancholy but hopeful") and
generates a personalized Spotify playlist with a per-track explanation and
an emotional-arc summary. The system uses a multi-agent LLM pipeline:
four specialized agents (Mood Interpreter, Curator, Critic, Sequencer)
cooperate through strict structured contracts to produce the final
playlist. The project is a course assignment delivered over roughly two
weeks (beta in week one, final in week two).

## 2. Goals and scope

In scope:
- Translate free-form mood text into structured musical features.
- Search Spotify for candidate tracks using LLM-driven tool calls.
- Filter, order, and explain a final 15 to 25 track playlist.
- Persist playlists and full agent traces to a local database.
- Provide a CLI for development and a deployed web UI for end users.
- Provide embedded Spotify players per track so the playlist is listenable
  directly in the app.
- Provide a basic retrieval-augmented generation (RAG) step over a small
  lyrics corpus.
- Evaluate the system with reproducible ablation studies.

Out of scope:
- Multi-user collaboration, social features, or sharing.
- Production-grade scaling, authentication providers beyond Spotify, or
  payment.
- Real-time learning from user feedback during a single session.

## 3. Tech stack

The stack below reflects what was actually built. Section 14 notes the
deliberate deviations from the original draft.

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Strong LLM ecosystem and fast iteration. |
| Backend / API | FastAPI + Uvicorn | Lightweight HTTP server; deploys cleanly to Render. |
| Frontend | Single static HTML page (Tailwind via CDN + vanilla JS) | Full design control without a build step. |
| Database | SQLite | Zero setup; vector search via numpy cosine (sufficient at ~350 vectors). |
| LLM provider | Anthropic Claude | Haiku 4.5 throughout (Sonnet ablation dropped to stay within budget). |
| Embeddings | Voyage AI `voyage-3.5` | Free tier of 200M tokens removes the OpenAI prepay requirement. |
| External APIs | Spotify Web API (search), Genius API (corpus lyrics) | Catalog access and one-time lyric fetch. |
| Deployment | Render (free tier) | Static + Python server hosted together. |

## 4. Configuration

The system reads its secrets from a `.env` file at the project root:

- `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `GENIUS_ACCESS_TOKEN` (used only when rebuilding the corpus; not needed at runtime)

## 5. Architecture

The pipeline is a sequence of four LLM agent calls, each with a focused
system prompt and Pydantic-validated JSON output:

```
  user mood text
        |
        v
  [Mood Interpreter] -> structured mood profile
        |
        v
  [Curator] (tools: spotify_search, vector_search)
        |
        v  (candidate pool of approximately 30 to 50 tracks)
  [Critic] -> filtered list, may request that the Curator re-runs
        |
        v  (final 15 to 25 tracks)
  [Sequencer] -> ordered playlist with arc summary
        |
        v
  saved to local DB, rendered in the web UI with embedded Spotify players
```

Orchestration is plain Python rather than an LLM framework like
LangGraph. Every agent input and output is logged to the `agent_traces`
table for inspection, caching, and evaluation.

## 6. Project layout

```
mood-playlist-curator/
  SPEC.md                         (this file)
  CLAUDE.md                       (working notes for the AI coding assistant)
  REPORT.md                       (final report)
  README.md
  .env.example
  .gitignore
  requirements.txt
  render.yaml                     (Render deploy blueprint)
  server.py                       (FastAPI backend: serves the page + /api/generate)
  static/index.html               (HTML + Tailwind + vanilla JS, the web UI)
  app.py                          (deprecated Streamlit prototype, kept for history)
  config.py                       (env vars, model names)
  db.py                           (SQLite setup, schema, helpers)
  spotify_client.py               (Spotify Web API wrapper, client credentials only)
  genius_client.py                (Genius lyrics fetcher, used only by corpus build)
  embeddings.py                   (Voyage embedding wrapper)
  agents/
    __init__.py
    base.py                       (shared agent runner: tool loop, JSON parsing, cache, tracing)
    mood_interpreter.py
    curator.py                    (with_rag and without_rag prompts)
    critic.py
    sequencer.py
  orchestrator.py                 (pipeline; use_rag flag for ablation)
  rag/
    seed_tracks.py                (curated seed list by mood)
    build_corpus.py               (one-time: fetch lyrics, embed with Voyage, store)
    retriever.py                  (vector_search by numpy cosine similarity)
  eval/
    test_set.json                 (15 mood prompts + reference data)
    metrics.py                    (artist_diversity, theme_coverage, expl_quality, mood_fit)
    run_eval.py                   (the ablation runner)
    plot_results.py               (saves PNGs in eval/plots/)
  data/
    app.db                        (SQLite incl. the prebuilt corpus; committed)
```

## 7. Database schema

```
users(id, spotify_user_id, display_name, access_token, refresh_token,
      token_expires_at, created_at)

playlists(id, user_id, mood_prompt, created_at, spotify_playlist_id,
          arc_summary)

playlist_tracks(id, playlist_id, position, spotify_track_id, track_name,
                artist, explanation)

tracks(spotify_track_id PK, name, artist, album, audio_features_json,
       lyrics_text, lyrics_embedding BLOB)

agent_traces(id, playlist_id, agent_name, input_json, output_json,
             tokens_in, tokens_out, latency_ms, created_at)

taste_profile(user_id PK, profile_json, updated_at)

feedback(id, user_id, playlist_id, track_id, action, created_at)
  -- action: 'skip' | 'save' | 'replay' | 'thumbs_up' | 'thumbs_down'

eval_runs(id, config_json, created_at)
eval_results(id, run_id, prompt_id, metrics_json, playlist_json)
```

The `agent_traces` table is central to the design. It serves three
purposes: persistent debugging logs, a content-addressed cache that lets
agents skip duplicate LLM calls during development, and the data source
for the evaluation harness.

## 8. Agent contracts

### 8.1 Mood Interpreter

Input:
```json
{ "mood_text": "string", "user_taste_profile": "object | null" }
```
Output:
```json
{
  "emotions": ["string"],
  "energy": 0.0,
  "valence": 0.0,
  "tempo_range": [60, 90],
  "themes": ["string"],
  "genres_favor": ["string"],
  "genres_avoid": ["string"],
  "context": "background reading | workout | drive | etc."
}
```

### 8.2 Curator

Input: mood profile plus user taste profile.

Tools available via function calling:
- `spotify_search(query, limit)` returns tracks (id, name, artist, album).
- `vector_search(query_text, k)` returns semantically similar tracks
  from the local corpus.

The original draft also exposed `genius_lyrics` as a live tool; in the final
build it is used only during the one-time corpus construction.

Output:
```json
{
  "candidates": [
    {
      "spotify_track_id": "string",
      "name": "string",
      "artist": "string",
      "rationale": "string",
      "lyrical_themes": ["string"]
    }
  ]
}
```
Target: 30 to 50 candidates. The original draft included an `audio_features`
field per candidate; this was removed because Spotify deprecated the
`/audio-features` endpoint for new apps.

### 8.3 Critic

Input: mood profile plus the Curator's candidate pool.

Output:
```json
{
  "kept": ["spotify_track_id"],
  "rejected": [{"spotify_track_id": "string", "reason": "string"}],
  "global_warnings": ["string"],
  "request_more": false,
  "feedback_for_curator": null
}
```

If `request_more` is true, the orchestrator re-runs the Curator with
`feedback_for_curator` appended to its input. The retry is capped at
one round to bound latency and cost.

### 8.4 Sequencer

Input: kept tracks (name, artist, rationale).

Output:
```json
{
  "ordered_tracks": [
    {
      "spotify_track_id": "string",
      "position": 1,
      "transition_note": "why this track follows the previous one"
    }
  ],
  "arc_summary": "2 to 3 sentences describing the emotional shape"
}
```

## 9. Build plan

### 9.1 Day 1, Core pipeline end to end (CLI only, no UI yet)
1. Project scaffold: `requirements.txt`, `.env.example`, `.gitignore`.
2. `db.py`: schema and `init_db` function.
3. `spotify_client.py`: Spotify Client Credentials flow, search,
   audio features.
4. `agents/base.py`: shared Anthropic call wrapper with JSON output,
   tracing, content-addressed caching, and retry on parse failure
   (maximum two retries).
5. All four agents with their system prompts and Pydantic models.
6. `orchestrator.py`: chains the agents, persists traces, supports the
   Critic's retry loop.
7. CLI: `python -m orchestrator "rainy sunday, melancholy"` prints a
   playlist.

Milestone: one working playlist generated from the CLI, with full
traces in the database.

### 9.2 Week 1, RAG and the web UI
1. `rag/build_corpus.py`: fetch lyrics for approximately 300 to 500
   popular tracks via Genius, embed with Voyage, store in the
   `tracks` table.
2. `rag/retriever.py`: vector search over the corpus (numpy cosine).
3. Wire the `vector_search` tool into the Curator.
4. `server.py` (FastAPI) and `static/index.html` (Tailwind + vanilla JS):
   - Mood input box, "Generate" button, playlist rendered with
     explanations and arc summary.
   - Embedded Spotify mini-player per track (no OAuth, no save flow).
5. Multi-turn refinement: a follow-up text input that re-runs the
   pipeline with the previous mood plus the refinement appended.

Milestone: anyone with the link can generate a playlist through the web UI.

### 9.3 Week 2, Evaluation, deployment, and writeup
1. `eval/test_set.json`: 15 mood prompts with reference data.
2. `eval/metrics.py`: `artist_diversity`, `theme_coverage`,
   `explanation_quality`, and a lyric-embedding-based `mood_fit`. The
   genre-based metrics from the draft (`genre_diversity`,
   `forbidden_genre_rate`) were dropped after Spotify restricted the
   artist `genres` field for new apps. The original `mood_fit`
   (audio-feature distance) was also unavailable for the same reason
   and was substituted with the lyric-embedding version.
3. `eval/run_eval.py`: with-RAG versus without-RAG ablation across all
   15 prompts. The Haiku-versus-Sonnet and single-agent-baseline
   ablations from the draft were dropped to stay within the Anthropic
   budget.
4. `eval/plot_results.py`: matplotlib plots saved in `eval/plots/`.
5. `README.md` with setup instructions.
6. Deploy to Render (FastAPI + static page) via `render.yaml`.
7. Final written report (`REPORT.md`).

## 10. Evaluation methodology

Metrics actually computed (four):
- `artist_diversity`: unique artists divided by total tracks.
- `theme_coverage`: fraction of `expected_themes` mentioned (case-insensitive
  substring) across the generated explanations.
- `explanation_quality`: heuristic average of length / lyric-vocab mentions /
  audio-vocab mentions.
- `mood_fit`: average cosine similarity between the embedded mood text and
  each track's lyric embedding. Substituted for the draft's audio-feature
  distance, which became uncomputable when Spotify deprecated
  `/audio-features` for new apps.

Metrics dropped from the draft:
- `genre_diversity` and `forbidden_genre_rate`: Spotify also restricted the
  artist `genres` field for new apps; no reliable per-track genre source
  remained within the project's time budget.

The runner additionally reports cross-prompt unique-track counts per
configuration, which surfaces the corpus-size trade-off in RAG mode. All 15
prompts run on both configurations; metrics are averaged. Results appear as
bar charts in `eval/plots/`.

## 11. Implementation rules

- Every LLM JSON output is validated with a Pydantic model.
- LLM responses are cached by `hash(agent_name + input_json)` during
  development to save cost. The cache resides in `agent_traces`; the
  pipeline performs a cache lookup before each Anthropic call.
- Every agent invocation, successful or failed, is logged to
  `agent_traces`.
- The model used throughout (including the final evaluation) is
  `claude-haiku-4-5-20251001`. A planned Sonnet ablation was dropped to
  stay within the Anthropic budget.
- Functions are kept small and explicit; abstraction is added only when
  required by concrete code.
- Secrets (`.env`) are excluded from version control. `data/app.db` IS
  committed so the prebuilt corpus ships with the deployed application.

## 12. Operational guardrails

Certain actions are gated to avoid surprises and to bound cost:
- Creation of the Spotify Developer app is performed manually by the
  developer through the Spotify dashboard.
- Public deployment to Render is performed manually via `render.yaml`.
- Any LLM run with an estimated cost greater than one US dollar
  (for example, the full ablation evaluation) requires explicit
  confirmation before kicking off.
- Installation of system-level dependencies requires explicit
  confirmation.

## 13. Manual setup prerequisites

The following steps are performed by the developer outside of the codebase:
1. Create a Spotify Developer application at
   developer.spotify.com/dashboard.
2. Obtain a Genius API token from genius.com/api-clients (used only when
   rebuilding the corpus; not required at runtime).
3. Obtain an Anthropic API key from console.anthropic.com.
4. Obtain a Voyage API key from dash.voyageai.com.
5. Push the project to GitHub and connect the repository to Render via
   the included `render.yaml` Blueprint.

## 14. Deviations from the original draft

The deliverable diverged from the first draft of this specification in a
small number of deliberate ways, all of which are recorded in
`REPORT.md` (Section 4). The most significant are:

- App framework changed from Streamlit to FastAPI + a custom static HTML
  page, to allow a more polished UI than Streamlit can produce out of the
  box.
- Embeddings changed from OpenAI to Voyage AI, to avoid an upfront prepay
  requirement and to use a free tier (200M tokens) that comfortably
  covers the project.
- Vector search uses numpy cosine similarity instead of `sqlite-vec` (the
  scale, ~350 vectors, did not justify the dependency).
- Spotify OAuth and the "save to Spotify" feature were dropped in favor
  of embedded Spotify mini-players, to fit the timeline.
- Two metrics (`genre_diversity`, `forbidden_genre_rate`) and the
  audio-feature `mood_fit` were dropped or substituted because Spotify
  deprecated the relevant endpoints (`/audio-features`, the artist
  `genres` field) for new apps. A lyric-embedding `mood_fit` replaced
  the original.
- Deployment moved from Streamlit Community Cloud to Render, which the
  new FastAPI service requires.
