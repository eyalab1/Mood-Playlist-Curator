# Mood-Aware Playlist Curator, Build Instructions

## Project goal
Build a web app that takes a natural-language mood description (e.g., "rainy
Sunday, melancholy but hopeful") and generates a personalized Spotify playlist
with per-track explanations, using a multi-agent LLM pipeline. This is a
course project for an LLM course. Timeline: roughly two weeks (beta in week one, final the next).

## Hard constraints
- Timeline: ~2 weeks. Do not add features beyond this spec without asking.
- This is a course project, not production. Keep code simple and readable.
- Never use the em dash character in any text, comments, or UI copy.
- Do not over-engineer. Prefer the dumbest thing that works.

## Changes from the original plan (kept for history)
The stack below reflects what was actually built. It diverged from the first
draft in a few deliberate ways:
- App framework is FastAPI + a static HTML/Tailwind page, not Streamlit.
  Streamlit looked too plain for the desired UI.
- Embeddings use Voyage AI, not OpenAI (OpenAI required a paid prepay; Voyage
  has a free tier).
- Vector search uses numpy cosine similarity, not the sqlite-vec extension
  (simpler for a few hundred vectors).
- Deployment is Render, not Streamlit Community Cloud (FastAPI cannot run on
  Streamlit Cloud).
- Spotify OAuth and "save to Spotify" were dropped in favor of Spotify embed
  players, to fit the timeline. Spotify is used in client-credentials mode.

## Locked tech stack (do not change without asking the user)
- Language: Python 3.11+
- Backend / API: FastAPI + Uvicorn
- Frontend: single static page, HTML + Tailwind (CDN) + vanilla JS
- Database: SQLite (zero setup); vector search via numpy cosine similarity
- LLM provider: Anthropic Claude
  - Default model for agents: claude-sonnet-4-6 (final eval)
  - Cheap model for dev/testing and the live beta: claude-haiku-4-5-20251001
- Embeddings: Voyage AI (voyage-3.5)
- External APIs: Spotify Web API (search), Genius API (corpus lyrics)
- Deployment: Render (free tier)

## User-provided secrets (in a .env file at project root)
- ANTHROPIC_API_KEY
- VOYAGE_API_KEY (embeddings)
- SPOTIFY_CLIENT_ID
- SPOTIFY_CLIENT_SECRET
- GENIUS_ACCESS_TOKEN (only needed when rebuilding the corpus; not used at runtime)

If any are missing when starting work, pause and ask the user to obtain them.
Provide step-by-step instructions for getting each key.

## Architecture
Multi-agent pipeline. Each agent is a separate Anthropic API call with a
focused system prompt and Pydantic-validated JSON output.

```
  user mood text
        |
        v
  [Mood Interpreter] -> structured mood profile
        |
        v
  [Curator] (tools: spotify_search, vector_search)
        |
        v  (candidate pool of ~30 to 50 tracks)
  [Critic] -> filtered list, can request Curator re-run with feedback
        |
        v  (final 15 to 25 tracks)
  [Sequencer] -> ordered playlist with arc summary
        |
        v
  Web UI with embedded Spotify players
```

Orchestration is plain Python code (not LangGraph). Every agent input/output
is logged to the agent_traces table.

## File structure
```
mood-playlist-curator/
  CLAUDE.md                     (this file)
  README.md
  REPORT.md                     (final writeup)
  SPEC.md                       (specification doc)
  .env.example
  .gitignore
  requirements.txt
  render.yaml                   (Render deploy blueprint)
  server.py                     (FastAPI backend: serves the page + /api/generate)
  static/index.html             (HTML + Tailwind + vanilla JS, the web UI)
  app.py                        (deprecated Streamlit prototype, kept for history)
  config.py                     (env vars, model names)
  db.py                         (SQLite setup, schema, helpers)
  spotify_client.py             (Spotify Web API wrapper, client credentials only)
  genius_client.py              (Genius lyrics fetcher, used only by corpus build)
  embeddings.py                 (Voyage embedding wrapper)
  agents/
    __init__.py
    base.py                     (shared agent runner: tool loop, JSON, cache, tracing)
    mood_interpreter.py
    curator.py                  (with_rag and without_rag prompts)
    critic.py
    sequencer.py
  orchestrator.py               (pipeline; use_rag flag for ablation)
  rag/
    seed_tracks.py              (curated seed list by mood)
    build_corpus.py             (one-time: fetch lyrics, embed with Voyage, store)
    retriever.py                (vector_search by numpy cosine similarity)
  eval/
    test_set.json               (15 mood prompts + reference data)
    metrics.py                  (artist_diversity, theme_coverage, expl_quality, mood_fit)
    run_eval.py                 (the ablation runner)
    plot_results.py             (saves PNGs in eval/plots/)
  data/
    app.db                      (SQLite incl. the prebuilt corpus; committed)
```

## Database schema (SQLite)
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

## Agent contracts

### Mood Interpreter
Input: `{ "mood_text": str, "user_taste_profile": object | null }`
Output:
```json
{
  "emotions": ["str"],
  "energy": 0.0,
  "valence": 0.0,
  "tempo_range": [60, 90],
  "themes": ["str"],
  "genres_favor": ["str"],
  "genres_avoid": ["str"],
  "context": "background reading | workout | drive | etc."
}
```

### Curator
Input: mood profile + taste profile.
Tools available (function calling):
  - `spotify_search(query, limit)` -> tracks (id, name, artist, album)
  - `vector_search(query_text, k)` -> semantically similar tracks from corpus
(`genius_lyrics` is used only by the one-time corpus build, not as a live tool.)

Output:
```json
{
  "candidates": [
    {
      "spotify_track_id": "str",
      "name": "str",
      "artist": "str",
      "rationale": "str",
      "lyrical_themes": ["str"]
    }
  ]
}
```
Target: 30 to 50 candidates. The original spec also included `audio_features`
per candidate; this was removed because Spotify deprecated `/audio-features`
for new apps.

### Critic
Input: mood profile + candidate pool.
Output:
```json
{
  "kept": ["spotify_track_id"],
  "rejected": [{"spotify_track_id": "str", "reason": "str"}],
  "global_warnings": ["str"],
  "request_more": false,
  "feedback_for_curator": null
}
```
If `request_more` is true, orchestrator re-runs Curator with
`feedback_for_curator` appended to its input. Maximum 1 retry to bound
latency and cost.

### Sequencer
Input: kept tracks (name, artist, rationale).
Output:
```json
{
  "ordered_tracks": [
    {
      "spotify_track_id": "str",
      "position": 1,
      "transition_note": "why this comes after the previous"
    }
  ],
  "arc_summary": "2 to 3 sentences describing the emotional shape"
}
```

## Build order

### Day 1: Core pipeline end-to-end (CLI only, no UI yet)
1. Project scaffold, requirements.txt, .env.example, .gitignore
2. db.py: schema and init function
3. spotify_client.py: client credentials flow first (no user OAuth yet),
   implement search + audio_features
4. agents/base.py: Anthropic call wrapper with JSON output, tracing,
   retry on parse failure (max 2 retries)
5. Implement all 4 agents with their prompts
6. orchestrator.py: chain agents, persist traces
7. CLI: `python -m orchestrator "rainy sunday, melancholy"` prints playlist

Milestone: one working playlist generated from CLI, traces in DB.

### Day 2 (week 1): RAG + web UI
1. rag/build_corpus.py: fetch lyrics for ~300-500 popular tracks via Genius,
   embed with Voyage, store in tracks table
2. rag/retriever.py: vector search over corpus (numpy cosine)
3. Wire vector_search tool into Curator
4. FastAPI server.py + static/index.html:
   - Mood input box, "Generate" button
   - Playlist rendered with explanations + embedded Spotify players (no OAuth)
   - Arc summary shown above the tracks
5. Multi-turn refinement: text box for follow-up; combined prompt re-runs the pipeline

Milestone: a user can open the URL and generate a playlist through the web UI.

### Day 3 (week 2): Evaluation + polish + writeup
1. eval/test_set.json: 15 mood prompts with reference data
2. eval/metrics.py: artist_diversity, theme_coverage, explanation_quality,
   mood_fit (lyric-embedding substitute for the original audio-feature mood_fit,
   which is no longer computable). The genre-based metrics from the spec
   (genre_diversity, forbidden_genre_rate) were dropped after Spotify also
   restricted the artist genres field for new apps.
3. eval/run_eval.py: with-RAG vs without-RAG ablation across all 15 prompts.
   The Sonnet-vs-Haiku and single-agent-baseline ablations were dropped to
   stay within the Anthropic budget.
4. eval/plot_results.py: matplotlib plots saved to eval/plots/
5. README.md with setup instructions
6. Deploy to Render (FastAPI + static page)
7. Final report (REPORT.md)

## Evaluation harness details
Metrics actually computed (four):
- `artist_diversity`: unique_artists / total_tracks
- `theme_coverage`: fraction of `expected_themes` that appear in the track
  explanations (case-insensitive substring)
- `explanation_quality`: heuristic average of length / lyric-vocab mentions /
  audio-vocab mentions
- `mood_fit`: average cosine similarity between the embedded mood text and
  each track's lyric embedding (Voyage). Substituted for the spec's original
  audio-feature-distance metric, which became uncomputable when Spotify
  deprecated `/audio-features` for new apps.

Metrics dropped from the original spec:
- `genre_diversity`, `forbidden_genre_rate`: Spotify restricted the artist
  `genres` field for new apps; no reliable per-track genre source remained.

The runner also reports cross-prompt unique-track counts per configuration,
which surfaces the corpus-size trade-off in RAG mode. All 15 prompts run on
both configurations; metrics are averaged.

## Important rules for Claude Code
- Always validate every LLM JSON output with Pydantic models.
- Cache LLM responses by hash(agent_name + input_json) during development
  to save cost. Cache lives in agent_traces, lookup before calling.
- Log every agent input/output to agent_traces. No exceptions.
- Use claude-haiku-4-5-20251001 during development. Switch to
  claude-sonnet-4-6 for the final eval run only.
- No em dashes in code comments, docstrings, UI copy, or any output.
- Keep functions small and explicit. No premature abstraction.
- Do not commit `.env`. `data/app.db` IS committed deliberately so the prebuilt corpus ships with the deploy.
- Do not add features (auth providers, payment, social sharing, etc.)
  beyond what is in this spec.

## Things requiring user confirmation before action
- Creating the Spotify Developer app (user does this in browser, you guide)
- Deploying publicly to Render
- Any LLM run that will cost more than $1 (e.g., the full eval)
- Installing system-level dependencies

## What the user must do manually (Claude cannot do these)
1. Create Spotify Developer app at developer.spotify.com/dashboard
2. Get Genius API token at genius.com/api-clients (only needed to rebuild the corpus)
3. Get Anthropic key at console.anthropic.com
4. Get Voyage API key at dash.voyageai.com
5. Push to GitHub and deploy via Render (render.com), using `render.yaml` as a Blueprint
