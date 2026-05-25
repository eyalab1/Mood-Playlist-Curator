# Mood-Aware Playlist Curator, Build Instructions

## Project goal
Build a web app that takes a natural-language mood description (e.g., "rainy
Sunday, melancholy but hopeful") and generates a personalized Spotify playlist
with per-track explanations, using a multi-agent LLM pipeline. This is a
course project for an LLM course. Timeline: 2-3 days.

## Hard constraints
- Timeline: 2-3 days. Do not add features beyond this spec without asking.
- This is a course project, not production. Keep code simple and readable.
- Never use the em dash character in any text, comments, or UI copy.
- Do not over-engineer. Prefer the dumbest thing that works.

## Locked tech stack (do not change without asking the user)
- Language: Python 3.11+
- App framework: Streamlit (single-file UI, handles OAuth flow easily)
- Database: SQLite (zero setup) with sqlite-vec extension for embeddings
- LLM provider: Anthropic Claude
  - Default model for agents: claude-sonnet-4-6
  - Cheap model for dev/testing: claude-haiku-4-5-20251001
- Embeddings: OpenAI text-embedding-3-small
- External APIs: Spotify Web API, Genius API
- Deployment: Streamlit Community Cloud (free)

## User-provided secrets (in a .env file at project root)
- ANTHROPIC_API_KEY
- OPENAI_API_KEY (embeddings only)
- SPOTIFY_CLIENT_ID
- SPOTIFY_CLIENT_SECRET
- SPOTIFY_REDIRECT_URI (default: http://localhost:8501/callback)
- GENIUS_ACCESS_TOKEN

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
  [Curator] (tools: spotify_search, genius_lyrics, vector_search)
        |
        v  (candidate pool of ~40 tracks)
  [Critic] -> filtered list, can request Curator re-run with feedback
        |
        v  (final ~20 tracks)
  [Sequencer] -> ordered playlist with arc summary
        |
        v
  Save to user's Spotify, render in UI
```

Orchestration is plain Python code (not LangGraph). Every agent input/output
is logged to the agent_traces table.

## File structure
```
mood-playlist-curator/
  CLAUDE.md                     (this file)
  README.md
  .env.example
  .gitignore
  requirements.txt
  app.py                        (Streamlit entry point + UI)
  config.py                     (env vars, model names)
  db.py                         (SQLite setup, schema, helpers)
  spotify_client.py             (OAuth + Spotify API wrapper)
  genius_client.py              (Genius lyrics fetcher)
  embeddings.py                 (OpenAI embedding wrapper)
  agents/
    __init__.py
    base.py                     (shared agent runner, JSON parsing, tracing)
    mood_interpreter.py
    curator.py
    critic.py
    sequencer.py
  orchestrator.py               (runs the pipeline, handles feedback loop)
  rag/
    build_corpus.py             (one-time script: fetch lyrics, embed, store)
    retriever.py
  eval/
    test_set.json               (mood prompts + reference data)
    run_eval.py
    metrics.py
  data/
    app.db                      (gitignored)
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
  - `spotify_search(query, limit)` -> tracks with audio features
  - `genius_lyrics(track_id)` -> lyrics text
  - `vector_search(query_text, k)` -> semantically similar tracks from corpus

Output:
```json
{
  "candidates": [
    {
      "spotify_track_id": "str",
      "name": "str",
      "artist": "str",
      "rationale": "str",
      "audio_features": {},
      "lyrical_themes": ["str"]
    }
  ]
}
```
Target: 30 to 50 candidates.

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
Input: kept tracks (with audio features).
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

### Day 2: RAG + Streamlit UI + Spotify OAuth
1. rag/build_corpus.py: fetch lyrics for ~300-500 popular tracks via Genius,
   embed with OpenAI, store in tracks table
2. rag/retriever.py: vector search over corpus
3. Wire vector_search tool into Curator
4. Streamlit app.py:
   - Spotify OAuth (Authorization Code flow)
   - Mood input box
   - Run pipeline, show playlist with explanations
   - "Save to Spotify" button
   - Show arc summary
5. Multi-turn refinement: text box for follow-up, re-run pipeline with
   previous playlist + refinement instruction in context

Milestone: a friend with a Spotify account can log in and generate a
saveable playlist.

### Day 3: Evaluation + polish + writeup
1. eval/test_set.json: 15 mood prompts with reference data
2. eval/metrics.py: mood_fit (audio feature distance), diversity
   (unique artists/genres), theme coverage, forbidden genre rate
3. eval/run_eval.py: ablations
   - full pipeline vs. single-agent baseline
   - with-RAG vs. without-RAG
   - sonnet vs. haiku
4. Generate plots (matplotlib) of results
5. README.md with setup instructions, screenshots, results
6. Deploy to Streamlit Community Cloud
7. Final report (separate doc)

## Evaluation harness details
Metrics:
- `mood_fit`: 1 - cosine_distance(target_audio_profile, mean_track_features)
- `artist_diversity`: unique_artists / total_tracks
- `genre_diversity`: unique_genres / total_tracks
- `theme_coverage`: fraction of expected_themes mentioned in explanations
- `forbidden_genre_rate`: fraction of tracks in should_avoid_genres
- `explanation_quality`: heuristic (length, mentions audio features, mentions
  lyrics) plus optional human rating on a sample

For each ablation, run all 15 prompts, average metrics, plot bar chart.

## Important rules for Claude Code
- Always validate every LLM JSON output with Pydantic models.
- Cache LLM responses by hash(agent_name + input_json) during development
  to save cost. Cache lives in agent_traces, lookup before calling.
- Log every agent input/output to agent_traces. No exceptions.
- Use claude-haiku-4-5-20251001 during development. Switch to
  claude-sonnet-4-6 for the final eval run only.
- No em dashes in code comments, docstrings, UI copy, or any output.
- Keep functions small and explicit. No premature abstraction.
- Do not commit .env or data/app.db.
- Do not add features (auth providers, payment, social sharing, etc.)
  beyond what is in this spec.

## Things requiring user confirmation before action
- Creating the Spotify Developer app (user does this in browser, you guide)
- Adding tester Spotify emails to the developer dashboard
- Deploying publicly to Streamlit Cloud
- Any LLM run that will cost more than $1 (e.g., full eval with Sonnet)
- Installing system-level dependencies

## What the user must do manually (Claude cannot do these)
1. Create Spotify Developer app at developer.spotify.com/dashboard
2. Add redirect URI to Spotify app settings
3. Whitelist tester Spotify emails in Spotify dashboard (Development Mode
   limit is 25 users)
4. Get Genius API token at genius.com/api-clients
5. Get Anthropic key at console.anthropic.com
6. Get OpenAI key at platform.openai.com
7. Push to GitHub and connect to Streamlit Cloud for deployment
