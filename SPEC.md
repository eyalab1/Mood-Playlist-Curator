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
playlist. The project is a course assignment with a 2-3 day timeline.

## 2. Goals and scope

In scope:
- Translate free-form mood text into structured musical features.
- Search Spotify for candidate tracks using LLM-driven tool calls.
- Filter, order, and explain a final 15 to 25 track playlist.
- Persist playlists and full agent traces to a local database.
- Provide a CLI for development and a Streamlit UI for end users.
- Support saving the final playlist to a logged-in Spotify account.
- Provide a basic retrieval-augmented generation (RAG) step over a small
  lyrics corpus.
- Evaluate the system with reproducible ablation studies.

Out of scope:
- Multi-user collaboration, social features, or sharing.
- Production-grade scaling, authentication providers beyond Spotify, or
  payment.
- Real-time learning from user feedback during a single session.

## 3. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Strong LLM ecosystem and fast iteration. |
| App framework | Streamlit | Single-file UI, easy OAuth integration. |
| Database | SQLite with sqlite-vec extension | Zero setup, supports embeddings. |
| LLM provider | Anthropic Claude | Dev on Haiku 4.5; final eval on Sonnet 4.6. |
| Embeddings | OpenAI text-embedding-3-small | Inexpensive, sufficient quality. |
| External APIs | Spotify Web API, Genius API | Catalog access and lyrics. |
| Deployment | Streamlit Community Cloud | Free hosting for the demo. |

## 4. Configuration

The system reads its secrets from a `.env` file at the project root:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REDIRECT_URI` (default: `http://127.0.0.1:8501/callback`)
- `GENIUS_ACCESS_TOKEN`

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
  [Curator] (tools: spotify_search, genius_lyrics, vector_search)
        |
        v  (candidate pool of approximately 30 to 50 tracks)
  [Critic] -> filtered list, may request that the Curator re-runs
        |
        v  (final 15 to 25 tracks)
  [Sequencer] -> ordered playlist with arc summary
        |
        v
  saved to local DB, optionally pushed to the user's Spotify account
```

Orchestration is plain Python rather than an LLM framework like
LangGraph. Every agent input and output is logged to the `agent_traces`
table for inspection, caching, and evaluation.

## 6. Project layout

```
mood-playlist-curator/
  SPEC.md                         (this file)
  CLAUDE.md                       (working notes for the AI coding assistant)
  README.md
  .env.example
  .gitignore
  requirements.txt
  app.py                          (Streamlit entry point and UI)
  config.py                       (env vars, model names)
  db.py                           (SQLite setup, schema, helpers)
  spotify_client.py               (Spotify Web API wrapper)
  genius_client.py                (Genius lyrics fetcher)
  embeddings.py                   (OpenAI embedding wrapper)
  agents/
    __init__.py
    base.py                       (shared agent runner: JSON parsing, tracing)
    mood_interpreter.py
    curator.py
    critic.py
    sequencer.py
  orchestrator.py                 (runs the pipeline, handles feedback loop)
  rag/
    build_corpus.py               (one-time script: fetch lyrics, embed, store)
    retriever.py
  eval/
    test_set.json                 (mood prompts plus reference data)
    run_eval.py
    metrics.py
  data/
    app.db                        (gitignored)
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
- `spotify_search(query, limit)` returns tracks with audio features.
- `genius_lyrics(track_id)` returns lyrics text.
- `vector_search(query_text, k)` returns semantically similar tracks
  from the local corpus.

Output:
```json
{
  "candidates": [
    {
      "spotify_track_id": "string",
      "name": "string",
      "artist": "string",
      "rationale": "string",
      "audio_features": {},
      "lyrical_themes": ["string"]
    }
  ]
}
```
Target: 30 to 50 candidates.

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

Input: kept tracks with their audio features and rationales.

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

### 9.2 Day 2, RAG, Streamlit UI, and Spotify OAuth
1. `rag/build_corpus.py`: fetch lyrics for approximately 300 to 500
   popular tracks via Genius, embed with OpenAI, store in the
   `tracks` table.
2. `rag/retriever.py`: vector search over the corpus.
3. Wire the `vector_search` tool into the Curator.
4. `app.py` Streamlit application:
   - Spotify OAuth using the Authorization Code flow.
   - Mood input box, "Generate" button, playlist rendered with
     explanations.
   - "Save to Spotify" button.
   - Arc summary displayed.
5. Multi-turn refinement: a follow-up text input that re-runs the
   pipeline with the previous playlist and the refinement instruction
   in context.

Milestone: a user with a Spotify account can log in and generate a
saveable playlist through the web UI.

### 9.3 Day 3, Evaluation, polish, and writeup
1. `eval/test_set.json`: 15 mood prompts with reference data.
2. `eval/metrics.py`: `mood_fit`, `artist_diversity`, `genre_diversity`,
   `theme_coverage`, `forbidden_genre_rate`, `explanation_quality`.
3. `eval/run_eval.py`: ablation studies:
   - Full pipeline versus a single-agent baseline.
   - With-RAG versus without-RAG.
   - Haiku versus Sonnet.
4. Generate matplotlib plots of the results.
5. `README.md` with setup instructions, screenshots, and results.
6. Deploy to Streamlit Community Cloud.
7. Final written report.

## 10. Evaluation methodology

Metrics:
- `mood_fit`: 1 minus cosine distance between the target audio profile
  and the mean track features.
- `artist_diversity`: unique artists divided by total tracks.
- `genre_diversity`: unique genres divided by total tracks.
- `theme_coverage`: fraction of expected themes mentioned in
  the generated explanations.
- `forbidden_genre_rate`: fraction of tracks in `should_avoid_genres`.
- `explanation_quality`: a heuristic combining length, mention of audio
  features, and mention of lyrics, optionally supplemented by human
  rating on a sample.

For each ablation, all 15 prompts are run and the metrics are averaged.
Results are presented as bar charts comparing configurations.

## 11. Implementation rules

- Every LLM JSON output is validated with a Pydantic model.
- LLM responses are cached by `hash(agent_name + input_json)` during
  development to save cost. The cache resides in `agent_traces`; the
  pipeline performs a cache lookup before each Anthropic call.
- Every agent invocation, successful or failed, is logged to
  `agent_traces`.
- The default development model is `claude-haiku-4-5-20251001`. The
  final evaluation run uses `claude-sonnet-4-6`.
- Functions are kept small and explicit; abstraction is added only when
  required by concrete code.
- Secrets (`.env`) and the local database (`data/app.db`) are excluded
  from version control.

## 12. Operational guardrails

Certain actions are gated to avoid surprises and to bound cost:
- Creation of the Spotify Developer app and tester whitelisting are
  performed manually by the developer through the Spotify dashboard.
- Public deployment to Streamlit Cloud is performed manually.
- Any LLM run with an estimated cost greater than one US dollar
  (for example, a full evaluation on Sonnet) requires explicit
  confirmation before kicking off.
- Installation of system-level dependencies requires explicit
  confirmation.

## 13. Manual setup prerequisites

The following steps are performed by the developer outside of the codebase:
1. Create a Spotify Developer application at
   developer.spotify.com/dashboard.
2. Register the redirect URI on the Spotify application's settings page.
3. Whitelist tester Spotify email addresses on the Spotify dashboard
   (Development Mode caps at 25 users).
4. Obtain a Genius API token from genius.com/api-clients.
5. Obtain an Anthropic API key from console.anthropic.com.
6. Obtain an OpenAI API key from platform.openai.com.
7. Push the project to GitHub and connect the repository to Streamlit
   Cloud for deployment.
