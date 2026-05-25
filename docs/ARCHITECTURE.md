# Mood-Aware Playlist Curator — Architecture

A reference document covering what the system is, why each piece exists,
and how the pieces fit together. Read this when you want to refresh your
mental model.

---

## 1. What the system does

Takes a natural-language mood (e.g. *"rainy sunday, melancholy but hopeful"*)
and produces a personalized Spotify playlist with a per-track explanation
and a 2-3 sentence "emotional arc" summary.

The interesting design choice: instead of one giant LLM call that does
everything, we use a **multi-agent pipeline** where four small, focused
agents each handle one job.

---

## 2. The pipeline

```
  user mood text
        |
        v
  [Mood Interpreter]   ->  structured mood profile
        |
        v
  [Curator]            ->  ~40 candidate tracks
   (uses tools: spotify_search, genius_lyrics, vector_search)
        |
        v
  [Critic]             ->  ~20 filtered tracks
   (can ask the Curator to retry with feedback, max 1 retry)
        |
        v
  [Sequencer]          ->  ordered playlist + arc summary
        |
        v
  saved to DB, rendered in UI, optionally pushed to Spotify
```

Each arrow carries **structured JSON**, validated by Pydantic. If any agent
returns malformed output, the runner retries that agent (up to 2x) with the
error fed back to the model. If all retries fail, the pipeline fails loudly.

---

## 3. Why split into four agents

Four reasons stacked together:

1. **Focused prompts produce better output.** LLMs perform much better on
   narrow tasks than on broad multi-step ones. *"Judge these 40 tracks
   against this mood profile"* beats *"do everything from mood to playlist."*
2. **Validation at every boundary.** Pydantic catches a malformed mood
   profile before it ever reaches the Curator. We don't propagate garbage.
3. **Different agents have different needs.** The Curator needs Spotify
   tools; the others don't. The Critic can ask the Curator to re-run with
   feedback. A mega-agent can't do these loops cleanly.
4. **Caching pays off.** When you tweak the Sequencer prompt, the upstream
   three agents serve from cache. With a mega-agent, every prompt edit
   re-runs everything.

Cost: more code, more API calls per run. Accepted for a course project
where we care about observability and quality.

---

## 4. Agent contracts

Every agent has a strict input/output contract enforced by Pydantic.

### Mood Interpreter
Input:
```json
{ "mood_text": "rainy sunday, melancholy", "user_taste_profile": null }
```
Output:
```json
{
  "emotions": ["melancholy", "hopeful"],
  "energy": 0.3,
  "valence": 0.4,
  "tempo_range": [60, 90],
  "themes": ["rain", "solitude"],
  "genres_favor": ["folk", "indie"],
  "genres_avoid": ["edm"],
  "context": "background reading"
}
```

### Curator
Input: mood profile + user taste profile.
Tools: `spotify_search`, `genius_lyrics` (Day 2), `vector_search` (Day 2).
Output:
```json
{
  "candidates": [
    {
      "spotify_track_id": "...",
      "name": "...",
      "artist": "...",
      "rationale": "...",
      "audio_features": { },
      "lyrical_themes": [ ]
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
  "rejected": [{"spotify_track_id": "...", "reason": "..."}],
  "global_warnings": [ ],
  "request_more": false,
  "feedback_for_curator": null
}
```
If `request_more` is true, the orchestrator re-runs the Curator with
`feedback_for_curator` appended. Maximum 1 retry to bound latency and cost.

### Sequencer
Input: kept tracks (with audio features).
Output:
```json
{
  "ordered_tracks": [
    { "spotify_track_id": "...", "position": 1,
      "transition_note": "why this follows the previous" }
  ],
  "arc_summary": "2 to 3 sentences describing the emotional shape"
}
```

---

## 5. The shared agent runner (agents/base.py)

All four agents call into one function: `run_agent(...)`. Each agent
differs only in its **system prompt**, its **input shape**, its **output
shape**, and whether it uses **tools**. Everything else (API call, JSON
parsing, retry, caching, logging) is shared.

### What run_agent does, top to bottom

1. **Hash** the input and check `agent_traces` for a previous run with the
   same hash. If found, return that cached output, skip the API call.
2. **Call Claude** with the agent's system prompt + the user input.
3. If the agent has tools: **run the tool loop** (see below). Otherwise
   parse the response directly.
4. **Validate** the final response with the agent's Pydantic output model.
5. If validation fails, **retry up to 2 times** with the error fed back to
   the model as the next user message. If retries exhausted, raise.
6. **Save a trace row** to `agent_traces`: input, output, tokens used,
   latency, timestamp.

### Why each piece exists

**JSON instead of free text.** Our code is the consumer, not a human. The
orchestrator needs to pass typed data between agents.

**Pydantic validation.** *Valid JSON is not the same as correct JSON.*
A model might return `"emotions": "melancholy"` (string) when we asked for
a list of strings. Pydantic catches this at the boundary, with a clear
error message naming the bad field. We never propagate bad data to the
next agent.

**Retry with error feedback.** When validation fails, we feed the parser's
error back to Claude in a new user message. LLMs are very good at fixing
their own mistakes when shown the specific error. Bare retries without
feedback recover ~30-40% of failures. Retries *with* the error included
recover ~90%+.

**Caching keyed by `(agent_name, hash(user_input))`.** During dev you'll
run the pipeline 50-100 times. Same inputs should not produce paid API
calls every time. The cache key deliberately excludes the system prompt
so prompt tweaks don't invalidate everything. Tradeoff: when you change
a system prompt, you have to bypass the cache (`use_cache=False`) or
wipe `agent_traces`, or the model will silently return the old output.

**Trace logging.** Same data, two uses: (1) **caching** — skip duplicates.
(2) **debugging** — when a playlist is bad, you can read the exact input
and output of each agent and pinpoint which step broke. Free side effects
of the same table: token counts (cost tracking) and latency (perf
tracking).

### The tool loop (Curator only)

The LLM does not run code. It can only ask. The Curator needs to actually
search Spotify, so we declare a `spotify_search` tool. The choreography:

```
1. our code -> Anthropic API:  "here's the mood profile, here are your tools"
2. Anthropic API -> our code:  "I want spotify_search(query='rainy day', limit=10)"
                                (stop_reason = tool_use)
3. our code -> Spotify API:    GET /v1/search?q=rainy+day&limit=10
4. Spotify API -> our code:    [ list of tracks ]
5. our code -> Anthropic API:  "here are the results"
6. Anthropic API -> our code:  final JSON answer
                                (stop_reason = end_turn)
```

Three round trips, two different APIs. Our Python code is the middleman.
Claude never touches Spotify; Spotify never knows Claude exists.

The loop is capped at 8 tool iterations so a confused model can't burn
money in an infinite loop.

---

## 6. The two kinds of "energy"

A common point of confusion. The same word means two different things
depending on where the number came from.

| Number | Source | How it was produced |
|---|---|---|
| `target.energy = 0.3` | Mood Interpreter (Claude) | LLM mapping the words "rainy sunday melancholy" onto a 0 to 1 scale. Language to numbers translation. No audio involved. |
| `track.energy = 0.27` ("Skinny Love") | Spotify audio analysis | Spotify's signal-processing code analyzes the actual MP3 (tempo, loudness, timbre, frequency content) and outputs a single number. Computed once when the track is uploaded. |

The Critic compares the two and says "0.27 is close to 0.3 -> good fit."

The LLM does the language work. Spotify does the audio work. We glue them
together.

**Heads up:** Spotify deprecated the `/audio-features` endpoint for new
apps in 2024. If our app can't access it, we lose the right-hand column.
The agents then reason from track names, artists, and the LLM's general
knowledge of artists and songs (Bon Iver = melancholy folk, etc.). Less
precise, still functional.

---

## 7. The database (db.py, SQLite)

Single file at `data/app.db`. No server, zero setup. Good enough for a
course project.

Tables:

| Table | Purpose |
|---|---|
| `users` | Spotify login info (Day 2 OAuth) |
| `playlists` | One row per generated playlist (mood prompt, arc summary) |
| `playlist_tracks` | Tracks of each playlist with per-track explanation |
| `tracks` | Track metadata cache: audio features, lyrics, embeddings (RAG corpus) |
| **`agent_traces`** | Every single LLM call: input, output, tokens, latency |
| `taste_profile` | A user's listening preferences for personalization |
| `feedback` | User actions (skip, save, thumbs up/down) |
| `eval_runs`, `eval_results` | Day 3 evaluation runs |

`agent_traces` is the most important table. It does two jobs:
1. **Cache** — `find_cached_trace(agent_name, input_hash)` looks up a prior
   row before each call.
2. **Debugging / observability** — every input and output is recorded for
   inspection, plus token counts and latency.

---

## 8. The Spotify client (spotify_client.py)

Wraps the Spotify Web API. Two methods we use:

- `search(query, limit) -> list[dict]` — searches the public catalog, returns
  tracks with id, name, artist, album. Called by the Curator's
  `spotify_search` tool.
- `audio_features(track_ids) -> list[dict | None]` — Spotify's per-track
  numerical features (energy, valence, tempo, etc.). Used by the Critic
  and Sequencer. May return None per track if the endpoint is blocked.

Auth: **Client Credentials flow** for Day 1 (no user login needed for
public catalog access). **Authorization Code (OAuth)** added Day 2 for
saving playlists to a user's real account.

---

## 9. The orchestrator (orchestrator.py)

Plain Python, not a framework. Runs the four agents in sequence:

```
profile = mood_interpreter.run(mood_text, taste_profile)
candidates = curator.run(profile, taste_profile)
critique = critic.run(profile, candidates)
if critique.request_more:
    candidates += curator.run(profile, taste_profile,
                              feedback=critique.feedback_for_curator)
    critique = critic.run(profile, candidates)
kept = [c for c in candidates if c.id in critique.kept]
playlist = sequencer.run(kept)
save_playlist(...)
```

That's it. Every agent call is automatically traced. Every agent's output
is automatically validated. Every agent's input automatically hits the
cache first.

---

## 10. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | LLM ecosystem; fast iteration |
| UI | Streamlit | Single-file UI, handles OAuth easily |
| DB | SQLite + sqlite-vec | Zero setup, vector search for free |
| LLM | Anthropic Claude (Haiku dev, Sonnet final eval) | Tool use, JSON output, cost balance |
| Embeddings | OpenAI text-embedding-3-small | Cheap, good enough |
| External | Spotify Web API, Genius API | Catalog + lyrics |
| Deploy | Streamlit Community Cloud | Free |

---

## 11. Build plan recap

Day 1: CLI end-to-end (no UI yet). Scaffold, db, spotify client, base
runner, four agents, orchestrator, smoke test.

Day 2: RAG corpus + Streamlit UI + Spotify OAuth.

Day 3: Evaluation harness + ablations + plots + writeup + deploy.

---

## 12. Key principles to keep in mind

- **Don't trust LLM output.** Validate at every boundary.
- **Show the model its mistakes.** Don't retry blind — feed the error back.
- **Cache aggressively during dev, bypass for final eval.**
- **Trace everything.** Same data debugs problems and powers caching.
- **The LLM is a reasoning controller, not an executor.** It decides; we run.
- **Each agent has a narrow contract.** That's where the quality comes from.
