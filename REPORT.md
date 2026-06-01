# Mood-Aware Playlist Curator, Final Report

**Author:** Eyal Abisdris
**Course:** LLM-Augmented Software Practice
**Instructor:** Mikael Gorsky
**Date:** June 2026

**Live demo:** link shared privately with the instructor (no login required; first load can take ~50s while the free host wakes up)
**Repository:** https://github.com/eyalab1/Mood-Playlist-Curator


## 1. Executive Summary

This project is a web application that turns a natural-language mood ("rainy sunday, melancholy but hopeful") into a sequenced Spotify playlist, using a four-agent LLM pipeline (Mood Interpreter, Curator, Critic, Sequencer) and a lyrics-based RAG step over a curated 347-song corpus. The system was evaluated on 15 prompts across two configurations (with RAG and without RAG). The headline finding is calibrated rather than dramatic: **RAG modestly improves theme coverage (+0.074) and lyric-mood alignment (+0.029), with essentially no change to artist diversity or explanation quality, at the cost of lower cross-prompt variety (80% unique tracks vs 93% without RAG).** The full ablation, methodology, and limitations are documented below.


## 2. Problem and Approach

Matching music to a feeling expressed in plain language is genuinely hard. Keyword search (the only thing Spotify's public API offers easily) matches song *titles* and *artist names*, not what songs are about. A query for "lonely highway" returns songs literally titled that, and misses lyrically lonely classics like "Wichita Lineman".

The approach taken here is a **multi-agent LLM pipeline** in which each agent has a focused role and a strictly-validated structured output, plus a **lyrics-based RAG** step that lets the Curator search a curated corpus by *meaning of the lyrics*, not by titles. The agents are orchestrated by plain Python (no LangGraph), every input and output is logged for inspection and reuse, and the result is rendered in a polished web UI.


## 3. Architecture

```
  user mood text
        |
        v
  [Mood Interpreter]      structured profile (emotions, energy, valence, themes, genres_favor/avoid, context)
        |
        v
  [Curator]               tools: spotify_search (all of Spotify, by keyword)
                                 vector_search  (lyrics corpus, by meaning)
        |
        v   pool of ~30 to 50 candidates
  [Critic]                filters to a final 15 to 25, can request one Curator re-run with feedback
        |
        v
  [Sequencer]             orders kept tracks into an emotional arc + transition notes
        |
        v
  Web UI (FastAPI + static HTML/Tailwind) renders the playlist with live Spotify embed players
```

Each agent is an Anthropic Claude call with a focused system prompt and a Pydantic schema for its output. The Critic can ask the Curator to widen the pool once (one retry) by sending back free-form feedback. Every call is content-addressed cached and logged to an `agent_traces` table.

### Tech stack used (actual)
- Python 3.11+, FastAPI + Uvicorn backend, single static HTML/Tailwind frontend
- Anthropic Claude (Haiku 4.5 for dev and eval, the model used for everything in this report)
- Voyage AI `voyage-3.5` embeddings, numpy cosine similarity for vector search
- SQLite for persistence, including the RAG corpus and full agent traces
- Spotify Web API (client credentials) for catalog search
- Genius API for one-time corpus lyric fetching
- Deployed on Render free tier


## 4. Design Decisions and Deviations from the Original Spec

The original spec evolved during the build. Each change below was a deliberate choice driven by a real-world constraint, and is recorded honestly rather than rewritten over.

| Original plan | Final reality | Reason |
|---|---|---|
| Streamlit single-file UI | FastAPI backend + custom HTML/Tailwind frontend | Streamlit could not hit the visual polish required; FastAPI also serves the page on Render. |
| OpenAI `text-embedding-3-small` | Voyage AI `voyage-3.5` | OpenAI required a $5 prepay; Voyage has a 200M-token free tier that more than covers this project. |
| `sqlite-vec` extension | numpy cosine over ~347 vectors | Brute force is instant at this scale; one fewer compiled dependency. |
| Streamlit Community Cloud | Render web service | FastAPI cannot run on Streamlit Cloud. |
| Spotify OAuth + "save to Spotify" | Embedded Spotify players in the UI | OAuth was the riskiest, most time-consuming piece; embeds give a strong demo without it. Documented as future work. |
| 3 Curator tools (spotify_search, genius_lyrics, vector_search) | 2 (spotify_search, vector_search) | Genius is used only at corpus-build time; exposing it live offered marginal value. |
| `mood_fit` from Spotify audio features | `mood_fit` from lyric embeddings | Spotify deprecated `/audio-features` for new apps. |
| `genre_diversity`, `forbidden_genre_rate` from Spotify genres | dropped | Spotify silently restricted the artist `genres` field for new apps; verified empirically (Taylor Swift returns `None`). No reliable replacement source within the project's time budget. |
| 2 to 3 day timeline | ~10 days | Lecturer's stated milestones (beta this week, final next week) were the binding constraints, not the original aspirational sprint. |

These deviations are described again at the points where they matter (architecture, evaluation).


## 5. Implementation Details

A few engineering choices are worth calling out because they shaped both reliability and the evaluation:

- **Pydantic-validated structured outputs everywhere.** Every agent declares its output type. The shared `run_agent` wrapper extracts JSON from the model's response and validates it; on a parse or schema failure, it retries with the error fed back to the model as the next user message (up to 2 retries). This made the pipeline robust enough that a single course-project budget produced consistent results across 30 evaluation runs.
- **Content-addressed caching.** Every Anthropic call is keyed by `sha256(agent_name + canonical_json(input))`. Re-running an evaluation, or generating the same mood twice on the live site, returns cached results for free. The cache lives in the same `agent_traces` table that powers tracing, so there is no separate cache layer.
- **Tool-use loop.** The Curator's tool calls are handled by a small loop in `agents/base.py`. The loop has a hard iteration cap (15) so a misbehaving agent cannot run away. The cap was tuned upward (from 8) to accommodate the without-RAG ablation, which asks for more Spotify searches to compensate for the missing corpus tool.
- **Critic feedback loop.** If the Critic believes the Curator's pool is too narrow or unbalanced, it sets `request_more=true` and provides natural-language feedback. The orchestrator then calls the Curator again with that feedback appended. This loop is capped at one retry to bound latency and cost.
- **Resumable corpus build.** Building the 347-song RAG corpus involves Spotify searches, Genius lyric fetches, and Voyage embeddings. The builder writes each step to disk so a network failure halfway through is recoverable; on rerun it skips what is already done.
- **Throttled embeddings (then removed).** The Voyage free tier without a payment method throttles to 10,000 tokens/min. The embedder was written to respect this with batching and pauses; after a payment method was added, the throttle was relaxed but the safety retry/backoff stays as a guard.


## 6. Evaluation

### 6.1 Methodology

The system was evaluated as a **paired ablation**: the same 15 prompts were run through two configurations of the pipeline, and the same metrics were applied to the resulting playlists.

- **with_rag (the full system):** the Curator is given both `spotify_search` and `vector_search` tools, and uses a system prompt that recommends starting with the corpus.
- **without_rag (baseline):** the Curator is given only `spotify_search`, and a slightly different system prompt that asks it to widen the keyword net to compensate.

Each configuration produced one playlist per prompt, the playlist was scored on the metrics described below, and the scores were averaged across the 15 prompts. All results are saved in the `eval_runs` and `eval_results` tables so the run is fully reproducible.

### 6.2 Test set

`eval/test_set.json` contains 15 prompts. They were written deliberately in varied human registers, ranging from terse ("leg day at the gym, need stuff that makes me want to lift a car") to rambling ("found a box of old photos from high school today and now im in my feels about how fast it all went"), with intentional dropped apostrophes, lowercase, slang ("kinda lonely ngl"), and the occasional question mark. This is an intentional choice: a test set written in one polished voice tends to flatter the system. Each prompt carries `expected_themes` and `should_avoid_genres` reference data. The full set spans heartbreak, workout, focus, party, nostalgia, rainy melancholy, rage, romance, road trip, anxiety, hope, loneliness, chill, confidence, and bittersweet goodbye.

### 6.3 Metrics

Four metrics are computed per playlist. Two further metrics from the original spec were dropped (see Section 4) because they relied on Spotify endpoints that have been restricted for new apps.

- **`artist_diversity`** = unique artists / total tracks. Higher is more varied.
- **`theme_coverage`** = fraction of the prompt's `expected_themes` mentioned (case-insensitive substring) across the playlist's track explanations. Measures whether the playlist *explains itself in the language of the mood*.
- **`explanation_quality`** = heuristic average of three signals: average explanation length / 80 chars (capped at 1.0), fraction of explanations referencing lyric vocabulary, fraction referencing musical vocabulary. A proxy for "substantive" explanations.
- **`mood_fit`** = average cosine similarity between the embedded mood text and each track's lyric embedding. **This metric is a substitute for the original spec's audio-feature-based metric**, which is no longer computable; the substitution is named as such in this report. Tracks lacking a lyric embedding are fetched (Genius + Voyage) on the fly during the evaluation so both arms are scored fairly.

In addition, the runner aggregates **cross-prompt unique-track counts** per configuration: total tracks across all prompts, and unique (name, artist) pairs. This makes the cost of a small corpus visible.

### 6.4 Results

Averaged across 15 prompts (n=14 for with_rag because one prompt hit Anthropic's per-minute rate limit and was excluded; the runner saves each result independently so this loss did not propagate):

| Metric | with_rag | without_rag | Delta (with - without) |
|---|---|---|---|
| artist_diversity | 0.817 | 0.827 | -0.011 |
| theme_coverage | **0.857** | 0.783 | **+0.074** |
| explanation_quality | 0.736 | 0.720 | +0.015 |
| mood_fit | **0.364** | 0.335 | **+0.029** |

Cross-prompt diversity:

| Config | Unique tracks / Total | % Unique |
|---|---|---|
| with_rag | 271 / 339 | **80%** |
| without_rag | 301 / 325 | **93%** |

Plots: `eval/plots/metric_ablation.png` and `eval/plots/cross_prompt_diversity.png`.

### 6.5 Interpretation

The result is a calibrated, honest finding, not a dramatic one:

- **RAG modestly helps on the two metrics that measure whether the playlist actually addresses the mood.** `theme_coverage` rose by 0.074 (about 9% relative) and `mood_fit` by 0.029. Both moved in the predicted direction, and consistently across the 15 prompts.
- **RAG did not meaningfully change** artist diversity within a single playlist or the heuristic explanation quality. The Curator writes substantive rationales in both modes, and the Critic enforces variety in both modes.
- **RAG did increase cross-prompt repetition.** The with_rag arm repeated tracks across semantically similar prompts (80% unique vs 93% without RAG). This is exactly the expected behavior of RAG over a small, finite corpus: similar moods retrieve overlapping corpus tracks. The keyword-only arm draws from a vastly larger catalog and therefore repeats less.

So the most honest one-line summary is: **for a small, curated corpus, RAG modestly improves alignment metrics at the cost of cross-prompt variety.** Whether the trade is worth it depends on what the system is optimized for; for a personal mood-driven recommender, the alignment improvement is more important than absolute novelty.


## 7. Limitations

Named explicitly so the next reader knows what *not* to conclude from this work.

- **Sample size (n=15).** Results indicate direction, not statistical significance. A larger test set would be needed for confidence intervals. Paired comparison on identical prompts mitigates some of this, but does not replace it.
- **Corpus size (347 songs).** Small by design (one-day budget for the corpus build, free Genius rate limits). It is the dominant cause of the cross-prompt repetition observed.
- **English-skewed corpus.** Genius had little Hebrew coverage; ~85 Hebrew seed tracks were skipped during the build. A Hebrew-aware RAG path would require an alternative lyric source.
- **Spotify API deprecations.** The original `audio_features` and `genres` endpoints have been restricted for new apps. Two metrics were dropped (Section 4). The substituted `mood_fit` is a defensible proxy but is not the same measurement.
- **No human ratings.** Metrics are heuristic and objective; a small human evaluation on a sample of playlists would have made the case for or against RAG stronger. Out of scope for this submission.
- **Model = Haiku only.** Cost ran higher than initially expected (~$4.50 total for the project, of which ~$4 was the eval). A Sonnet vs Haiku ablation, originally in the spec, was dropped to stay within budget. A targeted Sonnet rerun on a few prompts would be informative.
- **Free-host caveats.** The deployed Render service sleeps after 15 minutes of no traffic; the first request after a pause is slow.


## 8. Future Work

The pieces that would most strengthen this project, in rough priority order:

1. **Spotify OAuth + "save to Spotify".** Reinstates the original spec's headline feature. Scoped to whitelisted users in Spotify Development Mode.
2. **Larger and multilingual corpus.** 1500-2000 songs would meaningfully reduce cross-prompt repetition; a Hebrew-capable lyric source would broaden coverage.
3. **Human-rated explanation quality.** Replace or augment the current heuristic with ratings from a small panel.
4. **Sonnet vs Haiku ablation.** Quantify the cost-quality trade.
5. **Per-prompt result inspection.** The current plots show averages; per-prompt scatter plots would expose which mood types benefit from RAG and which do not.
6. **Persistent storage on deploy.** The free Render filesystem is ephemeral; switching to a managed Postgres preserves user-generated playlists across redeploys.


## 9. Conclusion

The project delivered a functioning multi-agent LLM system end to end: a four-agent pipeline with structured outputs, a lyrics-based RAG step, a polished deployed web UI, a real evaluation harness with a paired ablation, and a reproducible test set with metrics computed and plotted. The evaluation was constructed to be honest rather than flattering: dropped metrics were named, substitutions were named, the small sample size was named, and the result is a measured improvement rather than a hype-driven one. Treating "I built it" and "I measured it" as separate deliverables, both completed, is the point.


## Appendix A. File layout

```
mood-playlist-curator/
  server.py              FastAPI backend (serves the page, runs the pipeline)
  static/index.html      web UI (HTML + Tailwind CDN + vanilla JS)
  orchestrator.py        runs the 4-agent pipeline (with optional use_rag flag)
  agents/
    base.py              shared agent runner: tool loop, JSON parsing, tracing, cache
    mood_interpreter.py
    curator.py           with_rag and without_rag system prompts
    critic.py
    sequencer.py
  rag/
    seed_tracks.py       the 435-seed list, organized by mood
    build_corpus.py      one-time: fetch lyrics + embed + store
    retriever.py         vector_search over the corpus
  eval/
    test_set.json        15 mood prompts + reference data
    metrics.py           the four metric functions
    run_eval.py          the runner (--full for the real ablation)
    plot_results.py      generates the two PNGs in eval/plots/
  spotify_client.py
  genius_client.py
  embeddings.py
  db.py
  data/app.db            SQLite with the corpus + agent traces + eval results
```

## Appendix B. How to run locally

1. Python 3.11+; `python -m venv .venv` and activate it; `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, and (for corpus rebuild only) `GENIUS_ACCESS_TOKEN`.
3. The prebuilt corpus ships at `data/app.db`. To regenerate it: `python -m rag.build_corpus`.
4. Run the web app: `uvicorn server:app --port 8000`, then open `http://localhost:8000`.
5. Or use the CLI: `python -m orchestrator "rainy sunday, melancholy but hopeful"`.
6. Reproduce the evaluation: `python -m eval.run_eval --full` (cost ~$4 on Haiku, ~30-50 min), then `python -m eval.plot_results`.


## Appendix C. Cost notes

Total project spend: approximately **$4.50** of Anthropic credit, almost entirely consumed by the full evaluation (30 pipeline runs, ~$0.15 each on Haiku). Voyage embeddings remained on the free tier (~0.3M tokens used out of 200M). Spotify and Genius are free. Render hosting is free with the caveat that the service sleeps when idle.
