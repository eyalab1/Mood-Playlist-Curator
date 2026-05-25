# Mood-Aware Playlist Curator

A web app that turns a natural-language mood description into a personalized
Spotify playlist with per-track explanations, using a multi-agent LLM
pipeline.

Course project. See `CLAUDE.md` for the full spec.

## Setup

1. Python 3.11+
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys.
4. Run the CLI smoke test:
   ```
   python -m orchestrator "rainy sunday, melancholy but hopeful"
   ```

## Architecture

Multi-agent pipeline:

```
mood text -> Mood Interpreter -> Curator -> Critic -> Sequencer -> playlist
```

Each agent is a separate Anthropic API call. All inputs and outputs are
logged to `agent_traces` for tracing and evaluation.
