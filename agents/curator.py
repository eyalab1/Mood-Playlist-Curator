"""Curator agent.

Takes a mood profile and produces a candidate pool of 30 to 50 tracks using
two tools: spotify_search (keyword search over all of Spotify) and
vector_search (lyric-meaning search over our curated RAG corpus). The model
decides the queries; we run them.
"""

from pydantic import BaseModel, Field

from agents.base import run_agent
from spotify_client import get_client
from rag.retriever import vector_search


class CuratorCandidate(BaseModel):
    spotify_track_id: str
    name: str
    artist: str
    rationale: str = Field(description="Why this track fits the mood.")
    lyrical_themes: list[str] = Field(
        default_factory=list,
        description="Themes the curator believes the lyrics touch on.",
    )


class CuratorOutput(BaseModel):
    candidates: list[CuratorCandidate]


TOOLS = [
    {
        "name": "spotify_search",
        "description": (
            "Search Spotify's catalog for tracks. Returns up to `limit` "
            "tracks with id, name, artist, album, popularity. "
            "Run multiple searches to widen the candidate net (different "
            "queries pull from different parts of the catalog)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search terms. May be free-form (e.g. 'rainy day "
                        "acoustic'), or include Spotify search filters like "
                        "artist:'Bon Iver' or genre:'indie folk'."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "How many tracks to return (1 to 50).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "vector_search",
        "description": (
            "Search a curated corpus of ~350 songs by LYRIC MEANING, not by "
            "keywords. Returns tracks whose lyrics are semantically closest to "
            "your query text, with id, name, artist, genres, and a similarity "
            "score (higher is closer). Use this to surface songs that match "
            "the emotional meaning of the mood even when their titles do not "
            "contain the mood words. Results are ready to use as candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": (
                        "A short phrase describing the feeling or scene to "
                        "match against song lyrics, e.g. 'sleepless, anxious "
                        "thoughts at night'."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": "How many tracks to return (1 to 20).",
                    "default": 10,
                },
            },
            "required": ["query_text"],
        },
    },
]


def _dispatch(name: str, args: dict) -> object:
    if name == "spotify_search":
        client = get_client()
        return client.search(args["query"], min(int(args.get("limit", 10)), 50))
    if name == "vector_search":
        return vector_search(args["query_text"], min(int(args.get("k", 10)), 20))
    raise ValueError(f"unknown tool: {name}")


SYSTEM_PROMPT = """You are a music curator agent. Given a mood profile, build a candidate pool of 30 to 50 tracks that fit the mood, using two complementary tools.

Tools:
- vector_search(query_text, k): finds songs from a curated corpus by LYRIC MEANING. Best for capturing the emotional core of the mood, since it matches what songs are about, not just their titles. Start here.
- spotify_search(query, limit): keyword search over all of Spotify. Best for breadth, specific artists, specific genres, and filling out the pool beyond the corpus.

Process:
1. Call vector_search 2 to 3 times with phrases that capture the feeling of the mood (use the emotions and themes from the profile). Take the strongest lyric-matched songs.
2. Call spotify_search 3 to 5 times targeting different angles (themes, artists, subgenres) to widen the pool beyond the corpus. Wider nets beat one big search.
3. Combine results from both tools into one candidate pool. After you have enough, return them as JSON. Do not exceed 50.
4. Only include tracks whose IDs actually came back from a tool call (either tool). Never invent IDs.

For each candidate include:
- spotify_track_id (exact id from a search result)
- name, artist (from the same search result)
- rationale: 1 to 2 sentences on why this track fits the mood profile, referencing the emotions, themes, energy, or genres from the profile.
- lyrical_themes: 1 to 4 short labels for themes you expect the lyrics touch on, based on what you know about the song.

Aim for 30 to 50 candidates. Favor variety across artists and subgenres within the mood. Respect genres_avoid strictly.

If the user input includes a `feedback` field, treat it as critic feedback on a previous attempt and adjust accordingly.

Final output: a single JSON object matching:
{ "candidates": [ { spotify_track_id, name, artist, rationale, lyrical_themes }, ... ] }

No markdown, no prose outside the JSON object."""


# Variant used by the evaluation ablation when RAG is disabled. The Curator
# is given only spotify_search and is asked to widen the keyword net to
# compensate for the missing corpus tool.
SYSTEM_PROMPT_NO_RAG = """You are a music curator agent. Given a mood profile, build a candidate pool of 30 to 50 tracks that fit the mood, using Spotify keyword search only.

Tool:
- spotify_search(query, limit): keyword search over all of Spotify. Use this to find candidate tracks.

Process:
1. Call spotify_search 4 to 7 times targeting different angles of the mood (themes, artists, subgenres, eras). Wider nets beat one big search.
2. Combine results into one candidate pool. Do not exceed 50.
3. Only include tracks whose IDs actually came back from a spotify_search call. Never invent IDs.

For each candidate include:
- spotify_track_id (exact id from a search result)
- name, artist (from the same search result)
- rationale: 1 to 2 sentences on why this track fits the mood profile, referencing the emotions, themes, energy, or genres from the profile.
- lyrical_themes: 1 to 4 short labels for themes you expect the lyrics touch on, based on what you know about the song.

Aim for 30 to 50 candidates. Favor variety across artists and subgenres within the mood. Respect genres_avoid strictly.

If the user input includes a `feedback` field, treat it as critic feedback on a previous attempt and adjust accordingly.

Final output: a single JSON object matching:
{ "candidates": [ { spotify_track_id, name, artist, rationale, lyrical_themes }, ... ] }

No markdown, no prose outside the JSON object."""


def run(
    mood_profile: dict,
    user_taste_profile: dict | None = None,
    feedback: str | None = None,
    playlist_id: int | None = None,
    use_cache: bool = True,
    use_rag: bool = True,
) -> CuratorOutput:
    """Run the Curator. Pass use_rag=False to disable the vector_search tool
    (used by the evaluation ablation). The cache key differs between modes,
    so the two configurations do not collide."""
    user_input = {
        "mood_profile": mood_profile,
        "user_taste_profile": user_taste_profile,
        "feedback": feedback,
    }
    if use_rag:
        tools = TOOLS
        system_prompt = SYSTEM_PROMPT
        agent_name = "curator"
    else:
        tools = [t for t in TOOLS if t["name"] != "vector_search"]
        system_prompt = SYSTEM_PROMPT_NO_RAG
        agent_name = "curator_no_rag"
    return run_agent(
        agent_name=agent_name,
        system_prompt=system_prompt,
        user_input=user_input,
        output_model=CuratorOutput,
        tools=tools,
        tool_dispatcher=_dispatch,
        playlist_id=playlist_id,
        use_cache=use_cache,
        max_tokens=8000,
    )


if __name__ == "__main__":
    import sys
    from agents.mood_interpreter import run as run_mood
    text = sys.argv[1] if len(sys.argv) > 1 else "rainy sunday, melancholy but hopeful"
    profile = run_mood(text)
    print("Mood profile:")
    print(profile.model_dump_json(indent=2))
    print("\nRunning curator...")
    out = run(profile.model_dump())
    print(f"\nGot {len(out.candidates)} candidates:")
    for c in out.candidates[:8]:
        print(f"- {c.name} by {c.artist}: {c.rationale[:100]}...")
    if len(out.candidates) > 8:
        print(f"... and {len(out.candidates) - 8} more")
