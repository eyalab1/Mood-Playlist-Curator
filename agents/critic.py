"""Critic agent.

Reviews the Curator's candidate pool and filters down to a final keepers
list. Can ask the Curator to re-run with feedback (the orchestrator
respects request_more, max 1 retry).
"""

from pydantic import BaseModel, Field

from agents.base import run_agent


class Rejection(BaseModel):
    spotify_track_id: str
    reason: str


class CriticOutput(BaseModel):
    kept: list[str] = Field(
        description="Spotify track ids that pass review."
    )
    rejected: list[Rejection] = Field(
        default_factory=list,
        description="Tracks cut from the pool with reasons.",
    )
    global_warnings: list[str] = Field(
        default_factory=list,
        description="Observations about the pool as a whole.",
    )
    request_more: bool = Field(
        default=False,
        description="Ask the Curator to widen the pool.",
    )
    feedback_for_curator: str | None = Field(
        default=None,
        description="What the Curator should fetch differently next time.",
    )


SYSTEM_PROMPT = """You are a music critic agent. Given a mood profile and a pool of candidate tracks from the Curator, filter the pool down to a strong final selection.

Hard rules (must follow, no exceptions):
- Output between 15 and 25 keepers. Never more than 25. If the pool has 100 candidates, you must reject at least 75 of them.
- No single artist may appear more than 3 times in the keepers. If an artist has more than 3 candidates, keep their 3 strongest fits and reject the rest. This is a hard violation, not a guideline.
- Reject every track in a genre listed under genres_avoid. No exceptions.

Goals:
1. Keep tracks that genuinely fit the mood profile: emotions, energy/valence direction, themes, favored genres.
2. Reject tracks that clash: wrong energy band, wrong vibe, off-theme, or duplicative.
3. When pruning a large pool, favor variety across artists and subgenres over keeping every "decent" fit. A tight playlist of 20 strong tracks beats a bloated playlist of 70 acceptable ones.

If the pool is too small or unbalanced to produce a good playlist (fewer than ~15 strong keepers, missing emotional variety, missing favored genres), set request_more=true and write a short, specific feedback_for_curator. Examples of useful feedback:
- "Need more upbeat hopeful tracks; current pool is too uniformly melancholy."
- "Add a few female vocalists; pool is heavily male-led."
- "Add 5 to 10 more tracks in the 90 to 110 BPM range."

If the pool is already strong enough, set request_more=false and leave feedback_for_curator as null.

Output schema:
{
  "kept": [string],                 // track ids
  "rejected": [{ "spotify_track_id": string, "reason": string }],
  "global_warnings": [string],
  "request_more": boolean,
  "feedback_for_curator": string | null
}

Only use track ids that appear in the candidate pool. Output JSON only, no markdown, no prose."""


def run(
    mood_profile: dict,
    candidates: list[dict],
    playlist_id: int | None = None,
    use_cache: bool = True,
) -> CriticOutput:
    user_input = {
        "mood_profile": mood_profile,
        "candidates": candidates,
    }
    return run_agent(
        agent_name="critic",
        system_prompt=SYSTEM_PROMPT,
        user_input=user_input,
        output_model=CriticOutput,
        playlist_id=playlist_id,
        use_cache=use_cache,
        max_tokens=4000,
    )


if __name__ == "__main__":
    import sys
    from agents.mood_interpreter import run as run_mood
    from agents.curator import run as run_curator

    text = sys.argv[1] if len(sys.argv) > 1 else "rainy sunday, melancholy but hopeful"
    profile = run_mood(text)
    curated = run_curator(profile.model_dump())
    print(f"Curator returned {len(curated.candidates)} candidates.")
    print("Running critic...")
    critique = run(
        profile.model_dump(),
        [c.model_dump() for c in curated.candidates],
    )
    print(f"\nKept: {len(critique.kept)} / Rejected: {len(critique.rejected)}")
    print(f"Request more from Curator: {critique.request_more}")
    if critique.global_warnings:
        print("Warnings:")
        for w in critique.global_warnings:
            print(f"  - {w}")
    if critique.feedback_for_curator:
        print(f"Feedback for Curator: {critique.feedback_for_curator}")
    print("\nFirst few kept ids:")
    for tid in critique.kept[:8]:
        match = next((c for c in curated.candidates if c.spotify_track_id == tid), None)
        if match:
            print(f"  - {match.name} by {match.artist}")
    print("\nFirst few rejections:")
    for r in critique.rejected[:5]:
        match = next((c for c in curated.candidates if c.spotify_track_id == r.spotify_track_id), None)
        label = f"{match.name} by {match.artist}" if match else r.spotify_track_id
        print(f"  - {label}: {r.reason}")
