"""Sequencer agent.

Takes the Critic's kept tracks and orders them into a final playlist with
an emotional arc and per-transition notes.
"""

from pydantic import BaseModel, Field

from agents.base import run_agent


class OrderedTrack(BaseModel):
    spotify_track_id: str
    position: int
    transition_note: str = Field(
        description="Why this track follows the previous one. "
        "For position 1, an opening statement that sets the tone."
    )


class SequencerOutput(BaseModel):
    ordered_tracks: list[OrderedTrack]
    arc_summary: str = Field(
        description="2 to 3 sentences describing the emotional shape of the playlist."
    )


SYSTEM_PROMPT = """You are a playlist sequencer agent. Given a mood profile and a list of kept tracks, order them into a sequence that builds a cohesive emotional arc.

Goals:
1. Decide a shape for the arc that fits the mood (gradual brightening, slow descent, U-shape, plateau then lift, etc.). State the chosen shape in arc_summary.
2. Order tracks so adjacent transitions feel natural. Avoid jarring shifts in energy or tempo unless intentional.
3. Spread tracks by the same artist; do not put two tracks by the same artist back to back unless the segue is deliberate.
4. Open with a track that establishes the dominant emotional tone. End with a track that resolves or completes the arc.

For each ordered track include:
- spotify_track_id (must be one of the kept ids; do not invent)
- position (1-indexed, contiguous)
- transition_note: 1 short sentence. For position 1, an opening line that sets the tone. For position 2 onward, why this track follows the previous (energy shift, thematic continuity, etc.).

Then provide arc_summary: 2 to 3 sentences describing the emotional shape.

Use every kept track exactly once. Output JSON only, no markdown or prose."""


def run(
    mood_profile: dict,
    kept_tracks: list[dict],
    playlist_id: int | None = None,
    use_cache: bool = True,
) -> SequencerOutput:
    user_input = {
        "mood_profile": mood_profile,
        "kept_tracks": kept_tracks,
    }
    return run_agent(
        agent_name="sequencer",
        system_prompt=SYSTEM_PROMPT,
        user_input=user_input,
        output_model=SequencerOutput,
        playlist_id=playlist_id,
        use_cache=use_cache,
        max_tokens=6000,
    )


if __name__ == "__main__":
    import sys
    from agents.mood_interpreter import run as run_mood
    from agents.curator import run as run_curator
    from agents.critic import run as run_critic

    text = sys.argv[1] if len(sys.argv) > 1 else "rainy sunday, melancholy but hopeful"
    profile = run_mood(text)
    curated = run_curator(profile.model_dump())
    candidates_dump = [c.model_dump() for c in curated.candidates]
    critique = run_critic(profile.model_dump(), candidates_dump)
    kept = [c for c in candidates_dump if c["spotify_track_id"] in critique.kept]
    print(f"Sequencing {len(kept)} tracks...")
    seq = run(profile.model_dump(), kept)
    print("\nArc:")
    print(f"  {seq.arc_summary}\n")
    print("Playlist:")
    by_id = {c["spotify_track_id"]: c for c in kept}
    for ot in seq.ordered_tracks:
        meta = by_id.get(ot.spotify_track_id)
        if not meta:
            continue
        print(f"  {ot.position:2}. {meta['name']} by {meta['artist']}")
        print(f"      -> {ot.transition_note}")
