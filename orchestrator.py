"""End-to-end pipeline orchestrator.

Runs Mood Interpreter -> Curator -> Critic (with optional retry) -> Sequencer,
persists the result, and returns the final playlist. Plain Python, no
framework. Every agent call is traced and validated by agents/base.py.

CLI:
    python -m orchestrator "rainy sunday, melancholy but hopeful"
"""

import sys
from typing import Callable

from db import init_db, save_playlist
from agents import mood_interpreter, curator, critic, sequencer


MAX_CRITIC_RETRIES = 1


def generate_playlist(
    mood_text: str,
    user_taste_profile: dict | None = None,
    persist: bool = True,
    use_cache: bool = True,
    on_step: Callable[[str], None] | None = None,
    use_rag: bool = True,
) -> dict:
    """Run the four-agent pipeline. Returns the final playlist as a dict.

    on_step, if given, is called with a short label before each agent runs.
    The UI uses it to show live progress; the CLI leaves it None.

    use_rag toggles the Curator's vector_search tool. Defaults to True; the
    evaluation ablation calls with use_rag=False to compare configurations.
    """

    def step(label: str) -> None:
        if on_step is not None:
            on_step(label)

    step("Interpreting your mood")
    profile = mood_interpreter.run(
        mood_text, user_taste_profile, use_cache=use_cache
    )
    profile_dump = profile.model_dump()

    step("Curating candidate tracks")
    curated = curator.run(
        profile_dump, user_taste_profile, use_cache=use_cache, use_rag=use_rag
    )
    candidates = list(curated.candidates)

    step("Critiquing the pool")
    critique = critic.run(
        profile_dump,
        [c.model_dump() for c in candidates],
        use_cache=use_cache,
    )

    retries = 0
    while critique.request_more and retries < MAX_CRITIC_RETRIES:
        retries += 1
        extra = curator.run(
            profile_dump,
            user_taste_profile,
            feedback=critique.feedback_for_curator,
            use_cache=use_cache,
            use_rag=use_rag,
        )
        # Avoid duplicate ids when extending the pool.
        seen = {c.spotify_track_id for c in candidates}
        for c in extra.candidates:
            if c.spotify_track_id not in seen:
                candidates.append(c)
                seen.add(c.spotify_track_id)
        critique = critic.run(
            profile_dump,
            [c.model_dump() for c in candidates],
            use_cache=use_cache,
        )

    # Safety cap: even if the Critic returns too many, never exceed 25.
    MAX_KEEPERS = 25
    kept_ids_list = critique.kept[:MAX_KEEPERS]
    kept_ids = set(kept_ids_list)
    # Preserve the Critic's ordering of kept ids when slicing the pool.
    by_id_lookup = {c.spotify_track_id: c for c in candidates}
    kept = [by_id_lookup[i] for i in kept_ids_list if i in by_id_lookup]
    if not kept:
        raise RuntimeError(
            "Pipeline produced zero kept tracks. Check Critic output."
        )

    step("Sequencing the emotional arc")
    seq = sequencer.run(
        profile_dump,
        [c.model_dump() for c in kept],
        use_cache=use_cache,
    )

    by_id = {c.spotify_track_id: c for c in kept}
    final_tracks = []
    for ot in seq.ordered_tracks:
        meta = by_id.get(ot.spotify_track_id)
        if not meta:
            # Sequencer returned an id that wasn't in kept. Skip rather than crash.
            continue
        final_tracks.append(
            {
                "spotify_track_id": ot.spotify_track_id,
                "track_name": meta.name,
                "artist": meta.artist,
                "explanation": meta.rationale,
                "transition_note": ot.transition_note,
                "position": ot.position,
            }
        )

    playlist_id = None
    if persist:
        playlist_id = save_playlist(
            mood_prompt=mood_text,
            arc_summary=seq.arc_summary,
            tracks=final_tracks,
        )

    return {
        "playlist_id": playlist_id,
        "mood_text": mood_text,
        "profile": profile_dump,
        "arc_summary": seq.arc_summary,
        "global_warnings": critique.global_warnings,
        "tracks": final_tracks,
        "critic_retries": retries,
    }


def _print_playlist(result: dict) -> None:
    print()
    print("=" * 70)
    print(f"PLAYLIST FOR: {result['mood_text']}")
    print("=" * 70)
    print()
    print("Emotional arc:")
    print(f"  {result['arc_summary']}")
    if result.get("global_warnings"):
        print()
        print("Critic warnings:")
        for w in result["global_warnings"]:
            print(f"  - {w}")
    print()
    print(f"Tracks ({len(result['tracks'])}):")
    print()
    for t in result["tracks"]:
        print(f"  {t['position']:>2}. {t['track_name']} by {t['artist']}")
        print(f"      why:        {t['explanation']}")
        print(f"      transition: {t['transition_note']}")
        print()
    if result.get("playlist_id") is not None:
        print(f"Saved as playlist id {result['playlist_id']} in data/app.db")
    print("=" * 70)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m orchestrator "<mood text>"')
        sys.exit(1)
    mood_text = " ".join(sys.argv[1:])
    init_db()
    result = generate_playlist(mood_text)
    _print_playlist(result)


if __name__ == "__main__":
    main()
