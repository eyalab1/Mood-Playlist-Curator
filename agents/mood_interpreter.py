"""Mood Interpreter agent.

Turns a free-form mood description into a structured profile that the rest
of the pipeline reasons about. No tools, no upstream dependencies.
"""

from pydantic import BaseModel, Field

from agents.base import run_agent


class MoodProfile(BaseModel):
    emotions: list[str] = Field(
        description="Short labels for the emotions in the mood."
    )
    energy: float = Field(
        ge=0.0, le=1.0,
        description="Target energy on Spotify's 0 to 1 scale.",
    )
    valence: float = Field(
        ge=0.0, le=1.0,
        description="Target positiveness on Spotify's 0 to 1 scale.",
    )
    tempo_range: tuple[int, int] = Field(
        description="Target BPM range, low to high."
    )
    themes: list[str] = Field(
        description="Concrete themes or imagery (e.g. rain, solitude)."
    )
    genres_favor: list[str] = Field(
        default_factory=list,
        description="Genres or subgenres that fit the mood well.",
    )
    genres_avoid: list[str] = Field(
        default_factory=list,
        description="Genres that should not appear in the playlist.",
    )
    context: str = Field(
        description="Listening context: background reading, workout, drive, etc."
    )


SYSTEM_PROMPT = """You are a music-savvy mood interpreter. Given a natural-language mood description, output a structured profile that downstream agents will use to curate a Spotify playlist.

Required JSON schema:
{
  "emotions": [string],            // 2 to 5 short emotion labels
  "energy": float,                 // 0.0 (calm/sleepy) to 1.0 (intense)
  "valence": float,                // 0.0 (negative/sad) to 1.0 (positive/happy)
  "tempo_range": [int, int],       // BPM range, e.g. [60, 90]
  "themes": [string],              // 2 to 5 concrete themes (e.g. rain, longing)
  "genres_favor": [string],        // genres that fit
  "genres_avoid": [string],        // genres that clash
  "context": string                // listening context: reading, workout, drive, sleep, party, focus, etc.
}

Calibration anchors:
- energy 0.2, valence 0.3, tempo 60-80 BPM: melancholy folk for a rainy evening
- energy 0.5, valence 0.7, tempo 90-110 BPM: warm, hopeful indie for a relaxed afternoon
- energy 0.9, valence 0.8, tempo 120-140 BPM: upbeat dance music for a party

Calibrate energy and valence carefully; they drive the audio-feature matching downstream. Tempo ranges should be realistic (rarely wider than 60 BPM, rarely narrower than 20 BPM).

If the user provides a taste profile, factor it into genre choices, but do not let it override the mood.

Output a single JSON object. No markdown fences, no prose."""


def run(
    mood_text: str,
    user_taste_profile: dict | None = None,
    playlist_id: int | None = None,
    use_cache: bool = True,
) -> MoodProfile:
    return run_agent(
        agent_name="mood_interpreter",
        system_prompt=SYSTEM_PROMPT,
        user_input={
            "mood_text": mood_text,
            "user_taste_profile": user_taste_profile,
        },
        output_model=MoodProfile,
        playlist_id=playlist_id,
        use_cache=use_cache,
    )


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "rainy sunday, melancholy but hopeful"
    profile = run(text)
    print(profile.model_dump_json(indent=2))
