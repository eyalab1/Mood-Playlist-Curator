"""Streamlit UI for the Mood Playlist Curator.

Clean, light, editorial look. Each track renders as a live Spotify embed
player with the agent's explanation and transition note beneath it.

Run:  streamlit run app.py
"""

import html

import streamlit as st
import streamlit.components.v1 as components

from db import init_db
from orchestrator import generate_playlist

st.set_page_config(
    page_title="Mood Playlist Curator",
    page_icon="\U0001F3B5",
    layout="centered",
)

init_db()


# --------------------------------------------------------------------------
# Styling: clean light editorial theme
# --------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {max-width: 760px; padding-top: 2.5rem;}

.hero-title {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
    color: #111;
}
.hero-sub {
    font-size: 1.05rem;
    color: #666;
    margin-bottom: 1.6rem;
}
.section-label {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.75rem;
    color: #999;
    margin: 1.8rem 0 0.6rem 0;
}
.chip {
    display: inline-block;
    border: 1px solid #111;
    border-radius: 999px;
    padding: 3px 12px;
    margin: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #111;
}
.chip-soft {
    display: inline-block;
    border: 1px solid #ddd;
    border-radius: 999px;
    padding: 3px 12px;
    margin: 0 6px 6px 0;
    font-size: 0.8rem;
    color: #555;
}
.metric {display: flex; align-items: center; gap: 10px; margin: 5px 0;}
.metric-label {width: 70px; font-size: 0.85rem; color: #444;}
.metric-val {width: 38px; font-size: 0.8rem; color: #888; text-align: right;}
.bar {flex: 1; height: 6px; background: #eee; border-radius: 999px;}
.bar-fill {height: 6px; background: #111; border-radius: 999px;}

.arc {
    font-family: Georgia, serif;
    font-size: 1.15rem;
    line-height: 1.6;
    color: #222;
    border-left: 3px solid #111;
    padding: 0.2rem 0 0.2rem 1rem;
    margin: 0.5rem 0 1rem 0;
}
.track-head {
    font-family: Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #111;
    margin: 1.4rem 0 0.4rem 0;
}
.why {font-size: 0.95rem; color: #333; margin: 0.5rem 0 0.2rem 0;}
.transition {font-size: 0.85rem; color: #888; font-style: italic; margin: 0;}
.track-divider {border: none; border-top: 1px solid #eee; margin: 1.2rem 0;}
.footer-note {color: #aaa; font-size: 0.8rem; margin-top: 3rem; text-align: center;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "mood" not in st.session_state:
    st.session_state.mood = ""
if "mood_input" not in st.session_state:
    st.session_state.mood_input = ""


EXAMPLE_MOODS = [
    "rainy sunday, melancholy but hopeful",
    "late night drive, neon lights, a little lonely",
    "sunday morning coffee, soft and easy",
    "3am, can't sleep, mind racing",
]


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def bar(label: str, value: float) -> str:
    pct = int(max(0.0, min(1.0, value)) * 100)
    return (
        f'<div class="metric"><span class="metric-label">{label}</span>'
        f'<div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>'
        f'<span class="metric-val">{value:.2f}</span></div>'
    )


def spotify_embed(track_id: str) -> None:
    components.html(
        f'<iframe style="border-radius:12px" '
        f'src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" '
        f'width="100%" height="80" frameBorder="0" loading="lazy" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; '
        f'picture-in-picture"></iframe>',
        height=90,
    )


def render_profile(profile: dict) -> None:
    st.markdown('<div class="section-label">Mood reading</div>', unsafe_allow_html=True)
    emotions = profile.get("emotions", [])
    if emotions:
        chips = "".join(f'<span class="chip">{html.escape(e)}</span>' for e in emotions)
        st.markdown(chips, unsafe_allow_html=True)

    bars = bar("energy", profile.get("energy", 0.0)) + bar(
        "valence", profile.get("valence", 0.0)
    )
    st.markdown(bars, unsafe_allow_html=True)

    context = profile.get("context")
    if context:
        st.markdown(
            f'<p style="color:#555;font-size:0.9rem;margin-top:0.6rem;">'
            f'Context: {html.escape(str(context))}</p>',
            unsafe_allow_html=True,
        )

    favor = profile.get("genres_favor", [])
    if favor:
        chips = "".join(f'<span class="chip-soft">{html.escape(g)}</span>' for g in favor)
        st.markdown(
            '<div style="margin-top:0.4rem;font-size:0.8rem;color:#999;">'
            'leaning toward</div>',
            unsafe_allow_html=True,
        )
        st.markdown(chips, unsafe_allow_html=True)


def render_playlist(result: dict) -> None:
    render_profile(result.get("profile", {}))

    st.markdown('<div class="section-label">Emotional arc</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="arc">{html.escape(result.get("arc_summary", ""))}</div>',
        unsafe_allow_html=True,
    )

    warnings = result.get("global_warnings") or []
    for w in warnings:
        st.caption(f"Note: {w}")

    tracks = result.get("tracks", [])
    st.markdown(
        f'<div class="section-label">The playlist &middot; {len(tracks)} tracks</div>',
        unsafe_allow_html=True,
    )
    for t in tracks:
        st.markdown(
            f'<div class="track-head">{t["position"]:02d} &nbsp; '
            f'{html.escape(t["track_name"])} '
            f'<span style="font-weight:400;color:#888;">/ '
            f'{html.escape(t["artist"])}</span></div>',
            unsafe_allow_html=True,
        )
        spotify_embed(t["spotify_track_id"])
        st.markdown(
            f'<p class="why"><strong>Why:</strong> '
            f'{html.escape(t.get("explanation", ""))}</p>',
            unsafe_allow_html=True,
        )
        if t.get("transition_note"):
            st.markdown(
                f'<p class="transition">{html.escape(t["transition_note"])}</p>',
                unsafe_allow_html=True,
            )
        st.markdown('<hr class="track-divider">', unsafe_allow_html=True)


def run_pipeline(mood_text: str) -> None:
    """Run the pipeline with a live status box and store the result."""
    with st.status("Running the 4-agent pipeline...", expanded=True) as status:
        def on_step(label: str) -> None:
            status.update(label=label)
            st.write(label)

        try:
            result = generate_playlist(mood_text, on_step=on_step)
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            status.update(label="Something went wrong", state="error")
            st.error(str(e))
            return
        status.update(label="Playlist ready", state="complete", expanded=False)

    st.session_state.result = result
    st.session_state.mood = mood_text


# --------------------------------------------------------------------------
# Header + input
# --------------------------------------------------------------------------
st.markdown('<div class="hero-title">Mood Playlist Curator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Describe a feeling. A team of AI agents reads it, '
    'curates tracks, critiques the selection, and sequences an emotional arc.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, ex in enumerate(EXAMPLE_MOODS):
    if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state.mood_input = ex
        st.rerun()

mood = st.text_area(
    "Your mood",
    key="mood_input",
    height=90,
    placeholder="e.g. cooking dinner on a warm friday, relaxed but a little festive",
    label_visibility="collapsed",
)

if st.button("Generate Playlist", type="primary", use_container_width=True):
    if mood.strip():
        run_pipeline(mood.strip())
    else:
        st.warning("Type a mood first, or pick an example above.")


# --------------------------------------------------------------------------
# Results + refinement
# --------------------------------------------------------------------------
if st.session_state.result is not None:
    st.divider()
    render_playlist(st.session_state.result)

    st.markdown('<div class="section-label">Refine it</div>', unsafe_allow_html=True)
    refinement = st.text_input(
        "Refinement",
        placeholder="e.g. more upbeat, add some funk, fewer sad songs",
        label_visibility="collapsed",
    )
    if st.button("Refine Playlist", use_container_width=True):
        if refinement.strip():
            combined = f"{st.session_state.mood}. Adjustment: {refinement.strip()}"
            run_pipeline(combined)
            st.rerun()
        else:
            st.warning("Type an adjustment first.")

st.markdown(
    '<div class="footer-note">Multi-agent pipeline (Mood Interpreter, Curator, '
    'Critic, Sequencer) with lyrics-based RAG.</div>',
    unsafe_allow_html=True,
)
