"""
Step 1: Video Ingestion
Download, transcription, and AI analysis.
"""

import streamlit as st
from pathlib import Path

from config import config
from core.downloader import download_video
from core.transcriber import transcribe, export_transcript, create_analysis_segments, segments_to_text, load_transcript
from core.brain import analyze_transcript, export_analysis
from core.exceptions import KlipMachineError

from .shared import navigate_to_step


def render_step1_ingestion():
    """
    Render the Step 1 video ingestion interface.

    Displays a two-column layout covering video source selection (URL or local
    file), search angle configuration, AI provider credentials, and Whisper
    model selection.  Triggers ``_process_video`` when the user submits the form.
    """
    st.markdown("### Step 1: Video Ingestion")
    st.caption("Upload or paste a video URL, then process it")
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # =========================================================================
    # LEFT COLUMN \u2014 VIDEO SOURCE
    # =========================================================================
    
    with col_left:
        st.markdown("#### Video Source")
        
        source_type = st.radio(
            "source_type",
            ["YouTube URL", "Local File"],
            label_visibility="collapsed",
            horizontal=True
        )
        
        video_source = None
        
        if source_type == "YouTube URL":
            video_source = st.text_input(
                "url",
                placeholder="https://youtube.com/watch?v=...",
                label_visibility="collapsed"
            )
        else:
            uploaded_file = st.file_uploader(
                "file",
                type=["mp4", "mkv", "mov", "webm", "avi"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                temp_path = config.TEMP_DIR / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                video_source = str(temp_path)
        
        st.markdown("---")
        
        st.markdown("#### Search Angle")
        
        angle_type = st.radio(
            "angle",
            ["Short Clips", "Monetizable", "Multi-Parts", "Custom"],
            label_visibility="collapsed"
        )
        
        custom_prompt = None
        if angle_type == "Custom":
            custom_prompt = st.text_area(
                "instructions",
                placeholder="Describe what you're looking for...",
                height=100,
                label_visibility="collapsed"
            )
        
        st.markdown("---")
        
        st.markdown("#### AI Provider")
        
        provider = st.radio(
            "provider",
            ["groq", "openai", "ollama"],
            index=0,
            label_visibility="collapsed",
            horizontal=True
        )

        if provider == "groq":
            groq_key = st.text_input(
                "Groq API Key",
                value=config.GROQ_API_KEY or "",
                type="password",
                placeholder="gsk_...",
                key="groq_key"
            )
            if groq_key:
                config.GROQ_API_KEY = groq_key
        
        elif provider == "openai":
            openai_key = st.text_input(
                "OpenAI API Key",
                value=config.OPENAI_API_KEY or "",
                type="password",
                placeholder="sk-...",
                key="openai_key"
            )
            if openai_key:
                config.OPENAI_API_KEY = openai_key
        
        else:  # ollama
            ollama_host = st.text_input(
                "Ollama Host",
                value=config.OLLAMA_HOST,
                placeholder="http://localhost:11434",
                key="ollama_host"
            )
            config.OLLAMA_HOST = ollama_host

    
    # =========================================================================
    # RIGHT COLUMN — TRANSCRIPTION SETTINGS
    # =========================================================================
    
    with col_right:
        st.markdown("#### Transcription")
        
        whisper_choice = st.radio(
            "whisper",
            ["Auto", "base", "small"],
            index=0,
            label_visibility="collapsed",
            horizontal=True
        )
        
        if whisper_choice == "Auto":
            # "small" gives better accuracy on GPU; "base" is faster and sufficient on CPU.
            whisper_model = "small" if config.WHISPER_DEVICE == "cuda" else "base"
            device_type = "GPU" if config.WHISPER_DEVICE == "cuda" else "CPU"
            st.caption(f"Auto-selected: {whisper_model} ({device_type})")
        else:
            whisper_model = whisper_choice
        
        st.markdown("---")
        
        can_start = bool(video_source)
        if provider == "groq" and not config.GROQ_API_KEY:
            can_start = False
            st.warning("Add Groq API key")
        elif provider == "openai" and not config.OPENAI_API_KEY:
            can_start = False
            st.warning("Add OpenAI API key")

        st.markdown("")

        if st.button(
            "Analyze Video",
            disabled=not can_start,
            use_container_width=True,
            type="primary"
        ):
            _process_video(
                video_source,
                whisper_model,
                angle_type,
                custom_prompt,
                provider
            )


def _process_video(
    video_source: str,
    whisper_model: str,
    angle_type: str,
    custom_prompt: str,
    provider: str
):
    """
    Execute the full ingestion pipeline: download → transcribe → AI analysis.

    Writes results to ``st.session_state`` so downstream steps can access them
    without re-processing.  A pre-existing transcript JSON is reused when found,
    skipping redundant Whisper inference on re-runs.

    Args:
        video_source: YouTube URL or absolute path to a local video file.
        whisper_model: Faster-Whisper model size (e.g. ``"base"``, ``"small"``).
        angle_type: Clip search strategy label (``"Short Clips"``,
            ``"Monetizable"``, ``"Multi-Parts"``, or ``"Custom"``).
        custom_prompt: Free-form instructions when ``angle_type`` is
            ``"Custom"``; may be ``None`` for preset angles.
        provider: LLM provider identifier (``"groq"``, ``"openai"``, or
            ``"ollama"``).

    Raises:
        KlipMachineError: Propagated from any pipeline stage and surfaced as a
            Streamlit error widget.
    """
    progress_container = st.empty()
    status_container = st.empty()
    
    with progress_container:
        progress_bar = st.progress(0)
    
    try:
        # Download
        status_container.info("Downloading video...")
        progress_bar.progress(0.1)
        download_result = download_video(video_source, config.TEMP_DIR)
        status_container.success(f"Downloaded: {download_result.title[:40]}...")
        
        # Transcribe
        status_container.info("Transcribing audio...")
        progress_bar.progress(0.3)

        # Reuse a cached transcript to avoid redundant Whisper inference on re-runs.
        transcript_path = config.TEMP_DIR / f"{download_result.title}_transcript.json"
        if transcript_path.exists():
            print(f"[INFO] Cached transcript found, skipping Whisper inference.")
            result, words = load_transcript(transcript_path)
            status_container.success(f"Transcript loaded: {len(words)} words")
        else:
            result, words = transcribe(
                audio_path=download_result.audio_path,
                model_size=whisper_model,
                device="auto"
            )
            export_transcript(result, words, transcript_path)
            status_container.success(f"Transcribed: {len(words)} words")

        
        # AI Analysis
        status_container.info("Analyzing with AI...")
        progress_bar.progress(0.6)
        analysis_segments = create_analysis_segments(words, max_duration=45.0)
        transcript_text = segments_to_text(analysis_segments)
        analysis = analyze_transcript(
            transcript=transcript_text,
            angle=angle_type.lower().replace(" ", "-"),
            custom_instructions=custom_prompt,
            provider=provider
        )
        analysis_path = config.TEMP_DIR / f"{download_result.title}_analysis_result.json"
        export_analysis(analysis, analysis_path)
        status_container.success(f"Found: {len(analysis.clips)} clips")
        
        # Complete
        progress_bar.progress(1.0)
        status_container.success("Analysis complete!")
        
        st.session_state.download_result = download_result
        st.session_state.transcript_result = result
        st.session_state.words = words
        st.session_state.analysis = analysis

        # Invalidate stale editing state from any previous run.
        if "editing_clips" in st.session_state:
            del st.session_state.editing_clips

        if "active_clip_index" in st.session_state:
            st.session_state.active_clip_index = 0

        # Pre-select all clips so the user can deselect rather than manually pick.
        st.session_state.selected_clips = list(range(len(analysis.clips)))

        # Brief pause so the success status is readable before the page re-renders.
        import time
        time.sleep(1)
        navigate_to_step(1.5)
        
    except Exception as e:
        status_container.error(f"Error: {str(e)}")
        st.exception(e)