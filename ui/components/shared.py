"""
Shared utilities for UI components.
Session state management and common widgets.
"""

import streamlit as st
from pathlib import Path


def init_session_state():
    """
    Initialize all session state variables.
    Called once at app startup.
    """
    # Workflow step
    if "step" not in st.session_state:
        st.session_state.step = 1
    
    # Step 1: Ingestion data
    if "video_source" not in st.session_state:
        st.session_state.video_source = None
    if "download_result" not in st.session_state:
        st.session_state.download_result = None
    if "transcript_result" not in st.session_state:
        st.session_state.transcript_result = None
    if "words" not in st.session_state:
        st.session_state.words = None
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    
    # Step 2: Design settings (global for all clips)
    if "crop_mode" not in st.session_state:
        st.session_state.crop_mode = "blur"
    if "blur_zoom" not in st.session_state:
        st.session_state.blur_zoom = 1.08
    if "subtitle_style" not in st.session_state:
        st.session_state.subtitle_style = "colors"
    if "subtitle_color" not in st.session_state:
        st.session_state.subtitle_color = "Purple"
    if "font_size" not in st.session_state:
        st.session_state.font_size = 18
    if "subtitle_position" not in st.session_state:
        st.session_state.subtitle_position = 87
    
    # Step 3: Export results
    if "export_results" not in st.session_state:
        st.session_state.export_results = None
    if "output_folder" not in st.session_state:
        st.session_state.output_folder = None


def reset_workflow():
    """
    Reset workflow to step 1 and clear all data.
    """
    st.session_state.step = 1
    st.session_state.video_source = None
    st.session_state.download_result = None
    st.session_state.transcript_result = None
    st.session_state.words = None
    st.session_state.analysis = None
    st.session_state.editing_clips = None # 1.5
    st.session_state.export_results = None
    st.session_state.output_folder = None
    st.session_state.selected_clips = []


def show_progress_bar(current: int, total: int, label: str = "Processing"):
    """
    Display a progress bar with label.
    
    Args:
        current: Current progress value
        total: Total value (100%)
        label: Progress label text
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{label} ({current}/{total})")


def navigate_to_step(step: int):
    """
    Navigate to a specific workflow step.
    
    Args:
        step: Step number (1, 2, or 3)
    """
    if step in [1, 1.5, 2, 3]:
        st.session_state.step = step
        st.rerun()


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "02:34")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    Format duration to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 34s" or "45s")
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        return f"{int(seconds)}s"