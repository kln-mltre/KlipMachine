"""
Step 1.5: Refine & Edit
Review AI suggestions, adjust timestamps, and add manual clips.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from copy import deepcopy
import time

from core.brain import ClipSuggestion
from .shared import navigate_to_step, format_duration, format_timestamp

def on_global_change():
    """
    Callback for the global seek slider.

    Shifts the entire clip window by the delta between the slider value and the
    current start time, preserving clip duration while updating the preview position.
    """
    idx = st.session_state.active_clip_index
    key = f"global_seek_{idx}"
    if key in st.session_state:
        new_val = st.session_state[key]
        clip = st.session_state.editing_clips[idx]
        
        # Shift the entire clip window, preserving its duration.
        duration = clip.end - clip.start
        clip.start = float(new_val)
        clip.end = clip.start + duration

        st.session_state.last_preview_start = clip.start

def on_fine_change():
    """
    Callback for the fine-tune range slider.

    Updates the clip start and end times from the slider value and seeks the
    video preview to whichever boundary was moved.
    """
    idx = st.session_state.active_clip_index
    key = f"fine_slider_{idx}"
    if key in st.session_state:
        new_start, new_end = st.session_state[key]
        clip = st.session_state.editing_clips[idx]
        
        # Seek to the boundary that changed so the player reflects the edit immediately.
        if new_start != clip.start:
            st.session_state.last_preview_start = float(new_start)
            
        clip.start = float(new_start)
        clip.end = float(new_end)

def get_word_index_for_clip(all_words, clip_start, clip_end):
    """
    Return the indices of words that fall within a clip's time window.

    A 0.1 s margin is applied on both edges to capture words whose boundaries
    sit exactly on the clip start or end time.

    Args:
        all_words: Full list of word dicts with "start" and "end" keys.
        clip_start: Clip start time in seconds.
        clip_end: Clip end time in seconds.

    Returns:
        List of integer indices into all_words.
    """
    indices = []
    for i, w in enumerate(all_words):
        # 0.1 s tolerance captures words whose boundaries align exactly with the clip edges.
        if w["start"] >= clip_start - 0.1 and w["end"] <= clip_end + 0.1:
            indices.append(i)
    return indices

def render_step1_5_refine():
    """
    Render the Step 1.5 clip refinement interface.

    Displays a three-column layout:
        - Left: scrollable clip list with add/select/delete controls.
        - Centre: video player with global seek and fine-tune range sliders.
        - Right: per-word transcript editor for the active clip.

    Reads and mutates ``st.session_state.editing_clips`` and
    ``st.session_state.words`` in place.
    """
    st.markdown("### Step 1.5: Refine Clip")
    st.caption("Review AI suggestions, adjust timestamps, or add manual clips.")

    # 1. Check Data
    if "analysis" not in st.session_state or not st.session_state.analysis:
        st.error("No analysis data found.")
        if st.button("Back to Step 1"):
            navigate_to_step(1)
        return
    
    # Initialize working copy of clips
    if "editing_clips" not in st.session_state:
        st.session_state.editing_clips = deepcopy(st.session_state.analysis.clips)

    clips = st.session_state.editing_clips
    video_path = st.session_state.download_result.video_path
    total_duration = st.session_state.transcript_result.duration 

    # =========================================================================
    # LAYOUT
    # =========================================================================
    
    col_list, col_editor, col_subtitles = st.columns([12, 30, 16], gap="large", width="stretch")

    # --- LEFT COLUMN: Clip List ---
    with col_list:
        st.markdown(f"**Clips ({len(clips)})**")
        
        if st.button("Add Manual Clip", use_container_width=True, type="primary"):
            new_clip = ClipSuggestion(
                start=0.0,
                end=10.0,
                title=f"New Clip {len(clips) + 1}",
                hook="Custom hook",
                score=1.0,
                reason="Manual"
            )
            clips.append(new_clip)
            st.session_state.active_clip_index = len(clips) - 1
            st.rerun()

        st.markdown("---")

        if "active_clip_index" not in st.session_state:
            st.session_state.active_clip_index = 0
        if st.session_state.active_clip_index >= len(clips):
            st.session_state.active_clip_index = len(clips) - 1 if clips else 0

        for i, clip in enumerate(clips):
            is_active = (i == st.session_state.active_clip_index)
            
            if is_active:
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(34, 197, 94, 0.1); 
                        border: 1px solid rgba(34, 197, 94, 0.4); 
                        border-radius: 6px; 
                        padding: 12px; 
                        margin-bottom: 8px;
                        border-left: 4px solid #22c55e;">
                        <div style="font-size: 0.75em; color: #22c55e; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                            EDITING
                        </div>
                        <strong style="color: #ffffff; font-size: 1em;">{i+1}. {clip.title}</strong><br>
                        <div style="font-family: monospace; font-size: 0.85em; opacity: 0.7; margin-top: 4px;">
                            {format_timestamp(clip.start)} — {format_timestamp(clip.end)}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                label = f"{i+1} - {clip.title}"
                if st.button(label, key=f"sel_{i}", use_container_width=True, type="secondary"):
                    st.session_state.active_clip_index = i
                    st.session_state.last_preview_start = float(clip.start)  # Reset the preview seek position.
                    st.rerun()

    
    # --- RIGHT COLUMN: Visual Editor ---
    with col_editor:
        if not clips:
            st.info("No clips available.")
        else:
            # Get Active Clip
            idx = st.session_state.active_clip_index
            active_clip = clips[idx]
            
            st.subheader(f"Editing: {active_clip.title}")

            # 1. GLOBAL NAVIGATION (MACRO)
            st.markdown("#### Global Navigation")
            
            # Global seek slider with per-clip key to avoid cross-clip state collisions.
            st.slider(
                "Global Seek",
                min_value=0.0,
                max_value=total_duration,
                value=float(active_clip.start),
                step=1.0,
                format="%.0fs",
                key=f"global_seek_{idx}",
                label_visibility="collapsed",
                on_change=on_global_change
            )

            st.markdown("---")
            
            # 2. VIDEO PLAYER & REPLAY
            
            if "last_preview_start" not in st.session_state:
                st.session_state.last_preview_start = float(active_clip.start)

            col_label, col_btn = st.columns([3, 2])
            with col_label:
                st.markdown("### Preview Clip")
            with col_btn:
                btn_label = f"Replay at {format_timestamp(active_clip.start)}"

                if st.button(btn_label, use_container_width=True, type="secondary", key=f"replay_btn_{idx}"):
                    st.session_state.seek_command = float(active_clip.start)
                    st.rerun()



            # --- VIDEO PLAYER ---

            safe_start = float(st.session_state.last_preview_start)
            safe_end = float(active_clip.end)

            # Guard: if end <= start (e.g. due to slider rounding), enforce a 1 s minimum duration.
            if safe_end <= safe_start:
                safe_end = safe_start + 1.0

            try:
                st.video(
                    str(video_path),
                    start_time=safe_start,
                    end_time=safe_end
                )
            except Exception as e:
                st.warning(f"Video display error ({safe_start:.2f}s -> {safe_end:.2f}s): {e}")




            if "seek_command" in st.session_state:
                seek_time = st.session_state.seek_command

                # Unique ID prevents the browser from caching and replaying a stale seek command.
                unique_id = int(time.time() * 1000)

                # Inject JavaScript that polls for the <video> element and seeks it.
                # Polling at 100 ms intervals is necessary because Streamlit re-renders
                # the DOM asynchronously; the video element may not exist immediately.
                js = f"""
                <script>
                // Run ID: {unique_id}
                var attemps = 0;
                var seekTime = {seek_time};

                function attemptSeek() {{
                        var video = window.parent.document.querySelector('video');
                        if (video) {{
                            console.log("Video found, seeking to " + seekTime);
                            video.currentTime = seekTime;
                            video.play();
                            // Seek succeeded; stop polling.
                            clearInterval(interval);
                        }} else {{
                            attempts++;
                            // Abort after 3 s (30 attempts) to avoid an infinite poll loop.
                            if (attempts > 30) clearInterval(interval);
                        }}
                    }}

                    // Poll every 100 ms until the video element is available.
                    var interval = setInterval(attemptSeek, 100);
                </script>
                """
                # height=0 keeps the injected component invisible in the layout.
                components.html(js, height=0)
                del st.session_state.seek_command
            st.markdown("")

            # 3. FINE-TUNE SLIDER

            # Zoom the timeline view to a window around the clip with 30 s margins.
            zoom_margin = 30.0
            view_min = max(0.0, active_clip.start - zoom_margin)
            view_max = min(total_duration, active_clip.end + zoom_margin)
            if view_max - view_min < (active_clip.end - active_clip.start) + 10:
                view_max = min(total_duration, active_clip.end + 60)

            st.caption(f"**Timeline View:** {format_timestamp(view_min)} — {format_timestamp(view_max)}")

            # Fine-tune range slider with per-clip key to avoid cross-clip state collisions.
            st.slider(
                "Timeline",
                min_value=float(view_min),
                max_value=float(view_max),
                value=(float(active_clip.start), float(active_clip.end)),
                step=0.1,
                format="%.1f s",
                key=f"fine_slider_{idx}",
                label_visibility="collapsed",
                on_change=on_fine_change
            )
            
            # Metrics
            st.markdown("")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric("Start", format_timestamp(active_clip.start))
            with c2:
                st.metric("End", format_timestamp(active_clip.end))
            with c3:
                dur = active_clip.end - active_clip.start
                st.metric("Duration", f"{dur:.1f}s")

            # Metadata inputs
            st.markdown("---")
            active_clip.title = st.text_input("Clip Title", value=active_clip.title)
            active_clip.hook = st.text_input("Hook Text", value=active_clip.hook)

            # Delete
            st.markdown("")
            col_del, _ = st.columns([1, 3])
            with col_del:
                if st.button("Delete Clip", type="secondary", use_container_width=True):
                    clips.pop(idx)
                    st.session_state.active_clip_index = max(0, idx - 1)
                    st.rerun()

    with col_subtitles:
        st.markdown("---")
        st.markdown("### Edit Subtitles (Word by Word)")
        st.caption("Adjust the transcript words for this clip.")
        # WORD EDITION LOGIC
        if "words" in st.session_state and st.session_state.words:
            all_words = st.session_state.words
            current_start = active_clip.start
            current_end = active_clip.end
            target_indices = get_word_index_for_clip(all_words, current_start, current_end)

            if target_indices:
                # Build a minimal dataframe exposing only the editable Word column.
                editor_data = []
                for idx in target_indices:
                    w = all_words[idx]
                    editor_data.append({
                        "Start": f"{w['start']:.2f}",
                        "End": f"{w['end']:.2f}",
                        "Word": w['word']
                    })

                df = pd.DataFrame(editor_data)

                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="fixed",
                    column_config={
                        "Start": st.column_config.TextColumn("Start", disabled=True),
                        "End": st.column_config.TextColumn("End", disabled=True),
                        "Word": st.column_config.TextColumn("Word (Editable)", required=True)
                    },
                    hide_index=True,
                    key=f"word_editor_{idx}"
                )

                # Propagate any word edits back to the shared session state word list.
                for local_idx, row_idx in enumerate(target_indices):
                    original_word = all_words[row_idx]["word"]
                    new_word = edited_df.iloc[local_idx]["Word"]
                    if original_word != new_word:
                        st.session_state.words[row_idx]["word"] = new_word
            else:
                st.info("No words found for this clip's timeframe.")

    st.markdown("---")

    # NAVIGATION
    col_back, col_next = st.columns(2, vertical_alignment="bottom")

    with col_back:
        if st.button("← Back to Analysis", use_container_width=True):
            navigate_to_step(1)

    with col_next:
        if st.button("Proceed to Design →", type="primary", use_container_width=True):
            st.session_state.analysis.clips = clips
            st.session_state.selected_clips = list(range(len(clips)))
            navigate_to_step(2)



