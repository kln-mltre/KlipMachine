"""
Step 2: Design & Preview
Interactive clip customization with real-time preview.
"""

import copy
import streamlit as st
from pathlib import Path
import tempfile

from config import config
from core.preview import generate_preview_frame
from core.presets import get_all_presets, ExportPreset
from core.transcriber import create_viral_subtitle_segments, export_ass, create_hook_overlay_png
from core.editor import normalize_crop_mode

from .shared import navigate_to_step, format_timestamp, format_duration

def _save_current_preset(name: str):
    """
    Persist the current session-state settings as a named user preset.

    Serialises crop mode, blur zoom, subtitle style, colour, font size, and
    vertical position into an ``ExportPreset`` and writes it to disk via
    ``save_user_preset``.  On success, updates ``last_preset`` and triggers a
    Streamlit rerun so the preset dropdown reflects the new entry.

    Args:
        name: Human-readable preset name provided by the user.
    """
    from core.presets import ExportPreset, save_user_preset
    
    # Create preset from current settings
    preset = ExportPreset(
        name=name,
        crop_mode=st.session_state.crop_mode,
        blur_zoom=st.session_state.blur_zoom,
        subtitle_style=st.session_state.subtitle_style,
        subtitle_color=st.session_state.subtitle_color,
        subtitle_font_size=st.session_state.font_size,
        subtitle_position=st.session_state.subtitle_position
    )
    
    try:
        filepath = save_user_preset(preset)
        st.success(f"Preset '{name}' saved successfully!")
        st.caption(f"Saved to: {filepath.name}")
        st.session_state.last_preset = name.lower()
        # Brief pause so the success message is visible before the page re-renders.
        import time
        time.sleep(0.5)
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to save preset: {e}")

def render_step2_design():
    """
    Render the Step 2 design and preview interface.

    Displays a two-column layout:
        - Left: global export settings (crop mode, subtitle style, presets).
        - Right: single-frame clip preview with per-clip selection checkboxes.

    Reads clip data from ``st.session_state.analysis`` and writes export
    configuration back to session state for consumption by Step 3.
    """
    st.markdown("### Step 2: Design Your Clips")
    st.caption("Customize export settings and preview results")
    
    if not st.session_state.analysis:
        st.error("No analysis data. Please complete Step 1 first.")
        if st.button("← Back to Step 1"):
            navigate_to_step(1)
        return
    
    analysis = st.session_state.analysis
    clips = analysis.clips

    col_settings, col_preview = st.columns([1, 1], gap="large")
    
    # =========================================================================
    # LEFT COLUMN - GLOBAL SETTINGS
    # =========================================================================
    
    with col_settings:
        st.markdown("#### Global Settings")
        st.caption("Apply to all selected clips")
        
        st.markdown("**Quick Presets**")
        presets = get_all_presets()
        preset_names = ["Custom"] + list(presets.keys())

        if "last_preset" not in st.session_state:
            st.session_state.last_preset = "Custom"
        
        selected_preset_name = st.selectbox(
            "preset",
            preset_names,
            index=preset_names.index(st.session_state.last_preset) if st.session_state.last_preset in preset_names else 0,
            label_visibility="collapsed"
        )

        if selected_preset_name != st.session_state.last_preset:
            if selected_preset_name != "Custom":
                _apply_preset(presets[selected_preset_name])
                # Note: _apply_preset will rerun, so code below won't execute
            else:
                st.session_state.last_preset = "Custom"

        # Only user-saved presets can be deleted; built-in presets are read-only.
        if selected_preset_name not in ["Custom", "viral", "clean", "cinematic", "colors"]:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Delete", key="delete_preset", help=f"Delete preset '{selected_preset_name}'"):
                    from core.presets import delete_user_preset
                    
                    if delete_user_preset(selected_preset_name):
                        st.success(f"Deleted!")
                        st.session_state.last_preset = "Custom"
                        import time
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("Failed to delete")
        
        st.markdown("---")

        st.markdown("**Export Format**")
        st.session_state.crop_mode = normalize_crop_mode(st.session_state.crop_mode)

        new_crop_mode = st.radio(
            "format",
            ["none", "blur", "black"],
            format_func=lambda x: {
                "none": "Original",
                "blur": "Blur Fill (9:16)",
                "black": "Black Fill (9:16)"
            }[x],
            index=["none", "blur", "black"].index(st.session_state.crop_mode),
            label_visibility="collapsed"
        )
        
        if new_crop_mode != st.session_state.crop_mode:
            st.session_state.crop_mode = new_crop_mode
            st.rerun()
        
        if st.session_state.crop_mode in ["blur", "black"]:
            st.markdown("**Zoom Level**")
            zoom_percent = st.slider(
                "zoom",
                min_value=0,
                max_value=85,
                value=int((st.session_state.blur_zoom - 0.65) * 100),
                step=1,
                label_visibility="collapsed",
                help="Zoom applied to main video"
            )
            new_zoom = 0.65 + (zoom_percent / 100)
            
            if abs(new_zoom - st.session_state.blur_zoom) > 0.01:
                st.session_state.blur_zoom = new_zoom
                st.rerun()
            
            st.caption(f"Zoom: {new_zoom:.2f}x")
        
        st.markdown("---")
        
        st.markdown("**Subtitle Style**")
        new_subtitle_style = st.radio(
            "subtitle_style",
            ["none", "colors", "box"],
            format_func=lambda x: {
                "none": "None",
                "colors": "Colors (Switch)",
                "box": "Box (Viral)"
            }[x],
            index=["none", "colors", "box"].index(st.session_state.subtitle_style),
            label_visibility="collapsed"
        )
        
        if new_subtitle_style != st.session_state.subtitle_style:
            st.session_state.subtitle_style = new_subtitle_style
            st.rerun()
        
        # Subtitle Color (only if subtitles enabled)
        if st.session_state.subtitle_style == "colors":
            st.markdown("**Starting color**")
            
            from core.presets import SUBTITLE_COLORS
            color_names = list(SUBTITLE_COLORS.keys())
            
            new_subtitle_color = st.selectbox(
                "subtitle_color",
                color_names,
                index=color_names.index(st.session_state.subtitle_color) if st.session_state.subtitle_color in color_names else 0,
                label_visibility="collapsed"
            )
            
            if new_subtitle_color != st.session_state.subtitle_color:
                st.session_state.subtitle_color = new_subtitle_color
                st.rerun()
        
        # Font size and position (only if subtitles enabled)
        if st.session_state.subtitle_style != "none":
            st.markdown("**Font Size**")
            new_font_size = st.slider(
                "font_size",
                min_value=12,
                max_value=50,
                value=st.session_state.font_size if st.session_state.font_size >= 12 else 12,
                step=2,
                label_visibility="collapsed"
            )
            
            if new_font_size != st.session_state.font_size:
                st.session_state.font_size = new_font_size
                st.rerun()
            
            st.caption(f"Size: {new_font_size}px")
            
            st.markdown("**Vertical Position**")
            new_subtitle_position = st.slider(
                "position",
                min_value=88.0,
                max_value=99.0,
                value=float(st.session_state.subtitle_position),
                step=0.25,
                label_visibility="collapsed",
                help="Distance from top of video frame"
            )
            
            if new_subtitle_position != st.session_state.subtitle_position:
                st.session_state.subtitle_position = new_subtitle_position
                st.rerun()
            
            st.caption(f"Position: {new_subtitle_position:.2f}% from top")

        st.markdown("---")
        
        st.markdown("**Save Current Settings**")
        
        col_name, col_save = st.columns([2, 1])
        
        with col_name:
            preset_name = st.text_input(
                "preset_name",
                placeholder="My Custom Preset",
                label_visibility="collapsed",
                help="Name for your custom preset"
            )
        
        with col_save:
            st.markdown("")
            if st.button(
                "Save",
                use_container_width=True,
                disabled=not preset_name or not preset_name.strip()
            ):
                _save_current_preset(preset_name.strip())
        
        st.markdown("---")
        
        col_back, col_next = st.columns(2, vertical_alignment="bottom")

        with col_back:
            if st.button("← Back", use_container_width=True):
                # Reset preset selection so Step 2 opens cleanly on re-entry.
                st.session_state.last_preset = "Custom"
                navigate_to_step(1.5)
        
        with col_next:
            selected_count = len(st.session_state.selected_clips)
            if st.button(
                f"Export {selected_count} Clips →",
                use_container_width=True,
                type="primary",
                disabled=selected_count == 0
            ):
                navigate_to_step(3)
    
    # =========================================================================
    # RIGHT COLUMN - PREVIEW & CLIP SELECTION
    # =========================================================================
    
    with col_preview:
        st.markdown("#### Preview & Clip Selection")

        if st.session_state.crop_mode != "none":
            show_tiktok_ui = st.checkbox("Show TikTok UI Overlay", value=True)
        else:
            show_tiktok_ui = False
        
        if st.session_state.selected_clips:
            first_clip_idx = st.session_state.selected_clips[0]
            clip = clips[first_clip_idx]

            hook_enabled, _, hook_font_size, hook_position = _get_clip_hook_settings(clip)

            st.markdown("**Hook Overlay**")
            new_hook_enabled = st.checkbox(
                "Show Hook Text",
                value=hook_enabled,
                key=f"hook_enabled_step2_{first_clip_idx}"
            )
            if new_hook_enabled != hook_enabled:
                clip.hook_enabled = new_hook_enabled
                st.rerun()

            st.markdown("**Hook Font Size**")
            new_hook_font_size = st.slider(
                "hook_font_size",
                min_value=20,
                max_value=50,
                value=max(20, min(50, int(hook_font_size))),
                step=2,
                label_visibility="collapsed"
            )
            if new_hook_font_size != hook_font_size:
                clip.hook_font_size = new_hook_font_size
                st.rerun()

            st.caption(f"Size: {new_hook_font_size}px")

            st.markdown("**Hook Vertical Position**")

            new_hook_position = st.slider(
                "Hook Position",
                min_value=5.0,
                max_value=99.0,
                value=float(hook_position),
                step=0.25,
                help="Distance from top of video frame",
                key=f"hook_position_step2_{first_clip_idx}"
            )
            if abs(new_hook_position - hook_position) > 0.01:
                clip.hook_position = new_hook_position
                st.rerun()

            st.caption(f"Hook position: {new_hook_position:.1f}% from top")
            st.markdown("")

            st.markdown(f"**Preview: Clip {first_clip_idx + 1}**")
            st.caption(f"{clip.title} ({format_duration(clip.end - clip.start)})")

            preview_path = _generate_clip_preview(first_clip_idx, clip, show_tiktok_ui)
            
            if preview_path and preview_path.exists():
                st.image(str(preview_path), width=500)
            else:
                st.warning("Preview generation failed")
        else:
            st.info("No clips selected. Select at least one clip below.")
        
        st.markdown("---")
        st.markdown("**Select Clips to Export**")

        for i, clip in enumerate(clips):
            col_check, col_info = st.columns([0.1, 0.9])
            
            with col_check:
                is_selected = i in st.session_state.selected_clips
                if st.checkbox(
                    f"clip_{i}",
                    value=is_selected,
                    label_visibility="collapsed",
                    key=f"select_clip_{i}"
                ):
                    if i not in st.session_state.selected_clips:
                        st.session_state.selected_clips.append(i)
                        st.rerun()
                else:
                    if i in st.session_state.selected_clips:
                        st.session_state.selected_clips.remove(i)
                        st.rerun()
            
            with col_info:
                duration = clip.end - clip.start
                time_range = f"{format_timestamp(clip.start)} → {format_timestamp(clip.end)}"
                st.markdown(
                    f"**{i+1}. {clip.title}**  \n"
                    f"{time_range} ({format_duration(duration)}) | "
                    f"{clip.score:.0%}"
                )
        
        st.markdown("")
        col_all, col_none = st.columns(2)
        
        with col_all:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_clips = list(range(len(clips)))
                st.rerun()
        
        with col_none:
            if st.button("Deselect All", use_container_width=True):
                st.session_state.selected_clips = []
                st.rerun()


def _apply_preset(preset: ExportPreset):
    """
    Write all fields of an ``ExportPreset`` into session state and rerun.

    Mapping preset fields to their session-state counterparts and calling
    ``st.rerun`` ensures the entire UI re-renders with the new values
    without requiring the user to interact with individual controls.

    Args:
        preset: The preset whose values should be applied to the current session.
    """
    st.session_state.crop_mode = normalize_crop_mode(preset.crop_mode)
    st.session_state.blur_zoom = preset.blur_zoom
    st.session_state.subtitle_style = preset.subtitle_style
    st.session_state.subtitle_color = preset.subtitle_color
    st.session_state.font_size = preset.subtitle_font_size
    st.session_state.subtitle_position = preset.subtitle_position
    st.session_state.last_preset = preset.name.lower()
    st.rerun()


def _get_clip_hook_settings(clip) -> tuple[bool, str, int, float]:
    """Return normalized hook settings for a clip with backward-compatible defaults."""
    hook_text = getattr(clip, "hook", "") or ""
    hook_enabled = bool(getattr(clip, "hook_enabled", bool(hook_text.strip())))
    hook_font_size = max(20, min(50, int(getattr(clip, "hook_font_size", 20))))
    hook_position = max(5.0, min(99.0, float(getattr(clip, "hook_position", 8.0))))

    clip.hook_enabled = hook_enabled
    clip.hook_font_size = hook_font_size
    clip.hook_position = hook_position

    return hook_enabled, hook_text, hook_font_size, hook_position

def _get_subtitle_timestamp(clip) -> float:
    """
    Return an absolute timestamp guaranteed to fall on an active subtitle.

    Builds viral subtitle segments for the clip words and targets the start of
    the first segment, offset by 0.1 s to land in the middle of the first
    displayed word.  Falls back to the clip midpoint when no words are found.

    Args:
        clip: ``ClipSuggestion`` object with ``start`` and ``end`` attributes.

    Returns:
        Absolute timestamp in seconds suitable for single-frame extraction.
    """
    words = st.session_state.words

    # Get words in this clip
    clip_words = [
        w for w in words
        if clip.start <= w["start"] <= clip.end
    ]

    if not clip_words:
        return (clip.start + clip.end) / 2

    # Normalise timestamps relative to clip start for create_viral_subtitle_segments.
    adjusted_words = [
        {
            "word": w["word"],
            "start": w["start"] - clip.start,
            "end": w["end"] - clip.start
        }
        for w in clip_words
    ]

    # Create viral segments to see where subtitles would appear
    from core.transcriber import create_viral_subtitle_segments
    viral_segments = create_viral_subtitle_segments(adjusted_words, max_words=3)

    if not viral_segments:
        return (clip.start + clip.end) / 2

    first_segment = viral_segments[0]
    segment_start = first_segment.start
    # +0.1 s nudge lands inside the first word's display window rather than on its boundary.
    return clip.start + segment_start + 0.1

    

def _generate_clip_preview(clip_idx: int, clip, show_ui: bool = False) -> Path:
    """
    Generate and cache a single preview frame for a clip.

    Builds a settings hash from all parameters that affect the visual output
    and skips extraction when a cached image for that hash already exists.
    When ``show_ui`` is ``True``, composites the TikTok UI overlay on top of
    the generated frame using Pillow (the overlay is applied in Python rather
    than via FFmpeg to avoid re-running the full filter graph on a PNG).

    Args:
        clip_idx: Zero-based index of the clip within the analysis clip list.
        clip: ``ClipSuggestion`` object with ``start`` and ``end`` attributes.
        show_ui: Whether to composite the TikTok UI overlay onto the preview.

    Returns:
        Path to the output JPEG, or ``None`` if generation failed.
    """
    # Hash all parameters that affect visual output to get a cache key.
    hook_enabled, hook_text, hook_font_size, hook_position = _get_clip_hook_settings(clip)

    preview_cache_version = "v3"
    settings_hash = f"{preview_cache_version}_{clip_idx}_{st.session_state.crop_mode}_{st.session_state.blur_zoom:.2f}_{st.session_state.subtitle_style}_{st.session_state.subtitle_color}_{st.session_state.font_size}_{st.session_state.subtitle_position}_{hook_enabled}_{hook_text}_{hook_font_size}_{hook_position:.1f}_{show_ui}"
    cache_key = f"preview_cache_{settings_hash}"

    if cache_key in st.session_state:
        cached_path = st.session_state[cache_key]
        if cached_path.exists():
            return cached_path

    video_path = st.session_state.download_result.video_path
    timestamp = _get_subtitle_timestamp(clip)

    temp_dir = Path(tempfile.gettempdir()) / "klipmachine_previews"
    temp_dir.mkdir(exist_ok=True)
    output_path = temp_dir / f"preview_{clip_idx}_{settings_hash}.jpg"

    subtitle_path = None
    mask_paths = None
    mask_info = None 
    time_offset = 0.0
    hook_overlay_path = None

    if hook_enabled and hook_text.strip():
        hook_dir = temp_dir / "hook_overlays"
        hook_overlay_path = create_hook_overlay_png(
            hook_text=hook_text,
            output_path=hook_dir / f"hook_preview_{clip_idx}.png",
            font_size=hook_font_size,
            top_position_percent=hook_position
        )

    if st.session_state.subtitle_style != "none" or (hook_enabled and hook_text.strip()):
        subtitle_path, mask_info, time_offset = _generate_preview_subtitle(clip_idx, clip, timestamp)

        if mask_info and mask_info.get("words"):
            from core.transcriber import generate_highlight_box_masks

            mask_dir = temp_dir / f"masks_{clip_idx}"
            box_color_rgb = mask_info.get("box_color", (128, 0, 128, 255))

            mask_paths = generate_highlight_box_masks(
                words=mask_info["words"],
                output_dir=mask_dir,
                segments=mask_info.get("segments"),
                font_size=mask_info["font_size"],
                v_position_percent=mask_info["v_position_percent"],
                box_color=box_color_rgb
            )

    preview_mask_info = None
    if mask_info:
        preview_mask_info = copy.deepcopy(mask_info)
        for w in preview_mask_info["words"]:
            w["start"] += time_offset
            w["end"] += time_offset

    try:
        success = generate_preview_frame(
            video_path=video_path,
            timestamp=timestamp,
            output_path=output_path,
            crop_mode=st.session_state.crop_mode,
            blur_zoom=st.session_state.blur_zoom,
            subtitle_path=subtitle_path,
            mask_paths=mask_paths,
            mask_info=preview_mask_info,
            # The TikTok UI overlay is composited in Python below to avoid
            # re-running the full FFmpeg filter graph for a static PNG layer.
            show_tiktok_ui=False
        )
        if success:
            if hook_overlay_path and hook_overlay_path.exists():
                from PIL import Image

                try:
                    base_img = Image.open(output_path).convert("RGBA")
                    hook_img = Image.open(hook_overlay_path).convert("RGBA")
                    base_img.paste(hook_img, (0, 0), hook_img)
                    base_img = base_img.convert("RGB")
                    base_img.save(output_path, quality=95)
                except Exception as e:
                    print(f"[ERROR] Failed to apply hook overlay: {e}")

            if show_ui:
                from PIL import Image

                overlay_path = Path("assets/tiktok_preview.png")

                if overlay_path.exists():
                    try:
                        base_img = Image.open(output_path).convert("RGBA")
                        overlay_img = Image.open(overlay_path).convert("RGBA")

                        # Resize overlay to match the generated frame dimensions.
                        if overlay_img.size != base_img.size:
                            overlay_img = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)

                        # Composite overlay on top of text using its alpha channel as mask.
                        base_img.paste(overlay_img, (0, 0), overlay_img)

                        # Convert back to RGB before saving as JPEG (no alpha support).
                        base_img = base_img.convert("RGB")
                        base_img.save(output_path, quality=95)

                    except Exception as e:
                        print(f"[ERROR] Failed to apply TikTok overlay: {e}")
                else:
                    print(f"[ERROR] Overlay asset not found: {overlay_path}")

            st.session_state[cache_key] = output_path
            return output_path
            
    except Exception as e:
        st.error(f"Preview error: {e}")
    
    return None


def _generate_preview_subtitle(clip_idx: int, clip, preview_timestamp: float) -> Path:
    """
    Generate a minimal ASS subtitle file aligned to the preview timestamp.

    Extracts words from a 3-second window centred on ``preview_timestamp``,
    normalises their timestamps to start at zero, builds viral segments, and
    calls ``export_ass`` to produce an ASS file consistent with the full
    export pipeline.  Returns the subtitle path, mask metadata dict, and the
    time offset needed to re-align word timestamps for the preview frame.

    Args:
        clip_idx: Zero-based index of the clip within the analysis clip list.
        clip: ``ClipSuggestion`` object with ``start`` and ``end`` attributes.
        preview_timestamp: Absolute timestamp (seconds) at which the preview
            frame is extracted.

    Returns:
        Tuple of ``(subtitle_path, mask_info, time_offset)`` where
        ``subtitle_path`` is the generated ASS file, ``mask_info`` is the
        box-highlight metadata dict (or ``None``), and ``time_offset`` is the
        absolute start time of the first preview word.
    """
    words = st.session_state.words
    hook_enabled, hook_text, hook_font_size, hook_position = _get_clip_hook_settings(clip)

    start_time = clip.start
    end_time = clip.end
    
    clip_words = [
        w for w in words
        if start_time <= w["start"] <= end_time
    ]

    temp_dir = Path(tempfile.gettempdir()) / "klipmachine_previews"
    temp_dir.mkdir(exist_ok=True)
    subtitle_path = temp_dir / f"preview_subtitle_{clip_idx}.ass"

    if st.session_state.subtitle_style == "none":
        return (None, None, 0.0)
    
    if not clip_words:
        st.warning(f"No words found for clip {clip_idx + 1}")
        return None

    # 3-second window (±1.5 s) around the preview timestamp ensures at least
    # one full subtitle segment is visible in the extracted frame.
    window_start = preview_timestamp - 1.5
    window_end = preview_timestamp + 1.5
    
    preview_words = [
        w for w in clip_words
        if window_start <= w["start"] <= window_end
    ]
    
    if not preview_words:
        st.warning(f"No words found at preview timestamp {preview_timestamp:.2f}s")
        # Fallback to the first few words when the window contains nothing.
        preview_words = clip_words[:min(3, len(clip_words))]

    first_word_start = 0.0
    if preview_words:
        first_word_start = preview_words[0]["start"]
        # Normalise to t=0 so export_ass produces valid ASS timestamps for a preview.
        normalized_words = [
            {
                "word": w["word"],
                "start": w["start"] - first_word_start,
                "end": w["end"] - first_word_start
            }
            for w in preview_words
        ]
    else:
        normalized_words = preview_words

    from core.transcriber import export_ass, create_viral_subtitle_segments

    segments = create_viral_subtitle_segments(normalized_words, max_words=3)

    # Reuse export_ass so the preview ASS header and \pos values are identical
    # to what the full export pipeline would produce.
    mask_info = export_ass(
        segments=segments,
        words=normalized_words,
        output_path=subtitle_path,
        style_type=st.session_state.subtitle_style,
        color_name=st.session_state.subtitle_color,
        font_size=st.session_state.font_size,
        v_position_percent=st.session_state.subtitle_position,
        primary_color=st.session_state.subtitle_color
    )

    if not subtitle_path.exists():
        st.error("Failed to create subtitle file")
        return (None, None, first_word_start)

    return (subtitle_path, mask_info if mask_info else None, first_word_start)
