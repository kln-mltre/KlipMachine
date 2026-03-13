"""
Step 3: Export & Download
Batch export clips with progress tracking.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import zipfile
import io

from config import config
from core.editor import ClipConfig, batch_export, sanitize_filename, normalize_crop_mode
from core.transcriber import create_viral_subtitle_segments, export_ass, create_subtitle_segments, export_srt
from core.exceptions import KlipMachineError

from .shared import navigate_to_step, format_duration


def render_step3_export():
    """
    Render the Step 3 export and download interface.

    Dispatches to either the export configuration view or the results view
    depending on whether ``st.session_state.export_results`` is populated.
    Guards against missing prerequisites (no analysis or no selected clips)
    and offers navigation back to Step 2.
    """
    st.markdown("### Step 3: Export")
    st.caption("Generate and download your clips")
    
    if not st.session_state.analysis or not st.session_state.selected_clips:
        st.error("No clips selected. Please complete Step 2 first.")
        if st.button("← Back to Step 2"):
            navigate_to_step(2)
        return
    
    if st.session_state.export_results:
        _show_results()
    else:
        _show_export_interface()


def _show_export_interface():
    """
    Display the pre-export summary and settings recap, then trigger the export.

    Renders clip count, total duration, and format metrics alongside a
    settings recap.  The primary action button calls ``_start_export``.
    """
    analysis = st.session_state.analysis
    selected_indices = st.session_state.selected_clips
    selected_clips = [analysis.clips[i] for i in selected_indices]
    crop_mode = normalize_crop_mode(st.session_state.crop_mode)

    st.markdown("#### Export Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected Clips", len(selected_clips))
    with col2:
        total_duration = sum(clip.end - clip.start for clip in selected_clips)
        st.metric("Total Duration", format_duration(total_duration))
    with col3:
        st.metric("Format", {
            "none": "Original",
            "blur": "Blur Fill",
            "black": "Black Fill"
        }[crop_mode])
    
    st.markdown("---")

    st.markdown("#### Export Settings")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"**Crop Mode:** {crop_mode}")
        st.markdown(f"**Subtitle Style:** {st.session_state.subtitle_style}")
    
    with col_right:
        if st.session_state.subtitle_style != "none":
            st.markdown(f"**Font Size:** {st.session_state.font_size}px")
            st.markdown(f"**Position:** {st.session_state.subtitle_position}%")
    
    st.markdown("---")

    col_back, col_export = st.columns([1, 2])
    
    with col_back:
        if st.button("← Back to Design", use_container_width=True):
            navigate_to_step(2)
    
    with col_export:
        if st.button(
            f"Export {len(selected_clips)} Clips",
            use_container_width=True,
            type="primary"
        ):
            _start_export()


def _start_export():
    """
    Execute the full export pipeline for all selected clips.

    Two-pass strategy: the first ``batch_export`` call produces raw video
    files; the second (subtitle path) burns ASS subtitles directly into the
    video stream.  The two-pass approach separates slow video encoding from
    subtitle generation so subtitle errors do not discard an already-encoded
    video.  When subtitle style is ``"none"``, plain SRT files are written
    alongside the clips for optional external use.

    Writes ``st.session_state.export_results`` and
    ``st.session_state.output_folder`` on success, then triggers a rerun to
    switch the UI to the results view.
    """
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # Unique timestamp-based directory prevents collisions between runs.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = sanitize_filename(st.session_state.download_result.title)
        output_dir = config.OUTPUT_DIR / f"{safe_title}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis = st.session_state.analysis
        selected_indices = st.session_state.selected_clips
        selected_clips = [analysis.clips[i] for i in selected_indices]

        clip_configs = [
            ClipConfig(
                start=clip.start,
                end=clip.end,
                title=clip.title,
                margin_before=config.MARGIN_BEFORE,
                margin_after=config.MARGIN_AFTER
            )
            for clip in selected_clips
        ]

        # Pass 1: export raw video clips without subtitle burn-in.
        status_container.info("Exporting video clips...")
        with progress_container:
            progress_bar = st.progress(0)
        
        results = batch_export(
            video_path=st.session_state.download_result.video_path,
            clips=clip_configs,
            output_dir=output_dir,
            crop_mode=st.session_state.crop_mode,
            blur_zoom=st.session_state.blur_zoom
        )
        
        progress_bar.progress(0.5)

        words = st.session_state.words
        subtitle_style = st.session_state.subtitle_style

        if subtitle_style != "none":
            status_container.info("Generating subtitles...")
            progress_bar.progress(0.6)

            subtitle_files = []

            for clip_config, export_result in zip(clip_configs, results):
                if not export_result.success:
                    subtitle_files.append(None)
                    continue

                # Expand the word window by the same margins used during extraction.
                start_with_margin = max(0, clip_config.start - clip_config.margin_before)
                end_with_margin = clip_config.end + clip_config.margin_after

                clip_words = [
                    w for w in words
                    if start_with_margin <= w["start"] <= end_with_margin
                ]

                if clip_words:
                    # Normalise timestamps so t=0 aligns with the clip's actual start after margin.
                    adjusted_words = [
                        {
                            "word": w["word"],
                            "start": w["start"] - start_with_margin,
                            "end": w["end"] - start_with_margin
                        }
                        for w in clip_words
                    ]

                    viral_segments = create_viral_subtitle_segments(adjusted_words, max_words=3)

                    ass_path = export_result.output_path.with_suffix('.ass')
                    mask_info_or_path = export_ass(
                        segments=viral_segments,
                        output_path=ass_path,
                        style_type=subtitle_style,
                        words=adjusted_words,
                        font_size=st.session_state.font_size,
                        v_position_percent=st.session_state.subtitle_position,
                        # Hardcoded to 1920 — the canonical TikTok/Reels 9:16 height.
                        video_height=1920,
                        color_name=st.session_state.subtitle_color
                    )
                    subtitle_files.append((ass_path, mask_info_or_path))
                else:
                    subtitle_files.append(None)

            progress_bar.progress(0.8)

            # Pass 2: re-encode clips with subtitles burned in.
            status_container.info("Burning subtitles into videos...")

            results = batch_export(
                video_path=st.session_state.download_result.video_path,
                clips=clip_configs,
                output_dir=output_dir,
                crop_mode=st.session_state.crop_mode,
                blur_zoom=st.session_state.blur_zoom,
                subtitle_files=subtitle_files
            )

        else:
            # No subtitle burn-in: write plain SRT files for optional external use.
            status_container.info("Generating SRT subtitles...")
            progress_bar.progress(0.8)

            for clip_config, export_result in zip(clip_configs, results):
                if not export_result.success:
                    continue

                start_with_margin = max(0, clip_config.start - clip_config.margin_before)
                end_with_margin = clip_config.end + clip_config.margin_after

                clip_words = [
                    w for w in words
                    if start_with_margin <= w["start"] <= end_with_margin
                ]

                if clip_words:
                    adjusted_words = [
                        {
                            "word": w["word"],
                            "start": w["start"] - start_with_margin,
                            "end": w["end"] - start_with_margin
                        }
                        for w in clip_words
                    ]

                    subtitle_segments = create_subtitle_segments(adjusted_words, words_per_segment=4)
                    srt_path = export_result.output_path.with_suffix('.srt')
                    export_srt(subtitle_segments, srt_path)

        progress_bar.progress(1.0)
        status_container.success("Export complete!")

        st.session_state.export_results = results
        st.session_state.output_folder = output_dir

        st.rerun()
        
    except Exception as e:
        status_container.error(f"Export failed: {str(e)}")
        st.exception(e)


def _show_results():
    """
    Render the post-export results view with download options.

    Displays success metrics (clip count, subtitle files, success rate),
    a ZIP download button aggregating all MP4 and subtitle files, and
    per-clip expanders with inline video players and individual download
    buttons.  Offers navigation back to Step 2 or a full workflow reset.
    """
    st.success("Export completed successfully!")

    output_dir = st.session_state.output_folder
    results = st.session_state.export_results

    st.markdown("#### Results")
    
    files = sorted(output_dir.glob("*"))
    mp4_files = [f for f in files if f.suffix == ".mp4"]
    srt_files = [f for f in files if f.suffix == ".srt"]
    ass_files = [f for f in files if f.suffix == ".ass"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Video Clips", len(mp4_files))
    with col2:
        subtitle_count = len(srt_files) + len(ass_files)
        st.metric("Subtitle Files", subtitle_count)
    with col3:
        successful = sum(1 for r in results if r.success)
        st.metric("Success Rate", f"{successful}/{len(results)}")
    
    st.markdown("---")

    if mp4_files:
        # ZIP_DEFLATED provides a good compression ratio for mixed binary/text content.
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for mp4_file in mp4_files:
                zip_file.write(mp4_file, mp4_file.name)
            for srt_file in srt_files:
                zip_file.write(srt_file, srt_file.name)
            for ass_file in ass_files:
                zip_file.write(ass_file, ass_file.name)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="Download All Clips (ZIP)",
            data=zip_buffer,
            file_name=f"{output_dir.name}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )
    
    st.markdown("---")
    st.markdown("#### Exported Clips")
    
    analysis = st.session_state.analysis
    selected_indices = st.session_state.selected_clips
    clip_configs = [
        ClipConfig(
            start=analysis.clips[i].start,
            end=analysis.clips[i].end,
            title=analysis.clips[i].title,
            margin_before=config.MARGIN_BEFORE,
            margin_after=config.MARGIN_AFTER
        )
        for i in selected_indices
    ]
    
    for i, (clip_result, clip_config) in enumerate(zip(results, clip_configs), 1):
        if not clip_result.success:
            st.error(f"Clip {i} failed: {clip_result.error}")
            continue

        original_idx = selected_indices[i-1]
        clip_analysis = analysis.clips[original_idx]
        
        duration = clip_result.duration
        title = f"Clip {i}: {clip_config.title}"
        
        with st.expander(f"{title} ({duration:.0f}s)", expanded=False):
            col_video, col_info = st.columns([2, 1])
            
            with col_video:
                st.video(str(clip_result.output_path), format="video/mp4")
                st.markdown(
                    """
                    <style>
                    video {
                        max-height: 800px !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_info:
                st.metric("AI Score", f"{clip_analysis.score:.0%}")
                st.caption("**Reason**")
                st.caption(clip_analysis.reason)
                st.caption("**Hook**")
                st.caption(f"_{clip_analysis.hook}_")
                
                st.markdown("")

                with open(clip_result.output_path, 'rb') as f:
                    st.download_button(
                        label="Download Clip",
                        data=f,
                        file_name=clip_result.output_path.name,
                        mime="video/mp4",
                        use_container_width=True
                    )

                srt_path = clip_result.output_path.with_suffix('.srt')
                ass_path = clip_result.output_path.with_suffix('.ass')
                
                if ass_path.exists():
                    with open(ass_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="Download ASS",
                            data=f.read(),
                            file_name=ass_path.name,
                            mime="text/plain",
                            use_container_width=True
                        )
                elif srt_path.exists():
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="Download SRT",
                            data=f.read(),
                            file_name=srt_path.name,
                            mime="text/plain",
                            use_container_width=True
                        )
    
    st.markdown("---")

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    with col1:
        if st.button("← Back to Design", use_container_width=True):
            # Clearing export results forces _show_export_interface on the next render.
            st.session_state.export_results = None
            navigate_to_step(2)
    
    with col2:
        if st.button("Start New Project", use_container_width=True):
            from .shared import reset_workflow
            reset_workflow()