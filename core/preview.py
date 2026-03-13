"""
Preview generation module.
Handles frame extraction and preview rendering with filters and subtitles.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from core.exceptions import FFmpegError
from core.editor import (
    verify_ffmpeg,
    verify_ffprobe,
    get_video_resolution,
    build_blur_fill_filter,
    build_black_fill_filter,
    normalize_crop_mode
)
from config import config


def generate_preview_frame(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    crop_mode: str = "none",
    blur_zoom: float = 1.08,
    target_width: int = 1080,
    target_height: int = 1920,
    subtitle_path: Optional[Path] = None,
    mask_paths: Optional[dict] = None,
    mask_info: Optional[dict] = None,
    show_tiktok_ui: bool = False
) -> bool:
    """
    Extract a single video frame with applied spatial filters and optional subtitle burn-in.

    Designed for near-real-time preview rendering: uses input-side seeking (-ss before -i)
    to avoid full decode overhead and targets a <500 ms wall-clock render time.

    Args:
        video_path: Path to the source video file.
        timestamp: Time position to sample, in seconds.
        output_path: Destination image file path (JPEG recommended for speed).
        crop_mode: Spatial transform mode — "none", "black" (black fill), or "blur" (blur fill).
        blur_zoom: Zoom factor applied in black/blur fill modes (default 1.08 = 8 % zoom).
        target_width: Output frame width in pixels (default 1080).
        target_height: Output frame height in pixels (default 1920).
        subtitle_path: Optional path to an .ass subtitle file to burn into the frame.
        mask_paths: Optional mapping of {word_index: Path} for PNG box-highlight overlays.
        mask_info: Optional per-word timing dict required when mask_paths is provided.
        show_tiktok_ui: Reserved for future TikTok UI chrome overlay (currently unused).

    Returns:
        True if the frame was extracted and written successfully.

    Raises:
        FFmpegError: If FFmpeg is not available, the extraction fails, or the command times out.
    """
    if not verify_ffmpeg():
        raise FFmpegError("FFmpeg not found. Install FFmpeg.")

    src_width, src_height = get_video_resolution(video_path)
    crop_mode = normalize_crop_mode(crop_mode)
    
    # Input-side seek (-ss before -i) skips decoding frames before the target, reducing latency.
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),  # Seek before input for fast frame access.
        "-i", str(video_path)
    ]

    subtitle_filter = ""

    # Box-highlight style uses a dedicated filter_complex path handled in the block below.
    if mask_paths and mask_info:
        pass  # Filter construction is deferred to the box-highlight branch.
    elif subtitle_path and subtitle_path.exists():
        # Standard styles (glow, pop): append ASS filter directly to the video filter chain.
        subtitle_str = str(subtitle_path).replace('\\', '/').replace(':', r'\:')
        subtitle_filter = f",ass={subtitle_str}"
    
    # Box-highlight pipeline (per-word PNG overlays + ASS burn-in)
    if mask_paths and mask_info:
        # Load each PNG mask as a separate FFmpeg input (one input per word).
        for i, mask_path in sorted(mask_paths.items()):
            cmd.extend(["-i", str(mask_path)])

        cmd.extend(["-vframes", "1"])

        # Prepare the base video stream according to the active crop mode.
        if crop_mode == "blur":
            # Blur fill: build a blurred background with the zoomed original overlaid.
            blur_filter = build_blur_fill_filter(
                src_width,
                src_height,
                target_width=target_width,
                target_height=target_height,
                zoom=blur_zoom
            )
            filter_complex = f"{blur_filter}[base];"
            base_video = "[base]"

        elif crop_mode == "black":
            # Black fill: same foreground sizing/zoom as blur mode, but over a black background.
            black_filter = build_black_fill_filter(
                src_width,
                src_height,
                target_width=target_width,
                target_height=target_height,
                zoom=blur_zoom
            )
            filter_complex = f"{black_filter}[base];"
            base_video = "[base]"

        else:
            # No spatial transform; use the raw decoded stream directly.
            filter_complex = ""
            base_video = "[0:v]"

        # Identify which word (if any) is active at the requested timestamp.
        mask_idx = None
        for i, word_data in enumerate(mask_info["words"]):
            if word_data["start"] <= timestamp <= word_data["end"]:
                mask_idx = i
                break

        # Overlay the corresponding mask frame when a word is active at this timestamp.
        if mask_idx is not None and mask_idx in mask_paths:
            mask_input = f"[{mask_idx + 1}:v]"  # Input index offset by 1; input 0 is the video.

            filter_complex += (
                f"{base_video}{mask_input}"
                f"overlay=x=0:y=0[with_mask];"
            )
            has_mask = True
        else:
            # No word is active at this timestamp; pass the base stream through unchanged.
            filter_complex += f"{base_video}null[with_mask];"
            has_mask = False

        # ASS subtitle burn-in must come last so text renders above all overlay layers.
        if subtitle_path and subtitle_path.exists():
            subtitle_str = str(subtitle_path).replace('\\', '/').replace(':', r'\:')
            filter_complex += f"[with_mask]ass={subtitle_str}[out]"
            output_label = "[out]"
        else:
            output_label = "[with_mask]"

        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", output_label])
    
    # --- Standard subtitle styles (glow, pop, none) ---
    else:
        cmd.extend(["-vframes", "1"])

        subtitle_filter = ""
        if subtitle_path and subtitle_path.exists():
            # Colon characters must be escaped for FFmpeg's filter option parser.
            subtitle_str = str(subtitle_path).replace('\\', '/').replace(':', r'\:')
            subtitle_filter = f",ass={subtitle_str}"
        
        if crop_mode == "blur":
            # Blur fill: blurred background with the zoomed original centred on top.
            blur_filter = build_blur_fill_filter(
                src_width,
                src_height,
                target_width=target_width,
                target_height=target_height,
                zoom=blur_zoom
            )
            filter_complex = f"{blur_filter}{subtitle_filter}"
            cmd.extend(["-filter_complex", filter_complex])

        elif crop_mode == "black":
            # Black fill: same foreground sizing/zoom as blur mode, but over a black background.
            black_filter = build_black_fill_filter(
                src_width,
                src_height,
                target_width=target_width,
                target_height=target_height,
                zoom=blur_zoom
            )
            filter_complex = f"{black_filter}{subtitle_filter}"
            cmd.extend(["-filter_complex", filter_complex])

        else:  # none
            # Preserve original aspect ratio with letterboxing; no crop applied.
            vf = f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease{subtitle_filter}"
            cmd.extend(["-vf", vf])
    
    # JPEG quality scale: 1 (best) to 31 (worst). Value 2 gives near-lossless quality at low file size.
    cmd.extend([
        "-q:v", "2",
        str(output_path)
    ])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=10
        )

        return True
    
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise FFmpegError(f"Preview generation failed: {error_msg}")
    
    except subprocess.TimeoutExpired:
        raise FFmpegError("Preview generation timeout (>10s)")
    

