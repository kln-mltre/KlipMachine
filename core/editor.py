"""
Video editing module using FFmpeg.
Handles clip extraction, cropping, and subtitle generation.
"""

import subprocess
import shutil
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from core.exceptions import ExportError, FFmpegError, InvalidClipError
from config import config


def normalize_crop_mode(crop_mode: str) -> str:
    """
    Normalize crop mode names, including legacy aliases.

    Args:
        crop_mode: Raw crop mode value from UI, presets, or CLI.

    Returns:
        Canonical crop mode string.
    """
    if crop_mode is None:
        return "none"

    normalized = str(crop_mode).lower()
    if normalized == "center":
        return "black"
    if normalized in {"none", "blur", "black"}:
        return normalized
    return "none"

@dataclass
class ClipConfig:
    """Configuration for a single clip to extract."""
    start: float # Start time in seconds
    end: float   # End time in seconds
    title: str # Title for the clip
    margin_before: float 
    margin_after: float 

@dataclass 
class ExportResult:
    """Result of the export operation."""
    output_path: Path
    duration: float
    success: bool 
    error: Optional[str] = None

def verify_ffmpeg() -> bool:
    """
    Check if FFmpeg is installed and accessible.
    
    Returns:
        True if FFmpeg is available.
    """
    return shutil.which("ffmpeg") is not None

def verify_ffprobe() -> bool:
    """
    Check if ffprobe is installed and accessible.
    
    Returns:
        True if ffprobe is available
    """
    return shutil.which("ffprobe") is not None

def get_video_duration(video_path: Path) -> float:
    """
    Get video duration using ffprobe.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        Duration in seconds.
        
    Raises:
        FFmpegError: If ffprobe fails.
    """
    if not verify_ffprobe():
        raise FFmpegError("ffprobe not found. Install FFmpeg.")
    
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        raise FFmpegError(f"Failed to get video duration: {e}")


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """
    Get video resolution using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        FFmpegError: If ffprobe fails
    """
    if not verify_ffprobe():
        raise FFmpegError("ffprobe not found. Install FFmpeg.")
    
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        width, height = result.stdout.strip().split(',')
        return int(width), int(height)
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        raise FFmpegError(f"Failed to get video resolution: {e}")

def calculate_center_crop(
        src_width: int,
        src_height: int,
        target_width: int = 1080,
        target_height: int = 1920
) -> dict:
    """
    Calculate center crop parameters for 9:16 vertical format.
    
    Args:
        src_width: Source video width
        src_height: Source video height
        target_width: Target width (default 1080)
        target_height: Target height (default 1920)
        
    Returns:
        Dict with crop_w, crop_h, crop_x, crop_y
    """
    target_ratio = target_width / target_height #9/16 =  0.5625
    src_ratio = src_width / src_height

    if src_ratio > target_ratio:
        # Source is wider than target -> crop horizontally
        crop_h = src_height
        crop_w = int(src_height * target_ratio)
        crop_x = (src_width - crop_w) // 2
        crop_y = 0
    else:
        # Source is taller than target -> crop vertically
        crop_w = src_width
        crop_h = int(src_width / target_ratio)
        crop_x = 0
        crop_y = (src_height - crop_h) // 2

    return {
        "w": crop_w,
        "h": crop_h,
        "x": crop_x,
        "y": crop_y
    }

def build_blur_fill_filter(
        src_width: int,
        src_height: int,
        target_width: int = 1080,
        target_height: int = 1920,
        zoom: float = 1.08
) -> str:
    """
    Build FFmpeg filter for blur fill effect with zoom.
    Creates a blurred background from the video itself and overlays
    the zoomed original on top.
    
    Args:
        src_width: Source video width
        src_height: Source video height
        target_width: Target width (default 1080)
        target_height: Target height (default 1920)
        zoom: Zoom factor for main video (1.08 = 8% zoom)
        
    Returns:
        FFmpeg filter string
    """
    # Cap foreground height at 65% of the canvas, leaving ~17.5% blur padding on each side.
    max_video_height = int(target_height * 0.65)
    
    # Compute scaled dimensions while preserving source aspect ratio.
    src_ratio = src_width / src_height

    # Derive width from the height-capped constraint.
    scaled_height = max_video_height
    scaled_width = int(scaled_height * src_ratio)

    # Apply zoom factor after establishing the base scale.
    final_width = int(scaled_width * zoom)
    final_height = int(scaled_height * zoom)

    # Assemble the two-stream filter graph (background blur + scaled foreground).
    filter_complex = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height},"
        f"gblur=sigma=20[blurred];"
        f"[fg]scale={final_width}:{final_height}[scaled];"
        f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2:format=auto"
    )
    
    return filter_complex


def build_black_fill_filter(
        src_width: int,
        src_height: int,
        target_width: int = 1080,
        target_height: int = 1920,
        zoom: float = 1.08
) -> str:
    """
    Build FFmpeg filter for black fill effect with the same zoom logic as blur fill.

    This mode keeps the exact foreground sizing/positioning behavior of blur fill,
    but replaces the blurred background with a pure black canvas.

    Args:
        src_width: Source video width
        src_height: Source video height
        target_width: Target width (default 1080)
        target_height: Target height (default 1920)
        zoom: Zoom factor for main video (1.08 = 8% zoom)

    Returns:
        FFmpeg filter string
    """
    max_video_height = int(target_height * 0.65)

    src_ratio = src_width / src_height
    scaled_height = max_video_height
    scaled_width = int(scaled_height * src_ratio)

    final_width = int(scaled_width * zoom)
    final_height = int(scaled_height * zoom)

    filter_complex = (
        f"color=c=black:s={target_width}x{target_height},setpts=PTS-STARTPTS[bg];"
        f"[0:v]setpts=PTS-STARTPTS,scale={final_width}:{final_height}[scaled];"
        f"[bg][scaled]overlay=(W-w)/2:(H-h)/2:format=auto:shortest=1"
    )

    return filter_complex

def detect_hardware_acceleration() -> Optional[str]:
    """
    Detect available hardware acceleration.
    
    Returns:
        "nvenc" (NVIDIA), "videotoolbox" (macOS), "vaapi" (Linux), or None
    """
    # Check NVIDIA (nvenc)
    try:
        subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=True,
            timeout=2
        )
        return "nvenc"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check VideoToolbox (macOS)
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_videotoolbox" in result.stdout:
            return "videotoolbox"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # No hardware acceleration
    return None

def sanitize_filename(title: str) -> str:
    """
    Sanitize title for use in filename.
    
    Args:
        title: Original title
        
    Returns:
        Sanitized filename-safe string
    """
    # Remove special characters, keep alphanumeric and spaces
    sanitized = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Lowercase and limit reach
    return sanitized.lower()[:50]


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    crop_params: Optional[dict],  # May be None when no spatial transformation is required.
    blur_filter: Optional[str] = None,  # Pre-built blur filter string; takes priority over crop_params.
    hw_accel: Optional[str] = None
) -> list[str]:
    """
    Build FFmpeg command for clip extraction.
    
    Args:
        input_path: Source video file
        output_path: Destination file
        start: Start timestamp (seconds)
        duration: Clip duration (seconds)
        crop_params: Crop parameters from calculate_center_crop() or None for no crop
        blur_filter: Pre-built blur filter string or None
        hw_accel: Hardware acceleration ("nvenc", "videotoolbox", or None)
        
    Returns:
        FFmpeg command as list of strings
    """
    cmd = ["ffmpeg", "-y"]  # Overwrite output
    
    # Hardware acceleration (input)
    if hw_accel == "nvenc":
        cmd.extend(["-hwaccel", "cuda"])
    
    # Input file with seek
    cmd.extend([
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration)
    ])
    
    # Video filter - priority: blur > crop > none
    if blur_filter:
        cmd.extend(["-filter_complex", blur_filter])
    elif crop_params:
        vf = (
            f"crop={crop_params['w']}:{crop_params['h']}:"
            f"{crop_params['x']}:{crop_params['y']},"
            f"scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}"
        )
        cmd.extend(["-vf", vf])
    # Otherwise, preserve original format (no crop)
    
    # Video codec
    if hw_accel == "nvenc":
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
    elif hw_accel == "videotoolbox":
        cmd.extend(["-c:v", "h264_videotoolbox", "-b:v", "5M"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
    
    # Audio codec
    cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    
    # Optimization
    cmd.extend(["-movflags", "+faststart"])
    
    # Output
    cmd.append(str(output_path))
    
    return cmd

def extract_clip(
    video_path: Path,
    clip: ClipConfig,
    output_path: Path,
    crop_mode: str = "black",  # "black", "blur", or "none"
    blur_zoom: float = 1.08,
    subtitle_file: Optional[Path] = None,  # .ass subtitle file to burn into the clip
    mask_info: Optional[dict] = None,  # Per-word box-highlight timing metadata
    hw_accel: Optional[str] = None,
    verbose: bool = False
) -> ExportResult:
    """
    Extract and export a single clip from a source video.

    Applies time-bounded extraction with optional margins, spatial transformation
    (black fill or blur fill), subtitle burn-in, and hardware acceleration.

    Args:
        video_path: Path to the source video file.
        clip: ClipConfig instance carrying timestamps, title, and margin values.
        output_path: Destination path for the exported .mp4 file.
        crop_mode: Spatial transform mode — "black" (black fill), "blur" (blur fill), or "none".
        blur_zoom: Zoom factor applied in black/blur fill modes (default 1.08 = 8% zoom).
        subtitle_file: Optional path to an .ass file to burn into the output.
        mask_info: Optional per-word timing dict for box-highlight overlay pipeline.
        hw_accel: Hardware encoder identifier ("nvenc", "videotoolbox", or None).
        verbose: When True, forwards FFmpeg stdout/stderr to the console.

    Returns:
        ExportResult carrying the output path, duration, success flag, and any error message.
    """
    if not verify_ffmpeg():
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")
    
    if clip.start < 0:
        raise InvalidClipError(f"Start timestamp cannot be negative: {clip.start}")
    
    if clip.end <= clip.start:
        raise InvalidClipError(
            f"End timestamp ({clip.end}) must be after start ({clip.start})"
        )
    
    # Apply margins
    start = max(0, clip.start - clip.margin_before)
    end = clip.end + clip.margin_after
    duration = end - start
    
    if duration <= 0:
        raise InvalidClipError(f"Clip duration is negative: {duration}")
    
    src_width, src_height = get_video_resolution(video_path)
    
    crop_mode = normalize_crop_mode(crop_mode)

    crop_params = None
    blur_filter = None
    
    if crop_mode == "black":
        blur_filter = build_black_fill_filter(src_width, src_height, zoom=blur_zoom)
    elif crop_mode == "blur":
        blur_filter = build_blur_fill_filter(src_width, src_height,zoom=blur_zoom)
    # else: mode "none" — pass video through without spatial modification.

    if subtitle_file and subtitle_file.exists():
        # Use the box-highlight pipeline when per-word timing metadata is present.
        if mask_info and "words" in mask_info:
            from core.transcriber import generate_highlight_box_masks

            # Generate per-word PNG mask frames in a dedicated subdirectory.
            mask_dir = output_path.parent / f"masks_{output_path.stem}"
            mask_paths = generate_highlight_box_masks(
                words=mask_info["words"],
                output_dir=mask_dir,
                segments=mask_info.get("segments"),
                font_size=mask_info["font_size"],
                v_position_percent=mask_info["v_position_percent"],
                video_height=mask_info.get("video_height", 1920),
                box_color=mask_info.get("box_color", (255, 242, 204, 255))  # Fall back to soft-yellow if not specified.
            )
            
            cmd = build_ffmpeg_command_with_box_highlights(
                input_path=video_path,
                output_path=output_path,
                start=start,
                duration=duration,
                subtitle_path=subtitle_file,
                mask_info=mask_info,
                mask_paths=mask_paths,
                crop_params=crop_params,
                blur_filter=blur_filter,
                hw_accel=hw_accel
            )
        else:
            cmd = build_ffmpeg_command_with_subtitles(
                input_path=video_path,
                output_path=output_path,
                start=start,
                duration=duration,
                subtitle_path=subtitle_file,
                crop_params=crop_params,
                blur_filter=blur_filter,
                hw_accel=hw_accel
            )
    else:
        cmd = build_ffmpeg_command(
            input_path=video_path,
            output_path=output_path,
            start=start,
            duration=duration,
            crop_params=crop_params,
            blur_filter=blur_filter,
            hw_accel=hw_accel
        )
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
            timeout=300  # 5 minutes max per clip
        )
        
        # Get actual duration from exported file
        try:
            actual_duration = get_video_duration(output_path)
        except Exception:
            actual_duration = duration  # Fall back to computed duration if ffprobe cannot read output.
        
        return ExportResult(
            output_path=output_path,
            duration=actual_duration,
            success=True
        )
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return ExportResult(
            output_path=output_path,
            duration=duration,
            success=False,
            error=f"FFmpeg failed: {error_msg}"
        )
    
    except subprocess.TimeoutExpired:
        return ExportResult(
            output_path=output_path,
            duration=duration,
            success=False,
            error="FFmpeg timeout (>5 minutes)"
        )
    

def batch_export(
    video_path: Path,
    clips: list[ClipConfig],
    output_dir: Path,
    crop_mode: str = "black",  
    blur_zoom: float = 1.08,
    subtitle_files: Optional[list[Optional[tuple[Path, dict]]]] = None,  # (ass_path, mask_info)
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> list[ExportResult]:
    """
    Export multiple clips from a video.
    
    Args:
        video_path: Source video file
        clips: List of clip configurations
        output_dir: Output directory
        crop_mode: "black"/"blur" (9:16 fill) or "none" (preserve original)
        progress_callback: Optional callback(current, total)
        
    Returns:
        List of ExportResult
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crop_mode = normalize_crop_mode(crop_mode)

    # Detect hardware acceleration once and reuse for all clips in the batch.
    hw_accel = detect_hardware_acceleration()
    if hw_accel:
        print(f"[OK] Hardware acceleration: {hw_accel}")
    else:
        print("[INFO] Using software encoding (no hardware acceleration detected)")

    # Report active crop mode for operator awareness.
    if crop_mode == "blur":
        print("[INFO] Crop mode: blur fill 9:16 (1080x1920)")
    elif crop_mode == "black":
        print("[INFO] Crop mode: black fill 9:16 (1080x1920)")
    else:
        print("[INFO] Crop mode: original format (no crop)")
    
    results = []
    
    for i, clip in enumerate(clips, start=1):
        if progress_callback:
            progress_callback(i, len(clips))
        
        safe_title = sanitize_filename(clip.title)
        filename = f"clip_{i:02d}_{safe_title}.mp4"
        output_path = output_dir / filename
        
        print(f"[BUSY] Exporting clip {i}/{len(clips)}: {clip.title}...")
        
        # Export clip (with subtitle if provided)
        subtitle_data = subtitle_files[i-1] if subtitle_files and i-1 < len(subtitle_files) else (None, None)
        subtitle_file = subtitle_data[0] if subtitle_data else None
        mask_info = subtitle_data[1] if subtitle_data else None
        
        result = extract_clip(
            video_path=video_path,
            clip=clip,
            output_path=output_path,
            crop_mode=crop_mode,
            blur_zoom=blur_zoom,
            subtitle_file=subtitle_file,
            mask_info=mask_info,  
            hw_accel=hw_accel,
            verbose=False
        )
        
        if result.success:
            print(f"[SUCCESS] Saved: {filename} ({result.duration:.1f}s)")
        else:
            print(f"[ERROR] {result.error}")
        
        results.append(result)
    
    return results


#===========================================================================#
# V 1.5.0
#===========================================================================#

def build_ffmpeg_command_with_subtitles(
    input_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    subtitle_path: Path,
    crop_params: Optional[dict] = None,
    blur_filter: Optional[str] = None,
    hw_accel: Optional[str] = None
) -> list[str]:
    """
    Build FFmpeg command with burned-in ASS subtitles.
    
    Args:
        input_path: Source video file
        output_path: Destination file
        start: Start timestamp (seconds)
        duration: Clip duration (seconds)
        subtitle_path: Path to .ass subtitle file
        crop_params: Crop parameters or None
        blur_filter: Pre-built blur filter or None
        hw_accel: Hardware acceleration ("nvenc", "videotoolbox", or None)
        
    Returns:
        FFmpeg command as list of strings
    """
    cmd = ["ffmpeg", "-y"]

    if hw_accel == "nvenc":
        cmd.extend(["-hwaccel", "cuda"])
    
    cmd.extend([
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration)
    ])

    subtitle_str = str(subtitle_path).replace('\\', '/').replace(':', r'\:')
    
    if blur_filter:
        # Blur mode: append subtitle after blur overlay
        filter_complex = f"{blur_filter},ass={subtitle_str}"
        cmd.extend(["-filter_complex", filter_complex])
    elif crop_params:
        # Center crop mode: crop -> scale -> subtitle
        filters = [
            f"crop={crop_params['w']}:{crop_params['h']}:{crop_params['x']}:{crop_params['y']}",
            f"scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}",
            f"ass={subtitle_str}"
        ]
        cmd.extend(["-vf", ",".join(filters)])
    else:
        # No crop mode: just subtitle
        cmd.extend(["-vf", f"ass={subtitle_str}"])

    # Video codec
    if hw_accel == "nvenc":
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
    elif hw_accel == "videotoolbox":
        cmd.extend(["-c:v", "h264_videotoolbox", "-b:v", "5M"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
    
    # Audio codec
    cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    
    # Optimization
    cmd.extend(["-movflags", "+faststart"])
    
    # Output
    cmd.append(str(output_path))
    
    return cmd



def build_ffmpeg_command_with_box_highlights(
    input_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    subtitle_path: Path,
    mask_info: dict,
    mask_paths: dict,
    crop_params: Optional[dict] = None,
    blur_filter: Optional[str] = None,
    hw_accel: Optional[str] = None,
    v_position_percent: int = 85,
    video_height: int = 1920
) -> list[str]:
    """
    Build an FFmpeg command combining per-word PNG box overlays with burned-in ASS subtitles.

    Filter graph pipeline:
        1. Prepare base video stream (crop or blur fill as requested) → [vbase].
        2. Chain per-word PNG mask overlays with a 250 ms ease-out cubic slide animation.
        3. Burn-in the ASS subtitle file on top of all overlay layers.

    Args:
        input_path: Path to the source video file.
        output_path: Destination path for the exported .mp4 file.
        start: Seek position in the source video (seconds).
        duration: Length of the output clip (seconds).
        subtitle_path: Path to the .ass subtitle file for text burn-in.
        mask_info: Word timing dict produced by export_ass(), must contain a ``words`` list.
        mask_paths: Mapping of {word_index: Path} for pre-rendered PNG overlay frames (1080x1920 canvas).
        crop_params: Crop parameter dict from calculate_center_crop(), or None.
        blur_filter: Pre-built blur fill filter string, or None.
        hw_accel: Hardware encoder identifier ("nvenc", "videotoolbox", or None).
        v_position_percent: Reserved — vertical anchor from top (0–100) for future layout use.
        video_height: Canvas height used for overlay positioning.

    Returns:
        Complete FFmpeg command as a list of argument strings.
    """
    cmd = ["ffmpeg", "-y"]
    
    cmd.extend([
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration)
    ])

    # Inputs 1..N: per-word PNG mask frames, sorted by word index to match timing data.
    sorted_mask_keys = sorted(mask_paths.keys())

    for i, key in enumerate(sorted_mask_keys):
        mask_path = mask_paths[key]
        cmd.extend(["-i", str(mask_path)])
    
    # Extract the filter body from a pre-built blur string (strips the leading [0:v] label).
    if blur_filter:
        filter_complex = f"[0:v]{blur_filter.split('[0:v]')[1]}[vbase];"
    elif crop_params:
        # Center-crop then scale to the target canvas dimensions.
        filter_complex = (
            f"[0:v]crop={crop_params['w']}:{crop_params['h']}:"
            f"{crop_params['x']}:{crop_params['y']},"
            f"scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}[vbase];"
        )
    else:
        # No spatial transform — scale to target resolution only.
        filter_complex = f"[0:v]scale={config.OUTPUT_WIDTH}:{config.OUTPUT_HEIGHT}[vbase];"
    
    # Each mask glides from x=-10 to x=0 over 250 ms using a cubic ease-out curve,
    # producing a subtle but perceptibly smooth entry without jarring motion.
    current_label = "[vbase]"
    num_masks = len(sorted_mask_keys)

    for i, word_data in enumerate(mask_info["words"]):

        if i >= num_masks:
            break  # Guard: skip words that have no corresponding mask frame.

        ffmpeg_input_idx = i + 1

        word_start = word_data["start"]  
        word_end = word_data["end"]
        
        mask_input = f"[{ffmpeg_input_idx}:v]"
        next_label = f"[v{i}]"
        
        # Ease-out cubic horizontal offset: x travels from -10 to 0 over 250 ms.
        # Formula: offset = -10 + 10 * (1 - (1 - progress)^3), where progress = (t - start) / 0.25.
        x_expr = (
            f"if(lt(t\\,{word_start}+0.25)\\,"
            f"-10+10*(1-pow(1-(t-{word_start})/0.25\\,3))\\,"
            f"0)"
        )
        
        filter_complex += (
            f"{current_label}{mask_input}"
            f"overlay=x='{x_expr}':y=0:format=auto:enable='between(t\\,{word_start:.3f}\\,{word_end:.3f})'"
            f"{next_label};"
        )
        
        current_label = next_label
    
    # Colons in the path must be escaped for FFmpeg's filter syntax.
    subtitle_str = str(subtitle_path).replace('\\', '/').replace(':', r'\:')
    filter_complex += f"{current_label}ass={subtitle_str}[out]"
    
    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", "[out]"])
    cmd.extend(["-map", "0:a"])  

    # Select video encoder based on available hardware acceleration.
    if hw_accel == "nvenc":
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
    elif hw_accel == "videotoolbox":
        cmd.extend(["-c:v", "h264_videotoolbox", "-b:v", "5M"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
    
    # Audio codec
    cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    
    # Optimization
    cmd.extend(["-movflags", "+faststart"])
    
    # Output
    cmd.append(str(output_path))
    
    return cmd
