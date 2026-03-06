"""
Audio transcription module using Faster-Whisper.
Converts audio to text with word-level timestamps.
"""

import json
import string
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict 
from typing import Optional

import cv2
import numpy as np
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont

from core.exceptions import TranscriptionError, AudioNotFoundError

# Whisper model configurations
WHISPER_MODELS = {
    "base" : {
        "compute_type": "int8",  # Fast, less accurate
        "beam_size": 5,
    },
    "small": {
        "compute_type": "float16", # Balanced speed and accuracy
        "beam_size": 5,
    }
}

# Global model cache — singleton pattern prevents redundant downloads after the first load.
_model_cache: dict[str, WhisperModel] = {}

@dataclass
class TranscriptSegment:
    """Single Segment of transcribed text with timing."""
    start: float  # seconds
    end: float    # seconds
    text: str

@dataclass
class TranscriptResult:
    """Complete transcription result."""
    segments: list[TranscriptSegment]
    language: str   # Detected language code (e.g., "en")
    duration: float # Total duration in seconds

def load_model(model_size: str, device: str = "cpu") -> WhisperModel:
    """
    Load a Whisper model, returning a cached instance on subsequent calls.

    The first call triggers a model download if not already present locally;
    all subsequent calls for the same (model_size, device) pair return the
    cached instance immediately.

    Args:
        model_size: Whisper model variant — "base" (fast) or "small" (accurate).
        device: Compute device — "cpu" or "cuda".

    Returns:
        Loaded WhisperModel instance.

    Raises:
        TranscriptionError: If the model size is invalid or loading fails.
    """
    cache_key = f"{model_size}_{device}"

    # Return cached model if available.
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Validate model size before attempting to load.
    if model_size not in WHISPER_MODELS:
        raise TranscriptionError(
            f"Invalid model size '{model_size}'"
            f"Must be one of: {list(WHISPER_MODELS.keys())}"
        )
    
    model_config = WHISPER_MODELS[model_size]
    compute_type = model_config["compute_type"]

    # float16 is unsupported on CPU; fall back to int8 to avoid a runtime error.
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    try:
        print(f"[BUSY] Loading Whisper model '{model_size}' on {device}...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        _model_cache[cache_key] = model
        print("[OK] Model loaded successfully.")

        return model
    
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model: {e}")
    

def transcribe(
    audio_path: Path,
    model_size: str = 'base',
    language: Optional[str] = None,
    device: str = "auto"
) -> tuple[TranscriptResult, list[dict]]:
    """
    Transcribe an audio file to text with word-level timestamps.

    Args:
        audio_path: Path to the audio file to transcribe.
        model_size: Whisper model variant — "base" (fast) or "small" (accurate).
        language: BCP-47 language code (e.g. "fr", "en"), or None for auto-detection.
        device: Compute device — "cuda", "cpu", or "auto" (resolved via config).

    Returns:
        A tuple of:
            - TranscriptResult: Segment-level transcription with language and duration.
            - list[dict]: Word-level timestamps as {"word": str, "start": float, "end": float}.

    Raises:
        AudioNotFoundError: If the audio file does not exist at the given path.
        TranscriptionError: If the Whisper transcription pipeline fails.
    """
    # Validate audio file
    if not audio_path.exists():
        raise AudioNotFoundError(f"Audio file not found: {audio_path}")
    
    # Auto-detect device
    if device == "auto":
        from config import config
        device = config.WHISPER_DEVICE 

    # Load model (cached after first call)
    model = load_model(model_size, device)

    # Get beam size from model config
    beam_size = WHISPER_MODELS[model_size]["beam_size"]

    try:
        print(f"[BUSY] Transcribing {audio_path.name}...")

        # VAD filter suppresses non-speech silence, reducing hallucination on quiet segments.
        segments_iter, info = model.transcribe(
            str(audio_path),
            language = language,
            beam_size = beam_size,
            vad_filter=True,
            word_timestamps=True
        )

        # Convert iterator to segments + extract words
        segments = []
        all_words = []


        for segment in segments_iter:
            segments.append(TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip()
            ))

            # Extract word timestamps
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    all_words.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end
                    })
            
        # Get total duration (from last segment)
        duration = segments[-1].end if segments else 0.0

        print(f"[OK] Transcription complete: {len(segments)} segments, "
              f"{len(all_words)} words, {duration:.1f}s, language: {info.language}")
        
        result = TranscriptResult(
            segments=segments,
            language=info.language,
            duration=duration
        )
        
        return result, all_words
        
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")
    

def segments_to_text(segments: list[TranscriptSegment]) -> str:
    """
    Convert segments to formatted text with timestamps.

    Format: [HH:MM:SS] Text content
    
    Args:
        segments: List of TranscriptSegment

    Returns:
        Formatted text string
    """
    lines = []

    for segment in segments:
        # Convert seconds to HH:MM:SS or MM:SS
        hours = int(segment.start // 3600)
        minutes = int((segment.start % 3600) // 60)
        seconds = int(segment.start % 60)

        if hours > 0:
            timestamp = f"[{hours:02}:{minutes:02}:{seconds:02}]"
        else:
            timestamp = f"[{minutes:02}:{seconds:02}]"

        lines.append(f"{timestamp} {segment.text}")
    
    return "\n".join(lines)

def export_transcript(
    result: TranscriptResult,
    words: list[dict],
    output_path: Path
) -> None:
    """
    Persist a transcription result to a JSON file.

    Stores both segment-level and word-level data in a single document so that
    downstream tools can reload the full result without re-running Whisper.

    Args:
        result: TranscriptResult containing segments, language, and duration.
        words: Word-level timestamp list as produced by transcribe().
        output_path: Destination path for the JSON output file.
    """
    data = {
        "language": result.language,
        "duration": result.duration,
        "segments": [asdict(seg) for seg in result.segments],
        "words": words
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Transcript saved to {output_path}")

def load_transcript(json_path: Path) -> tuple[TranscriptResult, list[dict]]:
    """
    Reload a previously exported transcription from a JSON file.

    Avoids re-running the Whisper model when the transcript already exists on disk.
    Handles files that pre-date word-level storage by defaulting to an empty list.

    Args:
        json_path: Path to a JSON transcript file produced by export_transcript().

    Returns:
        A tuple of:
            - TranscriptResult: Reconstructed segment-level transcription.
            - list[dict]: Word-level timestamps, or an empty list if absent.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = [
        TranscriptSegment(**seg) 
        for seg in data["segments"]
    ]
    
    result = TranscriptResult(
        segments=segments,
        language=data["language"],
        duration=data["duration"]
    )
    
    # Fall back to an empty list for transcripts saved before word-level support was added.
    words = data.get("words", [])
    
    return result, words


# ===============================================================
# ===============================================================

def create_analysis_segments(
    words: list[dict],
    max_duration: float = 45.0,
    silence_threshold: float = 1.5
) -> list[TranscriptSegment]:
    """
    Group words into long segments suitable for AI content analysis.

    Splits are triggered by either a silence gap between words or by reaching
    the maximum segment duration, whichever comes first.

    Args:
        words: List of word dicts with "word", "start", and "end" keys.
        max_duration: Maximum segment length in seconds (30–60 s recommended).
        silence_threshold: Minimum inter-word pause that triggers a split (seconds).

    Returns:
        List of TranscriptSegment instances sized for analysis (30–45 s each).
    """
    if not words:
        return []
    
    segments = []
    current_words = []
    current_start = words[0]["start"]

    for i, word in enumerate(words):
        current_words.append(word)

        # Check if we should split here
        should_split = False

        # Check duration
        current_duration = word["end"] - current_start
        if current_duration >= max_duration:
            should_split = True

        # Split when the pause to the next word meets the silence threshold.
        if i < len(words) - 1:
            next_word = words[i + 1]
            silence_gap = next_word["start"] - word["end"]
            if silence_gap >= silence_threshold:
                should_split = True

        # Always flush the buffer on the final word.
        if i == len(words) - 1:
            should_split = True

        if should_split and current_words:
            text = " ".join(w["word"] for w in current_words)
            segments.append(TranscriptSegment(
                start=current_start,
                end=current_words[-1]["end"],
                text=text
            ))

            # Reset for next segment
            if i < len(words) - 1:
                current_words = []
                current_start = words[i + 1]["start"]

    return segments


def create_subtitle_segments(
    words: list[dict],
    words_per_segment: int = 4
) -> list[TranscriptSegment]:
    """
    Partition words into fixed-size subtitle segments.

    Args:
        words: List of word dicts with "word", "start", and "end" keys.
        words_per_segment: Number of words per subtitle block (default 4).

    Returns:
        List of short TranscriptSegment instances suitable for subtitle display.
    """
    segments = []

    for i in range(0, len(words), words_per_segment):
        chunk = words[i:i + words_per_segment]

        if not chunk:
            continue

        text = " ".join(w["word"] for w in chunk)
        start = chunk[0]["start"]
        end = chunk[-1]["end"]

        segments.append(TranscriptSegment(
            start=start,
            end=end,
            text=text
        ))

    return segments


def export_srt(segments: list[TranscriptSegment], output_path: Path) -> None:
    """
    Write subtitle segments to an SRT file.

    SRT timestamps use the format HH:MM:SS,mmm as required by the standard.
    The output is compatible with CapCut, VLC, and most NLE tools.

    Args:
        segments: List of TranscriptSegment instances to serialise.
        output_path: Destination .srt file path.
    """
    lines = []

    for i, segment in enumerate(segments, start=1):
        # Convert to SRT timestamp format: HH:MM:SS,mmm
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)

        # Decompose millisecond totals to avoid floating-point drift in formatted output.
        start_h = start_ms // 3600000
        start_m = (start_ms % 3600000) // 60000
        start_s = (start_ms % 60000) // 1000
        start_ms_only = start_ms % 1000

        end_h = end_ms // 3600000
        end_m = (end_ms % 3600000) // 60000
        end_s = (end_ms % 60000) // 1000
        end_ms_only = end_ms % 1000

        start_time = f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms_only:03d}"
        end_time = f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms_only:03d}"

        lines.append(f"{i}")
        lines.append(f"{start_time} --> {end_time}")
        lines.append(f"{segment.text}\n")
        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"[SUCCESS] SRT subtitles saved to {output_path}")


def export_analysis_segments(
    segments: list[TranscriptSegment],
    output_path: Path
) -> None:
    """
    Write long analysis segments to a plain-text file.

    Each entry is prefixed with a segment index, its duration, and a
    formatted timestamp to facilitate manual review and AI prompting.

    Args:
        segments: List of analysis-sized TranscriptSegment instances (30–45 s each).
        output_path: Destination .txt file path.
    """
    lines = []
    
    for i, segment in enumerate(segments, start=1):
        hours = int(segment.start // 3600)
        minutes = int((segment.start % 3600) // 60)
        seconds = int(segment.start % 60)

        if hours > 0:
            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
        else:
            timestamp = f"[{minutes:02d}:{seconds:02d}]"

        duration = segment.end - segment.start
        lines.append(f"=== Segment {i} ({duration:.1f}s) ===")
        lines.append(f"{timestamp} {segment.text}")
        lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"[SUCCESS] Analysis segments saved to {output_path}")


def export_subtitle_segments_txt(
    segments: list[TranscriptSegment],
    output_path: Path
) -> None:
    """
    Write short subtitle segments to a plain-text file for quick inspection.

    Each row shows the segment index, start/end times, and text aligned in
    a fixed-width tabular format.

    Args:
        segments: List of subtitle-sized TranscriptSegment instances (3–5 words each).
        output_path: Destination .txt file path.
    """
    lines = []
    
    for i, segment in enumerate(segments, start=1):
        start_str = f"{segment.start:.2f}s"
        end_str = f"{segment.end:.2f}s"

        lines.append(f"{i:4d} | {start_str:>8s} - {end_str:>8s} | {segment.text}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"[SUCCESS] Subtitle segments saved to {output_path}")

#=============================================================================
# V 1.5.0
#=============================================================================

    
def seconds_to_ass_time(seconds: float) -> str:
    """
    Convert seconds to ASS subtitle format (H:MM:SS.CC).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "0:01:23.45")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int((secs % 1) * 100)

    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"


def generate_ass_header(
    style_type: str,
    font_size: int = 28,
    v_position_percent: int = 85,
    video_height: int = 1920
) -> str:
    """
    Generate ASS subtitle header with dynamic positioning.
    
    Args:
        style_type: "colors" or "box"
        font_size: Base font size in pixels (20-60, default 28)
        v_position_percent: Vertical position from top (0-100, default 85)
        video_height: Video height for MarginV calculation (default 1920)
        
    Returns:
        Complete ASS header string
    """
    # Calculate MarginV
    margin_v = int(video_height * (1 - v_position_percent / 100))
    
    header = """[Script Info]
Title: KlipMachine Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
    
    # Spacing=0 keeps characters tightly packed, matching the intended render style.
    header += f"Style: Default,Montserrat ExtraBold,{font_size},&HFFFFFF,&HFFFFFF,&H000000,&H000000,-1,0,0,0,100,100,0,0,1,1,1,2,10,10,{margin_v},1\n"
    
    header += "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    
    return header


def merge_apostrophe_words(words: list[dict]) -> list[dict]:
    """
    Merge word tokens that are split across an apostrophe boundary.

    Whisper sometimes emits contractions as two separate tokens (e.g. "c'" and
    "est" instead of "c'est"). This function re-joins such pairs so that each
    contraction is treated as a single timed word.

    Two cases are handled:
        - The current token ends with an apostrophe  ("c'" + "est" → "c'est").
        - The next token starts with an apostrophe   ("c" + "'est" → "c'est").

    Args:
        words: List of word dicts with "word", "start", and "end" keys.

    Returns:
        New list of word dicts with apostrophe-split pairs merged.
    """
    if not words:
        return []
    
    merged = []
    i = 0
    
    while i < len(words):
        current = words[i]
        word_text = current["word"].strip()

        # Case 1: current token ends with apostrophe ("c'" + "est" → "c'est").
        if i + 1 < len(words) and word_text.endswith("'"):
            next_word = words[i + 1]
            merged.append({
                "word": word_text + next_word["word"].strip(),
                "start": current["start"],
                "end": next_word["end"]
            })
            i += 2
        # Case 2: next token starts with apostrophe ("c" + "'est" → "c'est").
        elif i + 1 < len(words) and words[i + 1]["word"].strip().startswith("'"):
            next_word = words[i + 1]
            merged.append({
                "word": word_text + next_word["word"].strip(),
                "start": current["start"],
                "end": next_word["end"]
            })
            i += 2
        else:
            merged.append(current)
            i += 1
    
    return merged


def _get_ffmpeg_text_bbox(
    text: str = None,
    font_size: int = 80,
    video_width: int = 1080,
    video_height: int = 1920,
    v_position_percent: int = 85,
    spacing: int = 0,
    full_phrase: list[str] = None,
    word_index: int = None
) -> tuple[int, int, int, int]:
    """
    Measure the exact rendered bounding box of a text string via FFmpeg and OpenCV.

    Uses a render-and-analyze strategy: a phantom black frame is generated with
    the target text burned in by libass, then the resulting PNG is thresholded and
    inspected with OpenCV to obtain pixel-accurate bounds.  Mathematical estimation
    is avoided because libass kerning and font hinting produce results that differ
    meaningfully from naive calculations at display sizes.

    Two measurement modes are supported:
        - Isolated mode: measures a single text string on its own.
        - Contextual mode: measures one word within its full phrase by hiding all
          other words with \\alpha&HFF& tags, preserving libass line-break and
          kerning behaviour for the target word.

    Args:
        text: Text string to measure (uppercase). Used when full_phrase is None.
        font_size: ASS font size in pixels.
        video_width: Canvas width in pixels.
        video_height: Canvas height in pixels.
        v_position_percent: Vertical anchor position from the top (0–100).
        spacing: ASS character spacing value.
        full_phrase: Complete word list of the subtitle line (contextual mode).
        word_index: Index of the target word within full_phrase (contextual mode).

    Returns:
        Tuple (x, y, width, height) of the measured bounding box in pixels,
        or (0, 0, 0, 0) if no text pixels were detected.
    """
    margin_v = video_height - int(video_height * (v_position_percent / 100))

    # Contextual mode: render the full phrase but hide all words except the target
    # using \alpha tags. This preserves libass line layout for the measured word.
    if full_phrase is not None and word_index is not None:
        text_parts = []
        for i, word in enumerate(full_phrase):
            if i == word_index:
                # Target word: fully opaque.
                text_parts.append(f"{{\\alpha&H00&}}{word}{{\\r}}")
            else:
                # All other words: fully transparent (invisible but still laid out).
                text_parts.append(f"{{\\alpha&HFF&}}{word}{{\\r}}")

        dialogue_text = " ".join(text_parts)
    else:
        # Isolated mode: measure a standalone text string.
        if text is None:
            raise ValueError("Either 'text' or both 'full_phrase' and 'word_index' must be provided.")
        dialogue_text = text
    
    ass_content = f"""[Script Info]
Title: Phantom Render
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat ExtraBold,{font_size},&HFFFFFF,&HFFFFFF,&H000000,&H000000,-1,0,0,0,100,100,0,0,1,1,1,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{dialogue_text}
"""
    
    # Write the phantom ASS script and output image to temporary files.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as f:
        ass_path = Path(f.name)
        f.write(ass_content)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        phantom_path = Path(f.name)

    try:
        # Render a black frame with the text burned in; extract a single PNG.
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={video_width}x{video_height}:d=1",
            "-vf", f"ass={ass_path}",
            "-frames:v", "1",
            "-loglevel", "error",
            str(phantom_path)
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        img = cv2.imread(str(phantom_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Failed to load phantom frame: {phantom_path}")

        # Threshold at 200 to isolate white text pixels and discard anti-aliasing noise.
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresh)

        if coords is None or len(coords) == 0:
            print(f"[ERROR] No text pixels detected for '{text}'.")
            return (0, 0, 0, 0)

        x, y, w, h = cv2.boundingRect(coords)

        return (x, y, w, h)

    finally:
        # Remove temporary files regardless of success or failure.
        try:
            ass_path.unlink()
            phantom_path.unlink()
        except:
            pass


def generate_highlight_box_masks(
    words: list[dict],
    output_dir: Path,
    segments: list = None,
    font_size: int = 28,
    box_color: tuple = (255, 242, 204, 255),  # Default: soft cream yellow.
    video_width: int = 1080,
    video_height: int = 1920,
    border_radius: int = 12,
    padding_horizontal: int = 15,
    padding_vertical: int = 6,
    v_position_percent: int = 85
) -> dict[str, Path]:
    """
    Generate per-word PNG highlight mask frames using a render-and-analyze strategy.

    For each word, a phantom FFmpeg frame is rendered and analyzed with OpenCV to
    obtain the pixel-accurate bounding box as laid out by libass.  A rounded-rectangle
    highlight is then drawn on a transparent RGBA canvas matching the video dimensions,
    ready to be overlaid by FFmpeg.

    Padding is applied with a degressive ratio so that the box height scales
    sub-linearly with font size, preventing oversized boxes at large font sizes.

    Args:
        words: List of word dicts with "word", "start", and "end" keys.
        output_dir: Directory where PNG mask files will be written.
        segments: Subtitle segment list used to align words to their phrase context.
            When provided, each word is measured within its own segment line.
            When None, all words are treated as a single line (fallback mode).
        font_size: Base ASS font size in pixels.
        box_color: RGBA fill colour for the highlight rectangle (default: cream yellow).
        video_width: Canvas width in pixels (default 1080).
        video_height: Canvas height in pixels (default 1920).
        border_radius: Corner radius of the rounded rectangle in pixels.
        padding_horizontal: Unused — padding is calculated dynamically from bbox height.
        padding_vertical: Unused — padding is calculated dynamically from bbox height.
        v_position_percent: Vertical position of the subtitle line from the top (0–100).

    Returns:
        Dictionary mapping {word_index: Path} for each generated PNG mask file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = {}

    print(f"[BUSY] Generating highlight masks (render-and-analyze, contextual mode).")

    # --- Segment-aware mode: measure each word within its subtitle line context ---
    if segments:
        word_index = 0
        
        for seg_idx, segment in enumerate(segments):
            segment_words = [
                w for w in words
                if segment.start <= w["start"] < segment.end
            ]

            if not segment_words:
                continue

            segment_words_upper = [w["word"].upper() for w in segment_words]
            full_segment_text = " ".join(segment_words_upper).strip()

            # Measure each word in its phrase context to account for libass kerning.
            for i, word in enumerate(segment_words):
                w_fs = word.get("font_size", font_size)
                current_color = word.get("color", box_color)
                word_text = segment_words_upper[i]

                # Render the full line with only this word visible to get its exact position.
                x, y, w, h = _get_ffmpeg_text_bbox(
                    font_size=w_fs,
                    video_width=video_width,
                    video_height=video_height,
                    v_position_percent=v_position_percent,
                    spacing=0,
                    full_phrase=segment_words_upper,
                    word_index=i
                )
                
                if w == 0 or h == 0:
                    print(f"[INFO] Empty bbox for word '{word_text}', skipping.")
                    word_index += 1
                    continue

                # --- Degressive adaptive padding ---
                # Horizontal padding ratio decreases with bbox height so large-font
                # boxes do not appear disproportionately wide.
                ratio_x = max(0.35, 0.75 - (h * 0.004))
                pad_x = h * ratio_x

                # Vertical total padding follows the same degressive logic,
                # distributed 40 % above and 60 % below for optical balance.
                ratio_y_total = max(0.7, 1.0 - (h * 0.003))
                pad_y_total = h * ratio_y_total

                pad_y_top = pad_y_total * 0.4
                pad_y_bottom = pad_y_total * 0.6

                box_x = x - pad_x
                box_y = y - pad_y_top
                box_width = w + (2 * pad_x)
                box_height = h + pad_y_top + pad_y_bottom

                # Clamp box coordinates to video canvas boundaries.
                box_x = max(0, min(box_x, video_width - box_width))
                box_y = max(0, min(box_y, video_height - box_height))

                # Fixed corner radius gives a slightly rounded, rectangular appearance.
                adaptive_radius = 12
                
                # Créer masque
                img = Image.new('RGBA', (video_width, video_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle(
                    [(box_x, box_y), (box_x + box_width, box_y + box_height)],
                    radius=adaptive_radius,
                    fill=current_color
                )
                
                # Sauvegarder
                mask_path = output_dir / f"highlight_mask_{word_index:04d}.png"
                img.save(mask_path, "PNG")
                mask_paths[word_index] = mask_path
                
                word_index += 1
    
    # --- Fallback mode: treat all words as a single subtitle line ---
    else:
        print("[INFO] No segments provided; processing all words as a single line.")
        words_upper = [w["word"].upper() for w in words]

        for i, word in enumerate(words):
            word_text = words_upper[i]

            x, y, w, h = _get_ffmpeg_text_bbox(
                font_size=font_size,
                video_width=video_width,
                video_height=video_height,
                v_position_percent=v_position_percent,
                spacing=0,
                full_phrase=words_upper,
                word_index=i
            )

            if w == 0 or h == 0:
                print(f"[INFO] Empty bbox for word '{word_text}', skipping.")
                continue

            # Degressive padding — same formula as the segment-aware path above.
            ratio_x = max(0.35, 0.75 - (h * 0.004))
            pad_x = h * ratio_x

            ratio_y_total = max(0.7, 1.0 - (h * 0.003))
            pad_y_total = h * ratio_y_total

            pad_y_top = pad_y_total * 0.4
            pad_y_bottom = pad_y_total * 0.6

            box_x = x - pad_x
            box_y = y - pad_y_top
            box_width = w + (2 * pad_x)
            box_height = h + pad_y_top + pad_y_bottom

            # Clamp box coordinates to video canvas boundaries.
            box_x = max(0, min(box_x, video_width - box_width))
            box_y = max(0, min(box_y, video_height - box_height))

            adaptive_radius = 12

            img = Image.new('RGBA', (video_width, video_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle(
                [(box_x, box_y), (box_x + box_width, box_y + box_height)],
                radius=adaptive_radius,
                fill=current_color
            )

            mask_path = output_dir / f"highlight_mask_{i:04d}.png"
            img.save(mask_path, "PNG")
            mask_paths[i] = mask_path

    print(f"[SUCCESS] {len(mask_paths)} highlight masks generated.")
    return mask_paths


def create_viral_subtitle_segments(words: list[dict], max_words: int = 3) -> list[TranscriptSegment]:
    """
    Produce short, high-impact subtitle segments in the style popularised by viral
    short-form video editors (CapCut, Submagic, etc.).

    Segments are capped at max_words words and further split when speech is very
    fast (characters-per-second threshold) or the segment duration is short.
    After three consecutive fast segments, the algorithm forces a longer pause
    segment to prevent visual fatigue.

    All output text is uppercased and apostrophe-split tokens are re-joined before
    punctuation is stripped.

    Args:
        words: List of word dicts with "word", "start", and "end" keys.
        max_words: Maximum words per segment before a forced cut (default 3).

    Returns:
        List of TranscriptSegment instances with short, punchy text.
    """
    if not words:
        return []
    
    # Re-join apostrophe-split tokens before stripping punctuation.
    words = merge_apostrophe_words(words)

    # Strip punctuation while preserving apostrophes within contractions.
    for w in words:
        w["word"] = w["word"].translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    
    segments = []
    current_words = []
    current_segment_duration = 0.0
    fast_segments_count = 0

    for word in words:
        # Skip tokens that are empty after punctuation stripping.
        if not word["word"].strip():
            continue

        current_words.append(word)
        current_segment_duration = word["end"] - current_words[0]["start"]
        char_per_seconds = sum(len(w["word"]) for w in current_words) / current_segment_duration if current_segment_duration > 0 else 0

        # Emit a segment when any cut condition is satisfied.
        # After 3 consecutive fast segments, force a longer segment to avoid visual fatigue.
        if fast_segments_count >= 3:
            if len(current_words) >= 6 or current_segment_duration >= 1.2 or char_per_seconds >= 20:
                # Reset the consecutive-fast counter only when this segment is not itself fast.
                if char_per_seconds <= 20:
                    fast_segments_count = 0

                text = " ".join(w["word"] for w in current_words).upper()
                segments.append(TranscriptSegment(
                    start=current_words[0]["start"],
                    end=current_words[-1]["end"],
                    text=text
                ))
                current_words = []
        else:
            if len(current_words) >= max_words or current_segment_duration >= 0.4 or char_per_seconds >= 20 and current_segment_duration >= 0.3:
                if char_per_seconds >= 20:
                    fast_segments_count += 1
                else:
                    fast_segments_count = 0

                text = " ".join(w["word"] for w in current_words).upper()
                segments.append(TranscriptSegment(
                    start=current_words[0]["start"],
                    end=current_words[-1]["end"],
                    text=text
                ))
                current_words = []

    # Add remaining words 
    if current_words:
        text = " ".join(w["word"] for w in current_words).upper()
        segments.append(TranscriptSegment(
            start=current_words[0]["start"],
            end=current_words[-1]["end"],
            text=text
        ))

    return segments

def export_ass(
        segments: list[TranscriptSegment],
        output_path: Path,
        style_type: str = "pop",
        words: list[dict] = None,
        font_size: int = 28,
        v_position_percent: int = 85,
        video_height: int = 1920,
        color_name: str = "Yellow",
        primary_color: str = None
) -> Optional[dict]:
    """
    Export subtitle segments to an ASS file with configurable viral styling.

    Three style modes are supported:
        - "colors": Each word in a segment is highlighted in a cycling colour
          as it is spoken, while the rest of the line is white.
        - "box": Like colors, but adds per-word PNG box-highlight overlays for
          the first three of every seven segments.
        - Fallback ("glow", "pop", or no words): Plain white subtitles.

    Args:
        segments: List of subtitle segments to render.
        output_path: Destination .ass file path.
        style_type: Render style — "colors", "box", or plain fallback.
        words: Word-level timestamp list required for colors and box styles.
        font_size: Base ASS font size in pixels (default 28).
        v_position_percent: Vertical position of the subtitle line from the top (0–100).
        video_height: Canvas height for MarginV calculation (default 1920).
        color_name: Key into SUBTITLE_COLORS for the highlight colour.
        primary_color: Optional override for color_name (takes precedence when set).

    Returns:
        For "box" style: a mask_info dict with per-word timing and layout data.
        For all other styles: None.
    """
    from core.presets import SUBTITLE_COLORS
    # primary_color takes precedence over color_name to allow direct hex overrides.
    effective_color = primary_color if primary_color else color_name
    base_color_code = SUBTITLE_COLORS.get(effective_color, SUBTITLE_COLORS["Yellow"])

    # Colour rotation pairs (highlight, box). Violet is reserved for the box fill
    # and excluded from the rotation to maintain visual distinction.
    PAIRS = [
        ("Yellow", "&H8B008B&"),  # Yellow highlight / Violet box
        ("Green", "&H1E00A7&"),   # Green highlight / Red box
        ("Blue", "&H0772DE&")     # Blue highlight / Orange box
    ]

    offset_color = list(dict(PAIRS).keys()).index(effective_color) if effective_color in dict(PAIRS) else 0
    

    # Generate header with dynamic settings
    content = generate_ass_header(
        style_type=style_type,
        font_size=font_size,
        v_position_percent=v_position_percent,
        video_height=video_height
    )
    
    # Clean words: re-join apostrophe contractions and strip punctuation.
    if words:
        words = merge_apostrophe_words(words)
        for w in words:
            w["word"] = w["word"].translate(str.maketrans('', '', string.punctuation.replace("'", "")))

    # Initialise mask_info; populated only when style_type == "box".
    mask_info = {
        "words": [],
        "segments": segments,
        "font_size": font_size,
        "v_position_percent": v_position_percent,
        "video_height": video_height,
        "target_y": int(video_height * (v_position_percent / 100)),
        "box_color": None 
    }

    global_word_index = 0

    if style_type.lower() == "colors" and words:
        # Colors style: continuous base layer with the active word highlighted.
        for i, segment in enumerate(segments):
            segment_words = [
                w for w in words 
                if segment.start <= w["start"] < segment.end
            ]
            
            if not segment_words:
                continue
            
            # Cycle through colour pairs every 7 segments for variety.
            cycle_idx = (i // 7) % len(PAIRS) + offset_color
            current_pop_color, _ = PAIRS[cycle_idx % len(PAIRS)]
            pop_color_code = SUBTITLE_COLORS.get(current_pop_color, "&H00FFFF&")

            start_time = seconds_to_ass_time(segment.start)
            end_time = seconds_to_ass_time(segment.end)
            # Adaptive font size: fewer words = larger text for visual impact.
            loc_font_size = get_new_font_size(len(segment_words), font_size)
            
            # Layer 0: ALL words white, continuous
            all_words_white = " ".join(f"{{\\fs{loc_font_size}}}{{\\c&HFFFFFF&}}{w['word'].upper()}" for w in segment_words)
            line_base = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{all_words_white}\n"
            content += line_base
            
            # Layer 1: Each word in COLOR at its timing
            for i, word in enumerate(segment_words):
                word_start = seconds_to_ass_time(word["start"])
                word_end = seconds_to_ass_time(word["end"])
                
                highlight_parts = []
                for j, w in enumerate(segment_words):
                    if j == i:
                        # Active word: colored
                        highlight_parts.append(f"{{\\fs{loc_font_size}}}{{\\c{pop_color_code}}}{w['word'].upper()}{{\\r}}")
                    else:
                        # Other words: transparent
                        highlight_parts.append(f"{{\\fs{loc_font_size}}}{{\\alpha&HFF&}}{w['word'].upper()}{{\\alpha&H00&}}")
                
                highlight_text = " ".join(highlight_parts)
                line_highlight = f"Dialogue: 1,{word_start},{word_end},Default,,0,0,0,,{highlight_text}\n"
                content += line_highlight
    
    elif style_type.lower() == "box" and words:
        # Box style: active word highlighted and enlarged; first 3 of every 7 segments
        # also receive a PNG box overlay rendered by generate_highlight_box_masks.
        for i, segment in enumerate(segments):
            # Alternate period: first 3 of every 7 segments carry box highlights.
            is_box_segment = (i % 7) < 3

            current_pop_color, current_box_color_ass = PAIRS[0]  # Fixed to first pair; no colour rotation.
            pop_color_code = SUBTITLE_COLORS.get(current_pop_color, "&H00FFFF&")

            bgr_hex = current_box_color_ass.strip("&H&")
            r, g, b = int(bgr_hex[4:6], 16), int(bgr_hex[2:4], 16), int(bgr_hex[0:2], 16)
            current_rgb = (r, g, b, 255)


            segment_words = [ w for w in words if segment.start <= w["start"] < segment.end]
            nb_words = len(segment_words)

            if not segment_words:
                    # Empty segment: emit the raw segment text without word-level effects.
                    start_time = seconds_to_ass_time(segment.start)
                    end_time = seconds_to_ass_time(segment.end)
                    content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{segment.text}\n"
                    continue

            # Extend each word's end time to the next word's start to close timing gaps.
            for j, w in enumerate(segment_words):
                if j < nb_words - 1:
                    segment_words[j]["end"] = segment_words[j + 1]["start"]

            # ================== STYLE APPLICATION ==================

            loc_font_size = get_new_font_size(nb_words, font_size)
            highlight_size = int(loc_font_size * 1.1)  # 10 % zoom for the active word.
            global_word_index += nb_words

            for k, word in enumerate(segment_words):
                word_start = seconds_to_ass_time(word["start"])
                word_end = seconds_to_ass_time(word["end"])

                phrase_parts = []
                for j, w in enumerate(segment_words):
                    if j == k:
                        # Active word: Cycle Color + Bigger
                        if is_box_segment:
                            phrase_parts.append(f"{{\\c{pop_color_code}\\fs{loc_font_size}}}{w['word'].upper()}{{\\r}}")
                        else:
                            phrase_parts.append(f"{{\\c{pop_color_code}\\fs{highlight_size}}}{w['word'].upper()}{{\\r}}")
                    else:
                        phrase_parts.append(f"{{\\fs{loc_font_size}}}{w['word'].upper()}")

                full_phrase = " ".join(phrase_parts)
                content += f"Dialogue: 0,{word_start},{word_end},Default,,0,0,0,,{full_phrase}\n"
            
            if is_box_segment:
                # Add words to create highlight rectangles
                for word in segment_words:
                    mask_info["words"].append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "word_index": global_word_index,
                        "font_size": loc_font_size,
                        "color": current_rgb
                    })
    else:
        # Fallback: simple subtitles without effects (or if words is None)
        for segment in segments:
            start_time = seconds_to_ass_time(segment.start)
            end_time = seconds_to_ass_time(segment.end)
            line = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{segment.text}\n"
            content += line

    # Write the assembled ASS content to disk.
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[SUCCESS] ASS subtitles saved to {output_path} (style: {style_type}, color: {color_name})")
    
    if mask_info["words"]:
        return mask_info
    return None


def get_new_font_size(nb_words: int, fs: float) -> float:
    """
    Scale the base font size inversely with the word count of the segment.

    Shorter segments display fewer words and benefit from a larger font to fill
    the frame. Longer segments are scaled down proportionally to keep all words
    legible on a single line.

    Args:
        nb_words: Number of words in the subtitle segment.
        fs: Base font size in pixels.

    Returns:
        Adjusted font size as a float.
    """
    # Font scale factors are tuned empirically for a 1080x1920 canvas.
    if nb_words == 1:
        fs *= 1.25
    elif nb_words == 2:
        fs *= 1.15
    elif nb_words == 3:
        pass
    elif nb_words == 4:
        fs *= 0.9
    else:
        fs *= 0.8
    return fs