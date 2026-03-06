"""
Video downloader module.
Handles Youtube downloads and local file processing.
"""

import re
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import yt_dlp 

from core.exceptions import (
    DownloadError,
    VideoNotFoundError,
    NetworkError,
)

# Supported local video formats
SUPPORTED_FORMATS = [".mp4", ".mkv", ".mov", ".webm",".avi"]

@dataclass
class DownloadResult:
    """Result of a video download operation."""
    video_path: Path
    audio_path: Path 
    title: str
    duration: float  # seconds

@dataclass
class VideoInfo:
    """Video metadata without downloading."""
    title: str 
    duration: float  
    url: str 

def is_youtube_url(source: str) -> bool:
    """Check if source is a valid YT URL.
    
    Args:
        source: URL string to check.
    
    Returns:
        True if valid YT URL, else False.
    """
    # Matches both youtube.com full URLs and youtu.be short links; protocol and www prefix are optional
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+',
    ]

    for pattern in youtube_patterns:
        if re.match(pattern, source):
            return True
    return False

def is_local_file(source: str) -> bool:
    """Check if source is a valid local video file.
    
    Args:
        source: File path string to check.
        
    Returns:
        True if file exists and has supported format
    """
    path = Path(source)

    # Guard order matters: check existence first, then type, then format — cheapest checks first
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False
    
    return True 


def get_video_duration(video_path: Path) -> float:
    """
    Get video duration using ffprobe.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Duration in seconds.
    """
    try:
        # Query only the container-level duration field; noprint_wrappers and nokey reduce stdout to a bare float string
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
            check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise DownloadError(f"Failed to get video duration: {e}")


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """
    Extract audio from video using M4A format.
    Faster transcription than processing video directly.
    
    Args:
        video_path: Source video file 
        output_path: Destination audio file
        
    Returns:
        Path to extracted audio file.
    """
    # Skip re-extraction if a valid audio file already exists (avoids redundant ffmpeg calls)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[OK] Audio already cached, skipping extraction: {output_path}")
        return output_path
    
    try:
        # Stream-copy the audio codec (-acodec copy) to avoid re-encoding — much faster, zero quality loss
        # capture_output=True keeps ffmpeg's verbose logs off the terminal; stderr is still accessible on failure
        subprocess.run(
            [
                "ffmpeg",
                "-y",         # Overwrite output without prompting
                "-i", str(video_path),
                "-vn",        # Strip video stream, keep audio only
                "-acodec", "copy",
                str(output_path)
            ],
            capture_output=True,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise DownloadError(f"Audio extraction failed: {e.stderr.decode()}")


def download_youtube_video(url: str, output_dir: Path) -> DownloadResult:
    """
    Download video from YouTube.
    
    Args:
        url: YouTube video URL.
        output_dir: Directory to save downloaded files.
        
    Returns:
        DownloadResult with paths and metadata.
    """
    ydl_opts = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
        # Force final container to MP4 for broad compatibility
        'merge_output_format': 'mp4',
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'verbose': False,
        'restrictfilenames': True,
        # Use Android client to bypass server-side throttling and avoid bot-detection
        'extractor_args': {
            'youtube': {
                'player_client': ['android'],
            }
        },
        'external_downloader': 'aria2c',
        'external_downloader_args': [
            '-x16', # 16 parallel connections per server
            '-s16', # Split file into 16 segments
            '-k1M'  # Minimum segment size
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Fetch metadata first to resolve the final output filename before downloading
            info = ydl.extract_info(url, download=False)
            title = info['title']
            duration = info['duration']

            video_filename = ydl.prepare_filename(info)
            video_path = Path(video_filename)
            # Skip download if a valid file already exists (idempotent behavior)
            if video_path.exists() and video_path.stat().st_size > 0:
                print(f"[OK] Video already cached, skipping download: {video_path}")
            else:
                print(f"[BUSY] Downloading video: {video_path}")
                ydl.download([url])
        
            audio_path = output_dir / f"{video_path.stem}_audio.m4a"
            extract_audio(video_path, audio_path)

            return DownloadResult(
                video_path=video_path,
                audio_path=audio_path,
                title=title,
                duration=duration
            )
        
    except yt_dlp.utils.DownloadError as e:
        # yt-dlp collapses all failure modes into one exception type, so we inspect the
        # message string to map it to the appropriate domain exception for the caller
        error_msg = str(e).lower()

        if "video unavailable" in error_msg or "private video" in error_msg:
            raise VideoNotFoundError(f"Video not accessible: {e}")
        elif "sign in to confirm" in error_msg:
            raise VideoNotFoundError(f"Age restricted video (sign in required): {e}")
        elif "http error 429" in error_msg:
            raise NetworkError(f"Rate limit exceeded, try again later")
        else:
            raise DownloadError(f"Download failed: {e}")
        
    except Exception as e:
        raise DownloadError(f"Unexpected error during download: {e}")
    
def process_local_file(file_path: str, output_dir: Path) -> DownloadResult:
    """
    Process a local video file.
    
    Args:
        file_path: Path to local video
        output_dir: Temporary directory
        
    Returns:
        DownloadResult with paths and metadata.
    """
    source_path = Path(file_path)

    # Mirror the source into the working directory for non-destructive processing
    video_path = output_dir / source_path.name

    # copy2 preserves the original file's metadata (timestamps, permissions), unlike shutil.copy
    if not video_path.exists():
        shutil.copy2(source_path, video_path)
    else:
        print(f"[OK] Video already present in temp dir, skipping copy: {video_path}")

    duration = get_video_duration(video_path)
    title = source_path.stem

    audio_path = output_dir / f"{title}_audio.m4a"
    extract_audio(video_path, audio_path)

    return DownloadResult(
        video_path=video_path,
        audio_path=audio_path,
        title=title,
        duration=duration
    )

def download_video(source: str, output_dir: Path) -> DownloadResult:
    """
    Download or process a video (Youtube URL or local file).
    
    Args:
        source: YouTube URL or local file path.
        output_dir: Destination directory (usually temp).
        
    Returns:
        DownloadResult with paths and metadata.
    
    Raises:
        VideoNotFoundError: If video is not accessible.
        DownloadError: If download or processing fails.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_youtube_url(source):
        return download_youtube_video(source, output_dir)
    
    elif is_local_file(source):
        return process_local_file(source, output_dir)
    
    else:
        raise VideoNotFoundError(
            f"Invalid source: '{source}'."
            " Must be a valid YouTube URL or local video file."
            f"({','.join(SUPPORTED_FORMATS)})"
        )
    
def get_video_info(source: str) -> VideoInfo:
    """
    Get video metadata without downloading.
    Useful for previewing video details.
    
    Args:
        source: YT URL or local file path.
    
    Returns:
        VideoInfo with title and duration."""
    
    if is_youtube_url(source):
        ydl_opts = {'quiet': True, 'no_warnings': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                return VideoInfo(
                    title=info['title'],
                    duration=info['duration'],
                    url=source
                )
        except Exception as e:
            raise VideoNotFoundError(f"Failed to get video info: {e}")
        
    elif is_local_file(source):
        path = Path(source)
        duration = get_video_duration(path)
        return VideoInfo(
            title=path.stem,
            duration=duration,
            url=str(path)
        )
    else:
        raise VideoNotFoundError(f"Invalid source: '{source}'")
    
def cleanup_temp(temp_dir: Path) -> None:
    """
    Clean up temporary files after processing.
    
    Args:
        temp_dir: Path to temporary directory.
    """
    # Wipe the entire tree and recreate an empty directory rather than selectively deleting files —
    # simpler, safer, and handles nested subdirectories without extra logic
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        