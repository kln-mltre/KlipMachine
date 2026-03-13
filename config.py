"""Central configuration module for KlipMachine.
Handles paths, API keys, and default settings."""

import os
import ctypes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class Config:
    """Main configuration class for KlipMachine."""

    # Directories
    BASE_DIR: Path 
    TEMP_DIR: Path
    OUTPUT_DIR: Path
    LOG_DIR: Path
    PROMPTS_DIR: Path

    # FFmeg 
    FFMPEG_PATH: str

    # Whisper
    WHISPER_MODEL: str # "base" | "small" | "medium" | "large"
    WHISPER_DEVICE: str  # "cpu" | "cuda"

    #LLM 
    DEFAULT_PROVIDER: str # "groq" | "openai" | "ollama"
    GROQ_API_KEY: Optional[str]
    OPENAI_API_KEY: Optional[str]
    OLLAMA_HOST: str 

    # Video editing 
    MARGIN_BEFORE: float # seconds
    MARGIN_AFTER: float #seconds
    OUTPUT_WIDTH: int
    OUTPUT_HEIGHT: int

    # Chunking (for long videos)
    CHUNK_DURATION_MIN: int
    CHUNK_OVERLAP_MIN: int 

def detect_gpu() -> str:
    """
    Detect if CUDA GPU is available and usable by Whisper runtime.
    
    Returns:
        "cuda" if NVIDIA GPU is available, otherwise "cpu"
    """

    def cudnn_available() -> bool:
        """Return True only when required cuDNN libs are loadable."""
        candidates = [
            "libcudnn_ops.so.9.1.0",
            "libcudnn_ops.so.9.1",
            "libcudnn_ops.so.9",
            "libcudnn_ops.so",
        ]
        for lib in candidates:
            try:
                ctypes.CDLL(lib)
                return True
            except OSError:
                continue
        return False

    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        # If torch not installed, check nvidia-smi
        import subprocess
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True,
                timeout=2
            )
            cuda_ok = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            cuda_ok = False

    if not cuda_ok:
        return "cpu"

    # Guard against runtime crashes when CUDA exists but cuDNN is missing/incompatible.
    return "cuda" if cudnn_available() else "cpu"
            

def verify_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible.
    
    Returns:
        True if FFmpeg is available, False otherwise.
    """
    import shutil 
    return shutil.which("ffmpeg") is not None


def load_config() -> Config:
    """Load configuration from environment variables and defaults.
    
    Returns:
        Config object with all settings.
    """

    # Base directory (project root)
    base_dir = Path(__file__).parent.resolve()

    # Directories
    temp_dir = base_dir / "temp"
    output_dir = base_dir / "output"
    log_dir = base_dir / "logs"
    prompts_dir = base_dir / "prompts"

    # Create directories if they don't exist
    for directory in [temp_dir, output_dir, log_dir, prompts_dir]:
        directory.mkdir(exist_ok=True)

    # Detect GPU and allow manual override.
    detected_device = detect_gpu()
    whisper_device = os.getenv("KLIPMACHINE_WHISPER_DEVICE", detected_device).lower()
    if whisper_device not in {"cpu", "cuda"}:
        whisper_device = detected_device

    return Config(
        # Directories
        BASE_DIR=base_dir,
        TEMP_DIR=temp_dir,
        OUTPUT_DIR=output_dir,
        LOG_DIR=log_dir,
        PROMPTS_DIR=prompts_dir,

        # FFmpeg
        FFMPEG_PATH=os.getenv("KLIPMACHINE_FFMPEG_PATH", "ffmpeg"),

        # Whisper 
        WHISPER_MODEL=os.getenv("KLIPMACHINE_WHISPER_MODEL", "base"),
        WHISPER_DEVICE=whisper_device,

        # LLM
        DEFAULT_PROVIDER=os.getenv("KLIPMACHINE_PROVIDER", "groq"),
        GROQ_API_KEY=os.getenv("KLIPMACHINE_GROQ_KEY"),
        OPENAI_API_KEY=os.getenv("KLIPMACHINE_OPENAI_KEY"),
        OLLAMA_HOST=os.getenv("KLIPMACHINE_OLLAMA_HOST", "http://localhost:11434"),

        # Video editing
        MARGIN_BEFORE=float(os.getenv("KLIPMACHINE_MARGIN_BEFORE", "2.0")),
        MARGIN_AFTER=float(os.getenv("KLIPMACHINE_MARGIN_AFTER", "0.5")),
        OUTPUT_WIDTH=1080,
        OUTPUT_HEIGHT=1920,

        # Chucking 
        CHUNK_DURATION_MIN=10,
        CHUNK_OVERLAP_MIN=1,
    )

# Global config instance
config = load_config()