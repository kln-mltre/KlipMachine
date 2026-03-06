"""
Export presets management.
Handles predefined and custom export configurations.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from config import config

# Subtitle color palette — values are in ASS BGR hex format (blue-green-red), not RGB.
# Commented entries are kept for reference and can be re-enabled as needed.
SUBTITLE_COLORS = {
    "Yellow": "&H00FFFF&",   # #FFFF00
    #"Purple": "&H8B008B&",   # #8B008B
    "Green": "&H00FF00&",    # #00FF00
    #"Red": "&H1E00A7&",      # #A7001E
    "Blue": "&HFFFF00&",     # #00FFFF
    #"Orange": "&H0772DE&",   # #DE7207
}

@dataclass
class ExportPreset:
    """
    Immutable configuration preset for a video export pipeline.

    Bundles spatial transform, subtitle style, and layout parameters into a
    single reusable unit that can be serialised to/from JSON.
    """
    name: str
    crop_mode: str  # "none", "center", "blur"
    blur_zoom: float  # Zoom factor for blur mode (1.0 - 2.0)
    subtitle_style: str  # "none", "glow", "pop"
    subtitle_color: str  # "Purple", "Yellow", "Red", etc.
    subtitle_font_size: int  # Font size in pixels (20-60)
    subtitle_position: int  # Vertical position percentage (0-100, 0=top, 100=bottom)
    
    def to_dict(self) -> dict:
        """
        Serialise the preset to a plain dictionary.

        Returns:
            dict representation of all preset fields, suitable for JSON output.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExportPreset":
        """
        Deserialise a preset from a plain dictionary.

        Args:
            data: Dictionary whose keys match ExportPreset field names.

        Returns:
            Populated ExportPreset instance.
        """
        return cls(**data)


# ============================================================================
# DEFAULT PRESETS
# ============================================================================

# font_size=80 is calibrated for the 1920 px canvas with absolute vertical positioning.
PRESET_VIRAL = ExportPreset(
    name="Viral",
    crop_mode="blur",
    blur_zoom=1.08,
    subtitle_style="pop",
    subtitle_color="Purple",
    subtitle_font_size=80,
    subtitle_position=95  # Near the bottom of the frame (95 % from top).
)

PRESET_CLEAN = ExportPreset(
    name="Clean",
    crop_mode="center",
    blur_zoom=1.0,  # No zoom applied in center-crop mode.
    subtitle_style="glow",
    subtitle_color="Yellow",
    subtitle_font_size=26,
    subtitle_position=80
)

PRESET_CINEMATIC = ExportPreset(
    name="Cinematic",
    crop_mode="blur",
    blur_zoom=1.15,
    subtitle_style="pop",
    subtitle_color="Purple",
    subtitle_font_size=32,
    subtitle_position=100
)

PRESET_COLORS = ExportPreset(
    name="Colors",
    crop_mode="none",
    blur_zoom=1.0,
    subtitle_style="none",
    subtitle_color="Purple",
    subtitle_font_size=24,
    subtitle_position=95
)

# Dictionary of default presets
DEFAULT_PRESETS = {
    "viral": PRESET_VIRAL,
    "clean": PRESET_CLEAN,
    "cinematic": PRESET_CINEMATIC,
    "colors": PRESET_COLORS
}

# ============================================================================
# PRESET MANAGEMENT
# ============================================================================

def get_preset(name: str) -> Optional[ExportPreset]:
    """
    Retrieve a built-in preset by name.

    Lookup is case-insensitive to accommodate user input from CLI or UI forms.

    Args:
        name: Human-readable preset name (e.g. "Viral", "Clean").

    Returns:
        Matching ExportPreset, or None if no preset with that name exists.
    """
    return DEFAULT_PRESETS.get(name.lower())

def list_presets() -> list[str]:
    """
    Return the names of all built-in presets.

    Returns:
        List of lowercase preset name strings.
    """
    return list(DEFAULT_PRESETS.keys())

def save_preset(preset: ExportPreset, filepath: Path) -> None:
    """
    Serialise a preset to a JSON file on disk.

    Args:
        preset: ExportPreset instance to persist.
        filepath: Destination path for the output JSON file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(preset.to_dict(), f, indent=2)

    print(f"[SUCCESS] Preset '{preset.name}' saved to {filepath}")

def load_preset(filepath: Path) -> ExportPreset:
    """
    Deserialise a preset from a JSON file on disk.

    Args:
        filepath: Path to the JSON preset file.

    Returns:
        Reconstructed ExportPreset instance.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the JSON content cannot be mapped to an ExportPreset.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Preset file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preset = ExportPreset.from_dict(data)
    print(f"[OK] Preset '{preset.name}' loaded from {filepath}")

    return preset

def get_presets_directory() -> Path:
    """
    Return the path to the user presets directory, creating it if absent.

    Returns:
        Path to the presets directory under the application base directory.
    """
    presets_dir = config.BASE_DIR / "presets"
    presets_dir.mkdir(exist_ok=True)
    return presets_dir

def save_user_preset(preset: ExportPreset) -> Path:
    """
    Persist a user-defined preset to the managed presets directory.

    The preset name is lowercased and spaces are replaced with underscores to
    produce a valid, portable filename.

    Args:
        preset: ExportPreset instance to save.

    Returns:
        Path to the written JSON file.
    """
    presets_dir = get_presets_directory()
    # Normalise preset name to a portable filename (lowercase, underscores).
    safe_name = preset.name.lower().replace(" ", "_")
    filepath = presets_dir / f"{safe_name}.json"

    save_preset(preset, filepath)
    return filepath

def load_user_presets() -> dict[str, ExportPreset]:
    """
    Load all user-saved presets found in the managed presets directory.

    Invalid or malformed JSON files are skipped with a warning so that a single
    corrupt preset does not prevent the others from loading.

    Returns:
        Dictionary mapping lowercase preset names to ExportPreset instances.
    """
    presets_dir = get_presets_directory()
    user_presets = {}

    for filepath in presets_dir.glob("*.json"):
        try:
            preset = load_preset(filepath)
            # Key by lowercase name to allow case-insensitive lookup.
            user_presets[preset.name.lower()] = preset
        except Exception as e:
            print(f"[ERROR] Failed to load preset from {filepath}: {e}")

    return user_presets

def get_all_presets() -> dict[str, ExportPreset]:
    """
    Return the merged set of built-in and user-saved presets.

    User presets take precedence over built-in ones when names collide, allowing
    project-level customisation without modifying the default definitions.

    Returns:
        Dictionary mapping lowercase preset names to ExportPreset instances.
    """
    all_presets = DEFAULT_PRESETS.copy()
    user_presets = load_user_presets()
    # User presets intentionally shadow built-in defaults on name collision.
    all_presets.update(user_presets)

    return all_presets

def delete_user_preset(name: str) -> bool:
    """
    Remove a user-saved preset file from the managed presets directory.

    Args:
        name: Human-readable preset name to delete.

    Returns:
        True if the file was found and removed, False if no such preset exists.
    """
    presets_dir = get_presets_directory()
    # Reconstruct the canonical filename using the same normalisation as save_user_preset.
    safe_name = name.lower().replace(" ", "_")
    filepath = presets_dir / f"{safe_name}.json"

    if filepath.exists():
        filepath.unlink()
        print(f"[SUCCESS] Preset '{name}' deleted from {filepath}")
        return True
    else:
        print(f"[INFO] Preset '{name}' not found at expected path: {filepath}")
        return False