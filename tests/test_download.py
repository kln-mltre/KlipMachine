"""Test script for the downloader module."""

from pathlib import Path
from core.downloader import download_video, get_video_info
from config import config 

def test_download():
    """Test video download with a short Youtube video."""

    test_url = str(Path(__file__).parent / "test_video.mp4")

    print("KlipMachine - Downloader Test\n")
    print(f"[INFO] Temp directory: {config.TEMP_DIR}\n")

    # First, get video info without downloading
    print("Fetching video info...")
    try:
        info = get_video_info(test_url)
        print(f"[SUCCESS] Title: {info.title}")
        print(f"[SUCCESS] Duration: {info.duration:.1f} seconds\n")
    except Exception as e:
        print(f"[ERROR] Failed to get info: {e}\n")
        return
    

    # Now download
    print("Downloading video...")
    try:
        result = download_video(test_url, config.TEMP_DIR)
        print(f"[SUCCESS] Video downloaded: {result.video_path.name}")
        print(f"[SUCCESS] Audio extracted: {result.audio_path.name}")
        print(f"[SUCCESS] Title: {result.title}")
        print(f"[SUCCESS] Duration: {result.duration:.1f} seconds")
        
        # Check files exist
        print(f"\n[INFO] Files created:")
        print(f"   Video: {result.video_path.exists()} ({result.video_path.stat().st_size / 1024:.1f} KB)")
        print(f"   Audio: {result.audio_path.exists()} ({result.audio_path.stat().st_size / 1024:.1f} KB)")
        
        print("\n[SUCCESS] Download test PASSED!")
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_download()