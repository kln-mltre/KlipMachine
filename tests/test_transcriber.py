"""
Quick test script for the transcriber module.
Tests transcription + generation of all derivative files.
"""

from pathlib import Path
from core.transcriber import (
    transcribe,
    export_transcript,
    create_analysis_segments,
    create_subtitle_segments,
    export_analysis_segments,
    export_srt,
    export_subtitle_segments_txt,
    segments_to_text
)
from config import config

def test_transcriber():
    """Test audio transcription and derivative file generation."""
    
    print("KlipMachine - Transcriber Test\n")
    
    # Find audio file from previous download test
    temp_files = list(Path(config.TEMP_DIR).glob("*_audio.m4a"))
    
    if not temp_files:
        print("[ERROR] No audio file found in temp/")
        print("[INFO] Run test_download.py first to download a video")
        return
    
    audio_path = temp_files[0]
    print(f"[INFO] Found audio: {audio_path.name}\n")
    
    # ========================================================================
    # STEP 1: Transcription with word timestamps
    # ========================================================================
    print("Transcribing with 'base' model (word-level timestamps)...")
    print("This may take 1-2 minutes for the first run (downloading model)...\n")
    
    try:
        result, words = transcribe(
            audio_path=audio_path,
            model_size="base",
            language=None,  # Auto-detect
            device="auto"
        )
        
        print(f"\n[SUCCESS] Transcription complete!")
        print(f"   Language: {result.language}")
        print(f"   Duration: {result.duration:.1f}s")
        print(f"   Original segments: {len(result.segments)}")
        print(f"   Words extracted: {len(words)}\n")
        
        # Show first 3 segments
        print("First 3 original segments:")
        print("-" * 60)
        formatted = segments_to_text(result.segments[:3])
        print(formatted)
        print("-" * 60 + "\n")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 2: Export raw transcript (source of truth)
    # ========================================================================
    print("Exporting raw transcript...")
    transcript_path = str(Path(__file__).parent / "transcript.json")
    export_transcript(result, words, transcript_path)
    print()
    
    # ========================================================================
    # STEP 3: Generate AI analysis segments (long chunks)
    # ========================================================================
    print("Creating AI analysis segments (30-45s chunks)...")
    analysis_segments = create_analysis_segments(
        words,
        max_duration=45.0,
        silence_threshold=1.5
    )
    print(f"[SUCCESS] {len(analysis_segments)} analysis segments created")
    
    # Export to text file
    analysis_path = str(Path(__file__).parent / "analysis_segments.txt")
    export_analysis_segments(analysis_segments, analysis_path)
    
    # Preview first segment
    if analysis_segments:
        seg = analysis_segments[0]
        print(f"\n First AI segment preview:")
        print(f"   Duration: {seg.end - seg.start:.1f}s")
        print(f"   Text: {seg.text[:100]}...\n")
    
    # ========================================================================
    # STEP 4: Generate subtitle segments (short chunks)
    # ========================================================================
    print("Creating subtitle segments (4 words each)...")
    subtitle_segments = create_subtitle_segments(
        words,
        words_per_segment=4
    )
    print(f"[SUCCESS] {len(subtitle_segments)} subtitle segments created")
    
    # Export to SRT (for CapCut)
    srt_path = str(Path(__file__).parent / "subtitles.srt")
    export_srt(subtitle_segments, srt_path)
    
    # Export to TXT (for preview)
    subtitle_txt_path = str(Path(__file__).parent / "subtitle_segments.txt")
    export_subtitle_segments_txt(subtitle_segments, subtitle_txt_path)
    
    # Preview first 3 subtitle segments
    print(f"\nFirst 3 subtitle segments:")
    print("-" * 60)
    for seg in subtitle_segments[:3]:
        print(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
    print("-" * 60 + "\n")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\n[INFO] Generated files in {config.TEMP_DIR}:")
    print(f"   transcript.json          - Raw data (segments + words)")
    print(f"   analysis_segments.txt    - For AI analysis")
    print(f"   subtitles.srt           - For CapCut import")
    print(f"   subtitle_segments.txt    - Subtitle preview")
    print()

if __name__ == "__main__":
    test_transcriber()