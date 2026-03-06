"""
Test video editor - extract clips suggested by AI.
"""

from pathlib import Path
from core.brain import load_analysis
from core.editor import ClipConfig, batch_export
from core.transcriber import load_transcript, create_subtitle_segments, export_srt
from config import config

def test_editor():
    """Test clip extraction from AI suggestions."""
    
    print("KlipMachine - Editor Test\n")
    
    # ========================================================================
    # STEP 1: Load AI analysis results
    # ========================================================================
    analysis_path = str(Path(__file__).parent / "analysis_result.json")
    
    if not Path(analysis_path).exists():
        print("[ERROR] analysis_result.json not found")
        print("[INFO] Run test_brain.py first")
        return
    
    print(f"[INFO] Loading AI analysis from {Path(analysis_path).name}...")
    analysis = load_analysis(analysis_path)
    print(f"[SUCCESS] Loaded {len(analysis.clips)} clip suggestions\n")
    
    # ========================================================================
    # STEP 2: Find source video
    # ========================================================================
    video_files = list(Path(config.TEMP_DIR).glob("*.mp4"))
    
    if not video_files:
        print("[ERROR] No video file found in temp/")
        print("[INFO] Run test_download.py first")
        return
    
    video_path = video_files[0]
    print(f"[INFO] Source video: {video_path.name}\n")
    
    # ========================================================================
    # STEP 3: Convert AI suggestions to ClipConfig
    # ========================================================================
    print(" Preparing clips with margins...")
    clip_configs = []
    
    for i, suggestion in enumerate(analysis.clips, start=1):
        clip_config = ClipConfig(
            start=suggestion.start,
            end=suggestion.end,
            title=suggestion.title,
            margin_before=config.MARGIN_BEFORE,  # 2.0s
            margin_after=config.MARGIN_AFTER     # 0.5s
        )
        clip_configs.append(clip_config)
        
        actual_duration = suggestion.end - suggestion.start
        with_margins = actual_duration + config.MARGIN_BEFORE + config.MARGIN_AFTER
        print(f"   {i}. {suggestion.title[:40]:40s} "
              f"{actual_duration:5.1f}s → {with_margins:5.1f}s (with margins)")
    
    print()
    
    # ========================================================================
    # STEP 4: Create output directory
    # ========================================================================
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_DIR / f"test_clips_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Output directory: {output_dir}\n")
    
    # ========================================================================
    # STEP 5: Extract clips with FFmpeg
    # ========================================================================
    print("Extracting clips...")
    print("=" * 70)
    
    def progress_callback(current, total):
        percent = (current / total) * 100
        print(f"\nProgress: {current}/{total} ({percent:.0f}%)")
    
    results = batch_export(
        video_path=video_path,
        clips=clip_configs,
        output_dir=output_dir,
        progress_callback=progress_callback
    )
    
    print("=" * 70)
    
    # ========================================================================
    # STEP 6: Generate SRT subtitles for each clip
    # ========================================================================
    print("\nGenerating subtitles...")
    
    # Load transcript
    transcript_path = Path(__file__).parent / "transcript.json"
    if transcript_path.exists():
        result, words = load_transcript(transcript_path)
        
        for i, (clip_config, export_result) in enumerate(zip(clip_configs, results), start=1):
            if not export_result.success:
                continue
            
            # Filter words for this clip (with margins)
            start_with_margin = max(0, clip_config.start - clip_config.margin_before)
            end_with_margin = clip_config.end + clip_config.margin_after
            
            clip_words = [
                w for w in words 
                if start_with_margin <= w["start"] <= end_with_margin
            ]
            
            if clip_words:
                # Adjust timestamps to start from 0
                adjusted_words = []
                for w in clip_words:
                    adjusted_words.append({
                        "word": w["word"],
                        "start": w["start"] - start_with_margin,
                        "end": w["end"] - start_with_margin
                    })
                
                # Create subtitle segments
                subtitle_segments = create_subtitle_segments(
                    adjusted_words,
                    words_per_segment=4
                )
                
                # Export SRT
                srt_path = export_result.output_path.with_suffix('.srt')
                export_srt(subtitle_segments, srt_path)
                print(f"   {srt_path.name}")
    else:
        print("   [WARNING]  transcript.json not found, skipping subtitles")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("[INFO] EXPORT SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\n[SUCCESS] Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"[ERROR] Failed: {failed}")
    
    print(f"\n[INFO] Output directory:")
    print(f"   {output_dir}")
    
    print(f"\n[INFO] Generated files:")
    for file in sorted(output_dir.iterdir()):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name:50s} {size_mb:6.2f} MB")
    
    if successful > 0:
        print("\n[SUCCESS] Editor test PASSED!")
        print(f"[INFO] Check your clips in: {output_dir}")
    else:
        print("\n[ERROR] All clips failed to export")

if __name__ == "__main__":
    test_editor()