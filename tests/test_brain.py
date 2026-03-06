"""
Test AI analysis with brain.py
"""

from pathlib import Path
from core.transcriber import load_transcript, create_analysis_segments, segments_to_text
from core.brain import analyze_transcript
from config import config

def test_brain():
    """Test AI clip analysis."""
    
    print("[INFO] KlipMachine - Brain Test\n")
    
    # Check API key
    if not config.GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY not configured")
        print("Create a .env file with:")
        print("     KLIPMACHINE_GROQ_KEY=your_key_here")
        print("\nGet a free key at: https://console.groq.com")
        return
    
    # Load transcript
    transcript_path = str(Path(__file__).parent / "transcript.json")
    
    if not Path(transcript_path).exists():
        print("[ERROR] transcript.json not found")
        print("[INFO] Run test_transcriber.py first")
        return
    
    print(f"[INFO] Loading transcript from {Path(transcript_path).name}...")
    result, words = load_transcript(transcript_path)
    print(f"[SUCCESS] Loaded: {len(words)} words, {result.duration:.1f}s\n")
    
    # Create analysis segments (long chunks for AI)
    print("[INFO] Creating analysis segments...")
    analysis_segments = create_analysis_segments(
        words,
        max_duration=45.0,
        silence_threshold=1.5
    )
    print(f"[SUCCESS] {len(analysis_segments)} segments created\n")
    
    # Format transcript for AI
    transcript_text = segments_to_text(analysis_segments)
    
    print(f"[INFO] Transcript preview (first 200 chars):")
    print("-" * 60)
    print(transcript_text[:200] + "...")
    print("-" * 60 + "\n")
    
    # Analyze with AI
    print("[INFO] Analyzing with AI (Groq - Short Clips angle)...")
    print("[INFO] This may take 10-30 seconds...\n")
    
    try:
        analysis = analyze_transcript(
            transcript=transcript_text,
            angle="short-clips",
            provider="groq"
        )
        
        print(f"\n[SUCCESS] Analysis complete!")
        print(f"   Provider: {analysis.provider_used}")
        print(f"   Clips found: {len(analysis.clips)}")
        print(f"   Total duration: {analysis.total_duration:.1f}s\n")
        
        # Show suggested clips
        print("[INFO] Suggested clips:")
        print("=" * 80)
        
        for i, clip in enumerate(analysis.clips, start=1):
            duration = clip.end - clip.start
            print(f"\n{i}. {clip.title}")
            print(f"   Time: {clip.start:.1f}s - {clip.end:.1f}s ({duration:.1f}s)")
            print(f"   Score: {clip.score:.2f}")
            print(f"   Hook: {clip.hook}")
            print(f"   Reason: {clip.reason}")

        # Export analysis result
        from core.brain import export_analysis
        analysis_result_path = str(Path(__file__).parent / "analysis_result.json")
        export_analysis(analysis, analysis_result_path)
        print(f"\n[INFO] Analysis saved to {analysis_result_path}")
        
        print("\n" + "=" * 80)
        print("\n[SUCCESS] Brain test PASSED!")
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_brain()