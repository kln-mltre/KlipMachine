"""
AI analysis module using LLMs to identify clip-worthy moments.
Supports Groq, OpenAI, and Ollama.
"""

import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from functools import wraps

from groq import Groq
from openai import OpenAI
import requests

from core.exceptions import (
    AnalysisError,
    APIError,
    RateLimitError,
    JSONParseError,
    ConfigError
)
from config import config

@dataclass
class ClipSuggestion:
    """A suggested clip from AI analysis."""
    start: float          # Start timestamp (seconds)
    end: float            # End timestamp (seconds)
    title: str            # Suggested title for the clip
    hook: str             # Opening hook phrase
    score: float          # Relevance score (0-1)
    reason: str           # Explanation of why this moment is great


@dataclass
class AnalysisResult:
    """Result of AI analysis."""
    clips: list[ClipSuggestion]
    total_duration: float 
    provider_used: str 

# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry decorator with exponential backoff for API calls.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"[WARNING] Rate limit hit. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        raise
                except APIError as e:
                    if attempt < max_retries - 1 and "timeout" in str(e).lower():
                        delay = base_delay * (2 ** attempt)
                        print(f"[WARNING] API timeout. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator

# =============================================================================
# PROMPT LOADING
# =============================================================================


def load_prompt(template_name: str) -> str:
    """
    Load a prompt template from prompts/ directory.
    
    Args:
        template_name: Name of template ("multi-parts", "short-clips", "monetizable", "custom")
    
    Returns:
        Prompt string.
        
    Raises:
        AnalysisError if prompt file not found.
    """
    prompt_path = config.PROMPTS_DIR / f"{template_name}.txt"
    
    if not prompt_path.exists():
        raise AnalysisError(f"Prompt template '{template_name}' not found.")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
    

def build_prompt(
    transcript: str,
    angle: str,
    custom_instructions: Optional[str] = None
) -> str:
    """
    Build complete prompt for AI analysis.
    
    Args:
        transcript: Formatted transcript with timestamps
        angle: "multi-parts" | "short-clips" | "monetizable" | "custom"
        custom_instructions: Custom user instructions (if angle="custom")
        
    Returns:
        Complete prompt string.
    """
    # Always include system prompt (defines JSON format)
    system_prompt = load_prompt("system")

    # Add specific instructions
    if angle == "custom":
        if not custom_instructions:
            raise AnalysisError("Custom instructions must be provided for angle='custom'.")
        instructions = custom_instructions
    else:
        instructions = load_prompt(angle)

    full_prompt = f"{system_prompt}\n\n{instructions}\n\nTRANSCRIPT:\n{transcript}"
    
    return full_prompt


# =============================================================================
# API CALLS
# =============================================================================

_FAILED_MODELS_GROQ = set()
@retry_with_backoff(max_retries=3)
def call_groq(messages: list[dict], model: str = "llama-3.3-70b-versatile") -> str:
    """
    Call Groq API.
    
    Args:
        messages: List of message dicts with "role" and "content".
        model: Model name
        
    Returns:
        AI response content.
    """

    if not config.GROQ_API_KEY:
        raise ConfigError("Groq API key not configured.")
    
    
    potential_models = [model] #, "llama-3.1-8b-instant", "gemma2-9b-instant"]
    
    models_to_try = []
    seen = set()
    for m in potential_models:
        if m not in seen and m not in _FAILED_MODELS_GROQ:
            models_to_try.append(m)
            seen.add(m)
    if not models_to_try:
        raise RateLimitError("All Groq models have previously failed or been rate limited.")
        
    client = Groq(api_key=config.GROQ_API_KEY)
    last_exception = None

    for current_model in models_to_try:
        try:

            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            print(f"[OK] Groq response received using model {current_model}.")
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                print(f"[WARNING] Rate limit exceeded for model {current_model}, trying next...")
                _FAILED_MODELS_GROQ.add(current_model)
                continue  # Retry with next model
            else:
                raise APIError(f"Groq API error: {e}")
    
    raise RateLimitError("All Groq models rate limited or failed.")
        
@retry_with_backoff(max_retries=3)
def call_openai(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI API.
    
    Args:
        messages: List of message dicts
        model: Model name
        
    Returns:
        AI response text
    """
    if not config.OPENAI_API_KEY:
        raise ConfigError("OPENAI_API_KEY not configured")
    
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e).lower()
        if "rate_limit" in error_msg or "429" in error_msg:
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        else:
            raise APIError(f"OpenAI API error: {e}")
        
@retry_with_backoff(max_retries=3)
def call_ollama(messages: list[dict], model: str = "llama3") -> str:
    """
    Call Ollama local API.
    
    Args:
        messages: List of message dicts
        model: Model name
        
    Returns:
        AI response text
    """
    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    except requests.exceptions.ConnectionError:
        raise APIError(
            f"Cannot connect to Ollama at {config.OLLAMA_HOST}. "
            "Make sure Ollama is running and accessible."
        )
    except Exception as e:
        raise APIError(f"Ollama API error: {e}")
    

# =============================================================================
# JSON PARSING
# =============================================================================


def clean_json_response(text: str) -> str:
    """
    Clean LLM reponse to extract valid JSON.
    Handles common issues like markdown code blocks.
    
    Args:
        text: Raw LLM response 
        
    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks 
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find JSON object (between first { and last })
    start = text.find('{')
    end = text.rfind('}')

    if start == -1 or end == -1:
        raise JSONParseError("No JSON object found in LLM response.")
    
    return text[start:end+1]


def parse_timestamp(timestamp_str: str) -> float:
    """
    Convert timestamp string to seconds.
    Supports formats: "MM:SS", "HH:MM:SS", or just seconds as string.
    
    Args:
        timestamp_str: Timestamp as string
        
    Returns:
        Timestamp in seconds
    """
    timestamp_str = timestamp_str.strip()

    # Already in seconds
    try:
        return float(timestamp_str)
    except ValueError:
        pass

    # Parse MM:SS or HH:MM:SS
    parts = timestamp_str.split(':') 

    if len(parts) == 2: # MM:SS
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3: # HH:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    


def parse_llm_response(response: str) -> list[ClipSuggestion]:
    """
    Parse LLM JSON response into ClipSuggestion objects.
    
    Args: 
        response: Raw LLM response string
    
    Returns:
        List of ClipSuggestion objects
    
    Raises:
        JSONParseError : If response is not valid JSON 
    """ 
    cleaned = clean_json_response(response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise JSONParseError(f"Invalid JSON in response: {e}")
    
    if "clips" not in data:
        raise JSONParseError("No 'clips' field in LLM response JSON.")
    
    clips = []
    for clip_data in data["clips"]:
        try:
            clips.append(ClipSuggestion(
                start=parse_timestamp(str(clip_data["start"])),
                end=parse_timestamp(str(clip_data["end"])),
                title=clip_data["title"],
                hook=clip_data["hook"],
                score=float(clip_data.get("score", 0.8)),
                reason=clip_data.get("reason", "")
            ))
        except (KeyError, ValueError) as e:
            print(f"[WARNING] Skipping invalid clip: {e}")
            continue

    return clips


# =============================================================================
# CHUNKING LOGIC
# =============================================================================

def _split_transcript_into_chunks(transcript: str, chunk_minutes: int = 10, overlap_minutes: int = 1) -> list[str]:
    """
    Split the transcript into overlapping chunks.
    e.g. chunk 1 (0-10m), chunk 2 (9-19m), chunk 3 (18-28m)...
    The overlap prevents a clip from being cut off mid-sentence at a chunk boundary.
    """
    lines = transcript.splitlines()
    chunks = []
    
    # Regex to match the [MM:SS] timestamp at the start of each line
    timestamp_pattern = re.compile(r'^\[(\d{1,2}:)?(\d{2}):(\d{2})\]')
    
    # Parse the full transcript into a list of (seconds, line) tuples
    parsed_lines = []
    for line in lines:
        match = timestamp_pattern.match(line)
        if match:
            try:
                ts_str = match.group(0).strip('[]')
                ts_seconds = parse_timestamp(ts_str)
                parsed_lines.append((ts_seconds, line))
            except:
                continue # Skip lines with no valid timestamp
        else:
            # No timestamp found — attach this line to the previous entry
            if parsed_lines:
                last_ts, _ = parsed_lines[-1]
                parsed_lines.append((last_ts, line))

    if not parsed_lines:
        return [transcript] # Fallback if timestamp parsing fails

    total_duration = parsed_lines[-1][0]
    chunk_size = chunk_minutes * 60
    overlap_size = overlap_minutes * 60
    
    # Generate time-based chunks
    cursor = 0.0
    while cursor < total_duration:
        chunk_start = cursor
        chunk_end = cursor + chunk_size
        
        # Collect all lines that fall within the current time window
        current_chunk_lines = [
            line for ts, line in parsed_lines 
            if chunk_start <= ts < chunk_end
        ]
        
        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
        
        # Move the cursor forward, stepping back by the overlap amount to avoid cutting mid-sentence
        cursor += (chunk_size - overlap_size)
        
        # Safety check to prevent an infinite loop
        if chunk_size <= overlap_size: break 

    return chunks

def _deduplicate_clips(clips: list[ClipSuggestion]) -> list[ClipSuggestion]:
    """
    Remove duplicate clips produced by the chunking overlap.
    No two clips in the final output should share any overlapping time range.
    """
    if not clips: return []
    
    sorted_by_score = sorted(clips, key=lambda x: getattr(x, 'score', 0), reverse=True)
    accepted_clips = []

    for candidate in sorted_by_score:
        is_overlapping = False

        for kept in accepted_clips:
            # Discard the candidate if it overlaps with an already accepted clip
            if candidate.start < kept.end and kept.start < candidate.end:
                is_overlapping = True
                print(f"  [INFO] Dropping overlapping clip ({candidate.start}-{candidate.end}), keeping ({kept.start}-{kept.end})")
                break
        
        if not is_overlapping:
            accepted_clips.append(candidate)

    # Re-sort by start time
    accepted_clips.sort(key=lambda x: x.start)
    return accepted_clips


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


def analyze_transcript(
    transcript: str,
    angle: str = "short-clips",
    custom_instructions: Optional[str] = None,
    provider: str = "groq"
) -> AnalysisResult:
    """
    Analyse transcript to identify clip-worthy moments using AI.

    Args:
        transcript: Formatted transcript with timestamps
        angle: "multi-parts" | "short-clips" | "monetizable" | "custom"
        custom_instructions: Custom user instructions (if angle="custom")
        provider: "groq" | "openai" | "ollama"
    
    Returns:
        AnalysisResult with suggested clips.
    
    Raises:
        AnalysisError: If analysis fails
        ConfigError: If API keys missing
    """
    if len(transcript) < 15000:
        print("[INFO] Analyzing transcript in a single chunk...")
        prompt = build_prompt(transcript, angle, custom_instructions)
        messages = [{"role": "user", "content": prompt}]

        if provider == "groq": response = call_groq(messages)
        elif provider == "openai": response = call_openai(messages)
        elif provider == "ollama": response = call_ollama(messages)
        else: raise AnalysisError(f"Unsupported provider: {provider}")

        clips = parse_llm_response(response)

    else:
        print("[INFO] Transcript too long, splitting into chunks...")
        chunks = _split_transcript_into_chunks(transcript, chunk_minutes=5, overlap_minutes=1)
        clips = []

        for i, chunk in enumerate(chunks):
            print(f"[BUSY] Analyzing chunk {i+1}/{len(chunks)}...")

            if i > 0:
                time.sleep(2)  # brief pause between calls
            
            prompt = build_prompt(chunk, angle, custom_instructions)
            messages = [{"role": "user", "content": prompt}]

            try:
                if provider == "groq": response = call_groq(messages)
                elif provider == "openai": response = call_openai(messages)
                elif provider == "ollama": response = call_ollama(messages)
                else: raise AnalysisError(f"Unsupported provider: {provider}")

                chunk_clips = parse_llm_response(response)
                clips.extend(chunk_clips)
                print(f"  [INFO] Found {len(chunk_clips)} clips in this chunk.")
            except Exception as e:
                print(f"[ERROR] Error analyzing chunk {i+1}: {e}")
                continue
    
    # Summarize total duration
    clips = _deduplicate_clips(clips)
    total_duration = sum(clip.end - clip.start for clip in clips)

    print(f"[SUCCESS] Analysis complete. Found {len(clips)} clips totaling {total_duration:.1f} seconds.")
    

    return AnalysisResult(
        clips=clips,
        total_duration=total_duration,
        provider_used=provider
    )


def export_analysis(result: AnalysisResult, output_path: Path) -> None:
    """
    Export analysis result to JSON file.
    
    Args:
        result: AnalysisResult to export
        output_path: Destination JSON file
    """
    from dataclasses import asdict

    data = {
        "provider": result.provider_used,
        "total_duration": result.total_duration,
        "clips": [asdict(clip) for clip in result.clips]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Analysis exported to {output_path}")

def load_analysis(json_path: Path) -> AnalysisResult:
    """
    Load analysis restult from JSON file.
    Useful for testing editor without re-running AI analysis.
    
    Args:
        json_path: Path to analysis JSON file
        
    Returns:
        AnalysisResult object
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clips = [
        ClipSuggestion(**clip) 
        for clip in data["clips"]
    ]

    return AnalysisResult(
        clips=clips,
        total_duration=data["total_duration"],
        provider_used=data["provider"]
    )

