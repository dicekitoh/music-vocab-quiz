#!/usr/bin/env python3
"""
lyrics_vocab.py - Extract vocabulary from song lyrics and enrich with Gemini.

Usage:
  python lyrics_vocab.py "Jamiroquai" "Virtual Insanity"
  python lyrics_vocab.py "Jamiroquai" "Virtual Insanity" --lyrics-file lyrics.txt
"""

import sys
import os
import json
import re
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Windows UTF-8 safe
sys.stdout.reconfigure(encoding="utf-8")

# Import Gemini connector
sys.path.insert(0, str(Path(__file__).parent.parent / "connectors"))
from gemini_smart import smart_chat

# ── Constants ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
VOCAB_FILE = SCRIPT_DIR / "vocab_data.json"

STOP_WORDS = {
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "mine", "yours", "this", "that", "these", "those", "in", "on", "at",
    "to", "for", "of", "with", "and", "but", "or", "not", "no", "so", "if",
    "what", "who", "how", "when", "where", "which", "there", "here", "just",
    "up", "out", "all", "got", "get",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Lowercase, replace spaces with hyphens, strip non-alphanumeric (except hyphens)."""
    text = text.lower().strip()
    text = text.replace(" ", "-")
    text = re.sub(r"[^a-z0-9\-]", "", text)
    return text


def make_song_id(artist: str, title: str) -> str:
    return f"{slugify(artist)}__{slugify(title)}"


def fetch_lyrics(artist: str, title: str) -> str:
    """Fetch lyrics from lyrics.ovh API."""
    url = f"https://api.lyrics.ovh/v1/{urllib.request.quote(artist)}/{urllib.request.quote(title)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lyrics-vocab/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("lyrics", "")
    except urllib.error.HTTPError as e:
        print(f"Error: lyrics.ovh returned HTTP {e.code} for '{artist} - {title}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching lyrics: {e}")
        sys.exit(1)


def extract_words(lyrics: str) -> tuple[list[str], dict[str, str]]:
    """
    Extract unique words from lyrics.
    Returns (sorted word list, {word: first example line}).
    """
    # Build example lines mapping before cleaning annotations
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]

    # Remove annotations: [Chorus], [Verse 2], (x2), etc.
    cleaned = re.sub(r"\[.*?\]", "", lyrics)
    cleaned = re.sub(r"\(x?\d+\)", "", cleaned)

    # Split on whitespace and punctuation (keep apostrophes inside words)
    tokens = re.findall(r"[a-zA-Z']+", cleaned)

    seen = set()
    words = []
    example_lines: dict[str, str] = {}

    for token in tokens:
        w = token.lower().strip("'")
        # Strip possessive 's
        if w.endswith("'s"):
            w = w[:-2]
        # Remove single chars except "i"
        if len(w) <= 1 and w != "i":
            continue
        # Skip stop words
        if w in STOP_WORDS:
            continue
        if w in seen:
            continue
        seen.add(w)
        words.append(w)

        # Find first line containing this word
        pattern = re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
        for line in lines:
            if pattern.search(line):
                example_lines[w] = line
                break

    words.sort()
    return words, example_lines


def enrich_with_gemini(words: list[str], max_retries: int = 1) -> list[dict]:
    """Send words to Gemini for Japanese meaning, POS, and genre enrichment."""
    word_list = ", ".join(words)
    prompt = f"""You are an English-Japanese vocabulary assistant.
For each English word, provide:
- meaning_ja: concise Japanese meaning
- pos: one of noun/verb/adjective/adverb/preposition/conjunction/pronoun/interjection/phrase
- genre: one of daily_life/emotion/nature/abstract/body/time/social/academic/slang

Return ONLY a JSON array, no markdown fences.
Words: {word_list}

Example: [{{"word":"escape","meaning_ja":"逃げる","pos":"verb","genre":"daily_life"}}]"""

    for attempt in range(max_retries + 1):
        response = smart_chat(prompt)
        # Strip markdown fences if present
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            if attempt < max_retries:
                print(f"  JSON parse failed, retrying... (attempt {attempt + 1})")
                continue
            else:
                print(f"  Error: Could not parse Gemini response after {max_retries + 1} attempts.")
                print(f"  Raw response: {text[:200]}...")
                return []

    return []


def load_vocab() -> dict:
    """Load existing vocab_data.json or return empty structure."""
    if VOCAB_FILE.exists():
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "songs": [], "words": []}


def save_vocab(data: dict) -> None:
    """Save vocab_data.json."""
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def next_word_id(existing_words: list[dict]) -> int:
    """Find the max numeric ID from existing words and return next number."""
    max_id = 0
    for w in existing_words:
        match = re.match(r"w_(\d+)", w.get("id", ""))
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def merge_results(
    data: dict,
    song_id: str,
    artist: str,
    title: str,
    words: list[str],
    enriched: list[dict],
    example_lines: dict[str, str],
) -> tuple[int, int]:
    """
    Merge new words into vocab data.
    Returns (new_count, updated_count).
    """
    # Add song if not already present
    existing_song_ids = {s["id"] for s in data["songs"]}
    if song_id not in existing_song_ids:
        jst = timezone(timedelta(hours=9))
        data["songs"].append({
            "id": song_id,
            "artist": artist,
            "title": title,
            "added_at": datetime.now(jst).isoformat(),
        })

    # Build lookup of existing words
    word_lookup: dict[str, dict] = {}
    for w in data["words"]:
        word_lookup[w["word"]] = w

    # Build lookup from enriched data
    enriched_lookup: dict[str, dict] = {}
    for item in enriched:
        if "word" in item:
            enriched_lookup[item["word"].lower()] = item

    new_count = 0
    updated_count = 0
    wid = next_word_id(data["words"])

    for word in words:
        if word in word_lookup:
            # Word exists - append song_id if not already there
            existing = word_lookup[word]
            if song_id not in existing.get("songs", []):
                existing.setdefault("songs", []).append(song_id)
                updated_count += 1
        else:
            # New word
            info = enriched_lookup.get(word, {})
            entry = {
                "id": f"w_{wid:03d}",
                "word": word,
                "meaning_ja": info.get("meaning_ja", ""),
                "pos": info.get("pos", ""),
                "genre": info.get("genre", ""),
                "songs": [song_id],
                "example_line": example_lines.get(word, ""),
            }
            data["words"].append(entry)
            word_lookup[word] = entry
            new_count += 1
            wid += 1

    return new_count, updated_count


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract and enrich vocabulary from song lyrics.")
    parser.add_argument("artist", help="Artist name")
    parser.add_argument("title", help="Song title")
    parser.add_argument("--lyrics-file", help="Path to a text file with lyrics (skip API fetch)")
    args = parser.parse_args()

    artist = args.artist
    title = args.title
    song_id = make_song_id(artist, title)

    print(f"Song: {artist} - {title}")
    print(f"ID:   {song_id}")

    # Get lyrics
    if args.lyrics_file:
        lyrics_path = Path(args.lyrics_file)
        if not lyrics_path.exists():
            print(f"Error: File not found: {args.lyrics_file}")
            sys.exit(1)
        lyrics = lyrics_path.read_text(encoding="utf-8")
        print(f"Loaded lyrics from {args.lyrics_file}")
    else:
        print("Fetching lyrics from lyrics.ovh...")
        lyrics = fetch_lyrics(artist, title)
        if not lyrics.strip():
            print("Error: No lyrics returned.")
            sys.exit(1)
        print(f"Fetched {len(lyrics)} characters of lyrics.")

    # Extract words
    words, example_lines = extract_words(lyrics)
    print(f"Extracted {len(words)} unique words (after filtering).")

    if not words:
        print("No words to process.")
        return

    # Enrich with Gemini
    print(f"Sending {len(words)} words to Gemini for enrichment...")
    enriched = enrich_with_gemini(words)
    print(f"Received enrichment for {len(enriched)} words.")

    # Load, merge, save
    data = load_vocab()
    new_count, updated_count = merge_results(data, song_id, artist, title, words, enriched, example_lines)
    save_vocab(data)

    print(f"\nAdded {new_count} new words, updated {updated_count} existing words from {artist} - {title}")
    print(f"Saved to {VOCAB_FILE}")


if __name__ == "__main__":
    main()
