
from app.services.text_chunking_service import TextChunkingService


transcript = {
    "audio_filename": "sample.mp3",
    "language": "en",
    "duration": 30.0,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Artificial intelligence is transforming the world."
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "Machine learning helps computers learn from data."
        },
        {
            "start": 10.0,
            "end": 15.0,
            "text": "Deep learning uses neural networks to solve complex problems."
        },
        {
            "start": 15.0,
            "end": 20.0,
            "text": "Natural language processing helps computers understand human language."
        },
        {
            "start": 20.0,
            "end": 25.0,
            "text": "Large language models can generate meaningful responses."
        },
        {
            "start": 25.0,
            "end": 30.0,
            "text": "Retrieval augmented generation combines search with language models."
        }
    ]
}


chunking_service = TextChunkingService(
    chunk_size=120,
    chunk_overlap=30
)

chunks = chunking_service.chunk_transcript(transcript)

print("\nGenerated chunks:", len(chunks))

for chunk in chunks:
    print(
        f"Chunk {chunk['chunk_index']}: "
        f"{len(chunk['text'])} characters, "
        f"{chunk['start']}s - {chunk['end']}s"
    )
    print(chunk["text"])

# Chunk indexes should be sequential
assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))

# Every chunk must respect the configured maximum size
assert all(
    len(chunk["text"]) <= chunking_service.chunk_size
    for chunk in chunks
)

# Verify timestamps
assert chunks[0]["start"] == 0.0
assert chunks[0]["end"] == 15.0
assert chunks[1]["start"] == 5.0
assert chunks[1]["end"] == 20.0

# Verify word-boundary-aware overlap
assert chunks[1]["text"].startswith("from data.")
assert not chunks[1]["text"].startswith("lps")

print("All chunking assertions passed!")


print(f"Total chunks generated: {len(chunks)}")

for chunk in chunks:
    print("\n" + "=" * 60)
    print(f"Chunk index: {chunk['chunk_index']}")
    print(f"Start: {chunk['start']} seconds")
    print(f"End: {chunk['end']} seconds")
    print(f"Characters: {len(chunk['text'])}")
    print(f"Text: {chunk['text']}")


# Test overlap spanning multiple transcript segments

multi_segment_transcript = {
    "audio_filename": "multi_segment.mp3",
    "language": "en",
    "duration": 15.0,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Alpha topic contains several useful details here."
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "Beta topic explains another important concept here."
        },
        {
            "start": 10.0,
            "end": 15.0,
            "text": "Gamma topic introduces a completely different subject."
        }
    ]
}

multi_segment_service = TextChunkingService(
    chunk_size=110,
    chunk_overlap=80
)

multi_segment_chunks = multi_segment_service.chunk_transcript(
    multi_segment_transcript
)

print("\nMulti-segment overlap test:")

for chunk in multi_segment_chunks:
    print(
        f"Chunk {chunk['chunk_index']}: "
        f"{chunk['start']}s - {chunk['end']}s"
    )

# The overlap includes text from the first segment,
# so the second chunk should retain its start timestamp.
assert multi_segment_chunks[1]["start"] == 0.0

print("Multi-segment overlap assertion passed!")


# Test that chunks do not exceed the configured chunk size

oversized_transcript = {
    "audio_filename": "oversized.mp3",
    "language": "en",
    "duration": 15.0,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "word " * 16
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "word " * 16
        },
        {
            "start": 10.0,
            "end": 15.0,
            "text": "word " * 16
        }
    ]
}

oversized_service = TextChunkingService(
    chunk_size=100,
    chunk_overlap=30
)

oversized_chunks = oversized_service.chunk_transcript(
    oversized_transcript
)

print("\nOversized chunk test:")

for chunk in oversized_chunks:
    print(
        f"Chunk {chunk['chunk_index']}: "
        f"{len(chunk['text'])} characters"
    )

# Every chunk must respect the configured size
assert all(
    len(chunk["text"]) <= 100
    for chunk in oversized_chunks
)

print("Chunk size assertions passed!")


# Test word-boundary-aware text splitting

chunking_service = TextChunkingService(
    chunk_size=20,
    chunk_overlap=5
)

text = "Artificial intelligence is transforming the world"

text_parts = chunking_service._split_text(text)

print("\nText splitting test:")

for part in text_parts:
    print(f"{len(part)} characters: {part}")

assert all(len(part) <= 20 for part in text_parts)

assert " ".join(text_parts) == text

print("Text splitting assertions passed!")


import json
from pathlib import Path


# Load an actual transcript generated by Faster-Whisper

transcript_path = Path(
    "storage/transcripts/5780fcc3-57c8-4705-8cf1-4662061561a5.json"
)

with open(transcript_path, "r", encoding="utf-8") as file:
    actual_transcript = json.load(file)


# Initialize chunking service
actual_chunking_service = TextChunkingService(
    chunk_size=500,
    chunk_overlap=100
)


# Generate chunks
actual_chunks = actual_chunking_service.chunk_transcript(
    actual_transcript
)


# Display results
print("\nActual transcript chunking test:")
print(f"Original transcript segments: {len(actual_transcript['segments'])}")
print(f"Generated chunks: {len(actual_chunks)}")

for chunk in actual_chunks[:5]:
    print("\n" + "=" * 60)
    print(f"Chunk index: {chunk['chunk_index']}")
    print(f"Start: {chunk['start']} seconds")
    print(f"End: {chunk['end']} seconds")
    print(f"Characters: {len(chunk['text'])}")
    print(f"Text: {chunk['text']}")


# Validate generated chunks
assert len(actual_chunks) > 0

assert all(
    len(chunk["text"]) <= actual_chunking_service.chunk_size
    for chunk in actual_chunks
)

assert [
    chunk["chunk_index"] for chunk in actual_chunks
] == list(range(len(actual_chunks)))

assert all(
    chunk["start"] <= chunk["end"]
    for chunk in actual_chunks
)

print("\nActual transcript chunking assertions passed!")


# Test empty transcript

empty_transcript = {
    "audio_filename": "empty.mp3",
    "language": "en",
    "duration": 0.0,
    "segments": []
}

empty_chunks = actual_chunking_service.chunk_transcript(
    empty_transcript
)

assert empty_chunks == []

print("Empty transcript test passed!")


# Test transcript containing empty or whitespace-only segments

whitespace_transcript = {
    "audio_filename": "whitespace.mp3",
    "language": "en",
    "duration": 10.0,
    "segments": [
        {"start": 0.0, "end": 5.0, "text": ""},
        {"start": 5.0, "end": 10.0, "text": "   "}
    ]
}

whitespace_chunks = actual_chunking_service.chunk_transcript(
    whitespace_transcript
)

assert whitespace_chunks == []

print("Whitespace transcript test passed!")


# Test transcript containing a single segment

single_segment_transcript = {
    "audio_filename": "single.mp3",
    "language": "en",
    "duration": 5.0,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Artificial intelligence is transforming the world."
        }
    ]
}

single_segment_chunks = actual_chunking_service.chunk_transcript(
    single_segment_transcript
)

assert len(single_segment_chunks) == 1
assert single_segment_chunks[0]["chunk_index"] == 0
assert single_segment_chunks[0]["start"] == 0.0
assert single_segment_chunks[0]["end"] == 5.0

print("Single segment transcript test passed!")


# Test a word longer than the configured chunk size

long_word_transcript = {
    "audio_filename": "long_word.mp3",
    "language": "en",
    "duration": 5.0,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "abcdefghijklmnopqrstuvwxyz1234"
        }
    ]
}

long_word_service = TextChunkingService(
    chunk_size=20,
    chunk_overlap=5
)

long_word_chunks = long_word_service.chunk_transcript(
    long_word_transcript
)

print("\nLong word test:")

for chunk in long_word_chunks:
    print(f"{len(chunk['text'])} characters: {chunk['text']}")

assert len(long_word_chunks) == 2

assert all(
    len(chunk["text"]) <= 20
    for chunk in long_word_chunks
)

print("Long word test passed!")