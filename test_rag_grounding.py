
import re

from app.services.rag_service import RAGService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"
QUESTION = "How does admitting mistakes demonstrate humility?"

EXPECTED_ANSWER_TERMS = [
    "self-reflection",
    "humility",
    "mistakes",
    "improve"
]

UNSUPPORTED_TERMS = [
    "go to hell",
    "religious judgment",
    "judged by religion",
    "foundational concept of free will",
    "linking the act of admission directly to",
    "rather than defensiveness"
]


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

    return f"{minutes:02d}:{remaining_seconds:02d}"


def parse_timestamp(timestamp: str) -> int:
    parts = [int(part) for part in timestamp.split(":")]

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Invalid timestamp: {timestamp}")


def test_rag_grounding():
    print("Initializing RAG service...")
    rag_service = RAGService()

    print("\nGenerating answer...")
    result = rag_service.ask(
        document_id=DOCUMENT_ID,
        query=QUESTION,
        top_k=5
    )

    answer = result["answer"]
    answer_lower = answer.lower()

    print("\nGenerated Answer:")
    print(answer)

    # Check expected information
    missing_terms = [
        term for term in EXPECTED_ANSWER_TERMS
        if term not in answer_lower
    ]

    assert not missing_terms, (
        f"Expected information missing from answer: {missing_terms}"
    )

    print("\nExpected information validation passed!")

    # Check known unsupported claims
    unsupported_matches = [
        term for term in UNSUPPORTED_TERMS
        if term in answer_lower
    ]

    assert not unsupported_matches, (
        f"Potentially unsupported claims found: {unsupported_matches}"
    )

    print("Unsupported claim validation passed!")

    # Check retrieved source evidence
    source_chunks = {
        source["chunk_index"]: source["text"]
        for source in result["sources"]
    }

    assert 0 in source_chunks, "Relevant chunk 0 was not retrieved."
    assert 1 in source_chunks, "Relevant chunk 1 was not retrieved."

    assert "self-reflection and humility" in source_chunks[0].lower()
    assert "learn from my mistakes and improve" in source_chunks[1].lower()

    print("Source evidence validation passed!")

    # Check timestamp citation format
    citation_pattern = (
        r"\[(\d{2}:\d{2}(?::\d{2})?)\s*-\s*"
        r"(\d{2}:\d{2}(?::\d{2})?)\]"
    )

    citations = re.findall(citation_pattern, answer)

    assert citations, (
        "No valid timestamp citations found in the answer."
    )

    print("\nTimestamp citation format validation passed!")

    # Check that cited timestamps belong to retrieved source ranges
    invalid_citations = []

    for start_text, end_text in citations:
        citation_start = parse_timestamp(start_text)
        citation_end = parse_timestamp(end_text)

        if citation_start > citation_end:
            invalid_citations.append(
                f"Invalid range: {start_text} - {end_text}"
            )
            continue

        matching_source = any(
            source["start"] <= citation_start
            and source["end"] >= citation_end
            for source in result["sources"]
        )

        if not matching_source:
            invalid_citations.append(
                f"{start_text} - {end_text}"
            )

    assert not invalid_citations, (
        f"Citations do not match retrieved source ranges: "
        f"{invalid_citations}"
    )

    print("Timestamp source range validation passed!")

    print("\nAll grounding evaluation checks passed!")


if __name__ == "__main__":
    test_rag_grounding()
