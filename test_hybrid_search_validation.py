
from app.services.hybrid_search_service import HybridSearchService
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.chunk_repository import ChunkRepository


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

INVALID_DOCUMENT_ID = "nonexistent-document-id"

EMPTY_DOCUMENT_ID = "empty-test-document"


print("Initializing hybrid search service...")

hybrid_search_service = HybridSearchService()

embedding_repository = EmbeddingRepository()
chunk_repository = ChunkRepository()


# --------------------------------------------------
# Test 1: Existing document
# --------------------------------------------------

print("\nTest 1: Existing document retrieval")

results = hybrid_search_service.search(
    document_id=DOCUMENT_ID,
    query="What does the speaker say about self-reflection?",
    top_k=3
)

assert len(results) == 3
assert results[0]["chunk_index"] == 0

print("Existing document retrieval passed!")


# --------------------------------------------------
# Test 2: Nonexistent document
# --------------------------------------------------

print("\nTest 2: Nonexistent document retrieval")

try:

    hybrid_search_service.search(
        document_id=INVALID_DOCUMENT_ID,
        query="What is the discussion about?",
        top_k=3
    )

    raise AssertionError(
        "Expected ValueError for nonexistent document"
    )

except ValueError as error:

    assert "No embeddings found" in str(error)

    print("Nonexistent document validation passed!")


# --------------------------------------------------
# Test 3: Document with no embeddings
# --------------------------------------------------

print("\nTest 3: Document without embeddings")

try:

    hybrid_search_service.search(
        document_id=EMPTY_DOCUMENT_ID,
        query="What is the discussion about?",
        top_k=3
    )

    raise AssertionError(
        "Expected ValueError for document without embeddings"
    )

except ValueError as error:

    assert "No embeddings found" in str(error)

    print("Empty document validation passed!")


# --------------------------------------------------
# Test 4: Empty query
# --------------------------------------------------

print("\nTest 4: Empty query validation")

try:

    hybrid_search_service.search(
        document_id=DOCUMENT_ID,
        query="   ",
        top_k=3
    )

    raise AssertionError(
        "Expected ValueError for empty query"
    )

except ValueError as error:

    assert "Search query cannot be empty" in str(error)

    print("Empty query validation passed!")


# --------------------------------------------------
# Test 5: Invalid top_k
# --------------------------------------------------

print("\nTest 5: Invalid top_k validation")

try:

    hybrid_search_service.search(
        document_id=DOCUMENT_ID,
        query="What does the speaker say about teaching?",
        top_k=0
    )

    raise AssertionError(
        "Expected ValueError for invalid top_k"
    )

except ValueError as error:

    assert "top_k must be greater than zero" in str(error)

    print("Invalid top_k validation passed!")


print("\nAll hybrid search validation tests passed!")
