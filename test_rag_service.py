
from app.services.rag_service import RAGService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

QUESTIONS = [
    "What are the three most powerful words a teacher can say?",
    "Why should teachers admit when they are wrong?",
    "How does admitting mistakes demonstrate humility?"
]


print("Initializing RAG service...")

rag_service = RAGService()


for index, question in enumerate(QUESTIONS, start=1):

    print("\n" + "=" * 70)
    print(f"TEST {index}")
    print("=" * 70)

    print(f"\nQuestion: {question}")

    result = rag_service.ask(
        document_id=DOCUMENT_ID,
        query=question,
        top_k=5
    )

    print("\nRetrieved source texts:")

    for source in result["sources"]:
        print(f"\nChunk: {source['chunk_index']}")
        print(f"Text: {source['text']}")

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nRetrieved Sources:")

    for source in result["sources"]:

        print(
            f"\nChunk: {source['chunk_index']}"
            f"\nTimestamp: {source['start']} - {source['end']}"
            f"\nRRF Score: {source['rrf_score']:.6f}"
            f"\nText: {source['text'][:250]}..."
        )

    # Validate response structure
    assert result["document_id"] == DOCUMENT_ID
    assert result["question"] == question
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0
    assert len(result["sources"]) > 0

    print("\nRAG response validation passed!")


print("\nAll RAG integration tests completed successfully!")
