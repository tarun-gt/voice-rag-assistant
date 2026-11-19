from app.services.rag_service import RAGService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

QUESTIONS = [
    "What year was the speaker born?",
    "Which university did the speaker graduate from?",
    "What is the speaker's home address?"
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
        top_k=3
    )

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nRetrieved Sources:")
    for source in result["sources"]:
        print(
            f"Chunk {source['chunk_index']}: "
            f"{source['start']} - {source['end']}"
        )

    print("\n" + "-" * 70)

print("\nAbstention tests completed.")