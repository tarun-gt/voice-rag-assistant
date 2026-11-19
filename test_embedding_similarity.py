
from app.services.embedding_service import EmbeddingService


# 1. Initialize embedding service
embedding_service = EmbeddingService()


# 2. Define three sentences
sentence_1 = "Artificial intelligence is transforming the world."

sentence_2 = "AI is changing how people live and work."

sentence_3 = "I enjoy cooking delicious Italian food."


# 3. Generate embeddings
embedding_1 = embedding_service.generate_embedding(sentence_1)
embedding_2 = embedding_service.generate_embedding(sentence_2)
embedding_3 = embedding_service.generate_embedding(sentence_3)


# 4. Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):

    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )


similarity_1_2 = cosine_similarity(
    embedding_1,
    embedding_2
)

similarity_1_3 = cosine_similarity(
    embedding_1,
    embedding_3
)


# 5. Display results
print("\nSemantic Similarity Results")

print(
    f"Sentence 1 vs Sentence 2: "
    f"{similarity_1_2:.4f}"
)

print(
    f"Sentence 1 vs Sentence 3: "
    f"{similarity_1_3:.4f}"
)


# 6. Validate semantic similarity
assert similarity_1_2 > similarity_1_3, (
    "Similar sentences should have higher similarity!"
)

print("\nSemantic similarity test passed!")
