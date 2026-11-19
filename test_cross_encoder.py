
from sentence_transformers import CrossEncoder

print("Loading cross-encoder model...")

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)

print("Cross-encoder model loaded successfully!")

query = "Which university does the speaker say they attended?"

candidate_chunks = [
    "And what is right in your heart is to pursue the truth. I could imagine myself working a job. I didn't like saying things that I didn't believe.",
    "I went to Yale. I'm a very good writer. I'm a very good communicator.",
    "I would say the three most powerful words you can ever utter as a teacher is I am wrong."
]

pairs = [
    [query, chunk]
    for chunk in candidate_chunks
]

scores = model.predict(pairs)

ranked_results = sorted(
    zip(candidate_chunks, scores),
    key=lambda item: item[1],
    reverse=True
)

print("\nReranking results:")

for rank, (chunk, score) in enumerate(ranked_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Score: {float(score):.4f}")
    print(f"Text: {chunk}")
