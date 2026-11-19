
from app.services.llm_service import LLMService


print("Initializing LLM service...")

llm_service = LLMService()


# Test 1: Generate an answer using the local Qwen model
prompt = """
You are an assistant for an audio-based RAG application.

Answer using only the provided context.

Context:
Admitting mistakes demonstrates self-reflection and humility.
It gives students permission to accept mistakes, learn from them,
and improve.

Question:
Why should teachers admit when they are wrong?

Provide a concise answer.
"""

print("\nGenerating response...")

answer = llm_service.generate_response(prompt)

print("\nGenerated answer:")
print(answer)


# Validate the response
assert isinstance(answer, str)
assert len(answer.strip()) > 0

print("\nLLM response validation passed!")


# Test 2: Empty prompt validation
try:

    llm_service.generate_response("   ")

    raise AssertionError(
        "Expected ValueError for empty prompt"
    )

except ValueError as error:

    assert "Prompt cannot be empty" in str(error)

    print("Empty prompt validation passed!")


print("\nAll LLM service tests passed!")
