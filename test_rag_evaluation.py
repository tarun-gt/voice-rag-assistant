
from app.services.rag_service import RAGService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"


EVALUATION_CASES = [
    {
        "name": "Direct factual question",
        "question": "What are the three most powerful words a teacher can say?",
        "expected_terms": ["I am wrong"],
        "should_abstain": False
    },
    {
        "name": "Paraphrased question",
        "question": "Why should teachers admit when they make mistakes?",
        "expected_terms": ["self-reflection", "humility"],
        "should_abstain": False
    },
    {
        "name": "Conceptual question",
        "question": "How does admitting mistakes demonstrate humility?",
        "expected_terms": ["self-reflection", "humility"],
        "should_abstain": False
    },
    {
        "name": "Unanswerable birth year question",
        "question": "What year was the speaker born?",
        "expected_terms": [],
        "should_abstain": True
    },
    {
        "name": "Misleading graduation question",
        "question": "What exact year did the speaker graduate from Yale?",
        "expected_terms": [],
        "should_abstain": True
    },
    {
        "name": "Unanswerable personal information",
        "question": "What is the speaker's home address?",
        "expected_terms": [],
        "should_abstain": True
    }
]


def evaluate_case(rag_service, test_case):
    print("\n" + "=" * 80)
    print(f"Test: {test_case['name']}")
    print(f"Question: {test_case['question']}")

    result = rag_service.ask(
        document_id=DOCUMENT_ID,
        query=test_case['question'],
        evaluate_grounding=True
    )

    answer = result["answer"]
    answer_lower = answer.lower()

    print("\nRetrieved source chunks:")
    for source in result["sources"]:
        print(f"\nChunk index: {source['chunk_index']}")
        print(source["text"])

    print(f"\nAnswer:\n{answer}")

    # 1. Validate expected answer terms
    missing_terms = [
        term for term in test_case["expected_terms"]
        if term.lower() not in answer_lower
    ]

    if missing_terms:
        raise AssertionError(
            f"Expected information missing: {missing_terms}"
        )

    # 2. Validate answerability
    abstention_phrases = [
        "does not provide enough information",
        "not enough information",
        "couldn't find relevant information",
        "cannot determine",
        "not specified in the transcript",
        "does not mention"
    ]

    is_abstaining = any(
        phrase in answer_lower
        for phrase in abstention_phrases
    )

    if test_case["should_abstain"] and not is_abstaining:
        raise AssertionError(
            "Expected the RAG system to abstain, but it generated "
            "a substantive answer."
        )

    if not test_case["should_abstain"] and is_abstaining:
        raise AssertionError(
            "Expected an answer, but the RAG system abstained."
        )

    # 3. Validate factual grounding
    grounding_evaluation = result["grounding_evaluation"]
    grounding_summary = result["grounding_summary"]

    print("\nGrounding evaluation:")
    print("-" * 80)

    for claim in grounding_evaluation["claims"]:
        print(f"Claim: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Evidence: {claim['evidence']}")
        print()

    print("Grounding summary:")
    print(grounding_summary)

    # Any unsupported or insufficiently supported claim fails the test.
    if grounding_summary["unsupported"] > 0:
        raise AssertionError(
            f"Answer contains "
            f"{grounding_summary['unsupported']} unsupported claims."
        )

    if grounding_summary["insufficient_evidence"] > 0:
        raise AssertionError(
            f"Answer contains "
            f"{grounding_summary['insufficient_evidence']} claims "
            f"with insufficient evidence."
        )

    print("\nAnswer and grounding validation passed!")
    print(
        f"Expected behavior: "
        f"{'Abstention' if test_case['should_abstain'] else 'Answer'}"
    )


def test_rag_evaluation():
    print("Initializing RAG service...")
    rag_service = RAGService()

    passed = 0
    failed = 0

    for test_case in EVALUATION_CASES:

        try:
            evaluate_case(rag_service, test_case)
            passed += 1

        except (AssertionError, RuntimeError, ValueError) as error:
            failed += 1
            print(f"\nTEST FAILED: {error}")

    print("\n" + "=" * 80)
    print("RAG EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(EVALUATION_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    assert failed == 0, (
        f"{failed} out of {len(EVALUATION_CASES)} evaluation cases failed."
    )


if __name__ == "__main__":
    test_rag_evaluation()
