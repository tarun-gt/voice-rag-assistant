
from app.services.grounding_evaluation_service import (
    GroundingEvaluationService
)


def test_grounding_evaluation():
    evaluator = GroundingEvaluationService()

    context = """
    A teacher should be willing to say, "I am wrong."
    When teachers admit their mistakes, they model self-reflection
    and humility.
    This gives students permission to accept that they can also be
    wrong, learn from their mistakes, and improve.
    """

    answer = (
        "Admitting mistakes demonstrates humility and self-reflection. "
        "It helps students learn from their mistakes and improve. "
        "It also guarantees that students will become successful "
        "in their careers."
    )

    print("Evaluating generated answer...")

    result = evaluator.evaluate(
        answer=answer,
        context=context
    )

    print("\nGrounding evaluation results:")
    print("=" * 70)

    for index, claim in enumerate(result["claims"], start=1):
        print(f"\nClaim {index}: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Evidence: {claim['evidence']}")

    verdicts = [claim["verdict"] for claim in result["claims"]]

    assert "SUPPORTED" in verdicts, (
        "Expected at least one supported claim"
    )

    assert (
        "UNSUPPORTED" in verdicts
        or "INSUFFICIENT_EVIDENCE" in verdicts
    ), "Expected at least one unsupported or insufficient claim"

    print("\nGrounding evaluation test passed!")

def test_abstention_evaluation():
    print("\nTesting abstention evaluation...")

    answer = (
        "The provided transcript does not contain enough "
        "information to determine the speaker's home address."
    )

    context = """
    Chunk index: 20
    Timestamp: [09:24 - 10:15]
    Transcript:
    The speaker mentions his name and that he is from Colombia.
    """

    service = GroundingEvaluationService()

    result = service.evaluate(
        answer=answer,
        context=context
    )

    for claim in result["claims"]:
        print(f"Claim: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Evidence: {claim['evidence']}")

    assert len(result["claims"]) > 0

    assert all(
        claim["verdict"] == "SUPPORTED"
        for claim in result["claims"]
    ), "Abstention statement should be recognized as supported."

    print("\nAbstention evaluation test passed!")

def test_partial_evidence_evaluation():
    print("\nTesting partial evidence evaluation...")

    answer = (
        "The speaker is from Colombia and lives in New York."
    )

    context = """
    Chunk index: 20
    Timestamp: [09:24 - 10:15]
    Transcript:
    My name is Aldemar. I'm from Colombia.
    """

    service = GroundingEvaluationService()

    result = service.evaluate(
        answer=answer,
        context=context
    )

    for claim in result["claims"]:
        print(f"Claim: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Evidence: {claim['evidence']}")

    assert len(result["claims"]) >= 2, (
        "Compound claim should be broken into atomic claims."
    )

    assert any(
        claim["verdict"] in {"UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}
        for claim in result["claims"]
        if "New York" in claim["claim"]
    ), "Unsupported location claim was incorrectly accepted."

    print("\nPartial evidence evaluation test passed!")

def test_contradictory_evidence_evaluation():
    print("\nTesting contradictory evidence evaluation...")

    answer = "The speaker is from New York."

    context = """
    Chunk index: 20
    Timestamp: [09:24 - 10:15]
    Transcript:
    My name is Aldemar. I'm from Colombia.
    """

    service = GroundingEvaluationService()

    result = service.evaluate(
        answer=answer,
        context=context
    )

    for claim in result["claims"]:
        print(f"Claim: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Evidence: {claim['evidence']}")

    assert len(result["claims"]) > 0

    assert any(
        claim["verdict"] == "UNSUPPORTED"
        for claim in result["claims"]
    ), "Contradicted claim should be classified as UNSUPPORTED."

    print("\nContradictory evidence evaluation test passed!")

if __name__ == "__main__":
    test_grounding_evaluation()
    test_abstention_evaluation()
    test_partial_evidence_evaluation()
    test_contradictory_evidence_evaluation()