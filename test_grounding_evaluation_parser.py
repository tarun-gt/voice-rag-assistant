
from app.services.grounding_evaluation_service import (
    GroundingEvaluationService
)


def test_valid_json():
    evaluator = GroundingEvaluationService()

    response = """
    {
        "claims": [
            {
                "claim": "Teachers model humility.",
                "verdict": "SUPPORTED",
                "evidence": "They model self-reflection and humility."
            }
        ]
    }
    """

    result = evaluator._parse_evaluation(response)

    assert len(result["claims"]) == 1
    assert result["claims"][0]["verdict"] == "SUPPORTED"

    print("Valid JSON test passed!")


def test_markdown_json():
    evaluator = GroundingEvaluationService()

    response = """```json
    {
        "claims": [
            {
                "claim": "Students can improve.",
                "verdict": "SUPPORTED",
                "evidence": "Learn from their mistakes and improve."
            }
        ]
    }
    ```"""

    result = evaluator._parse_evaluation(response)

    assert result["claims"][0]["claim"] == "Students can improve."

    print("Markdown JSON test passed!")


def test_invalid_json():
    evaluator = GroundingEvaluationService()

    try:
        evaluator._parse_evaluation('{"claims": [}')
        raise AssertionError("Expected invalid JSON to raise ValueError")
    except ValueError as error:
        assert "invalid JSON" in str(error)

    print("Invalid JSON test passed!")


def test_missing_claims():
    evaluator = GroundingEvaluationService()

    try:
        evaluator._parse_evaluation('{"result": "SUPPORTED"}')
        raise AssertionError("Expected missing claims to raise ValueError")
    except ValueError as error:
        assert "claims list" in str(error)

    print("Missing claims test passed!")


def test_invalid_verdict():
    evaluator = GroundingEvaluationService()

    response = """
    {
        "claims": [
            {
                "claim": "Teachers model humility.",
                "verdict": "MAYBE",
                "evidence": ""
            }
        ]
    }
    """

    try:
        evaluator._parse_evaluation(response)
        raise AssertionError("Expected invalid verdict to raise ValueError")
    except ValueError as error:
        assert "Invalid grounding verdict" in str(error)

    print("Invalid verdict test passed!")


def test_empty_claim():
    evaluator = GroundingEvaluationService()

    response = """
    {
        "claims": [
            {
                "claim": "",
                "verdict": "SUPPORTED",
                "evidence": "Some evidence"
            }
        ]
    }
    """

    try:
        evaluator._parse_evaluation(response)
        raise AssertionError("Expected empty claim to raise ValueError")
    except ValueError as error:
        assert "non-empty claim text" in str(error)

    print("Empty claim test passed!")


if __name__ == "__main__":
    test_valid_json()
    test_markdown_json()
    test_invalid_json()
    test_missing_claims()
    test_invalid_verdict()
    test_empty_claim()

    print("\nAll grounding parser tests passed!")
