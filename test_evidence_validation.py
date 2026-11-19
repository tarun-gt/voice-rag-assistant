from app.services.grounding_evaluation_service import (
    GroundingEvaluationService
)


def test_evidence_validation():
    service = GroundingEvaluationService()

    context = (
        "I've absolutely refused to compromise, because, "
        "you know, I went to Yale."
    )

    # Case 1: Exact transcript evidence with the LLM's label
    # and quotation marks should pass.
    evaluation = {
        "claims": [
            {
                "claim": "The speaker went to Yale.",
                "verdict": "SUPPORTED",
                "evidence": (
                    'Transcript evidence: '
                    '"I\'ve absolutely refused to compromise, because, '
                    'you know, I went to Yale."'
                )
            }
        ]
    }

    result = service._validate_evidence(evaluation, context)
    assert result == evaluation

    print("PASS: Exact evidence with label and quotes.")

    # Case 2: Fabricated evidence must still fail.
    invalid_evaluation = {
        "claims": [
            {
                "claim": "The speaker graduated from Yale in 2010.",
                "verdict": "SUPPORTED",
                "evidence": (
                    "The speaker graduated from Yale in 2010."
                )
            }
        ]
    }

    try:
        service._validate_evidence(invalid_evaluation, context)
        raise AssertionError(
            "Fabricated evidence was incorrectly accepted."
        )
    except ValueError as error:
        assert "does not exist in the transcript context" in str(error)
        print("PASS: Fabricated evidence was rejected.")

    print("\nAll evidence validation tests passed.")


if __name__ == "__main__":
    test_evidence_validation()