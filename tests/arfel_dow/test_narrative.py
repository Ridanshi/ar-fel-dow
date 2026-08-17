from arfel_dow.narrative import generate_narrative


def test_template_narrative_includes_label_score_and_top_features():
    narrative = generate_narrative(
        feature_values={
            "InvocationDelay": 1.5,
            "ResponseDelay": 310.0,
            "FunctionDuration": 1010.0,
        },
        anomaly_score=0.93,
        feature_attributions={
            "InvocationDelay": -0.41,
            "ResponseDelay": 0.35,
            "FunctionDuration": 0.02,
        },
        predicted_label="attack",
        financial_impact=0.0042,
    )
    assert "attack" in narrative
    assert "0.93" in narrative
    assert "InvocationDelay" in narrative
    assert "ResponseDelay" in narrative
    # FunctionDuration has the smallest |attribution| of the three, so it
    # must not appear among the top-2 factors named in the narrative.
    assert "FunctionDuration" not in narrative
    assert "0.0042" in narrative


def test_template_narrative_omits_financial_sentence_when_not_provided():
    narrative = generate_narrative(
        feature_values={"InvocationDelay": 12.0},
        anomaly_score=0.10,
        feature_attributions={"InvocationDelay": 0.05},
        predicted_label="benign",
        financial_impact=None,
    )
    assert "benign" in narrative
    assert "$" not in narrative


def test_llm_fn_hook_receives_prompt_and_its_output_is_returned():
    captured_prompts = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "MOCKED NARRATIVE OUTPUT"

    narrative = generate_narrative(
        feature_values={"InvocationDelay": 1.5, "ResponseDelay": 310.0},
        anomaly_score=0.93,
        feature_attributions={"InvocationDelay": -0.41, "ResponseDelay": 0.35},
        predicted_label="attack",
        financial_impact=0.0042,
        llm_fn=fake_llm,
    )

    assert narrative == "MOCKED NARRATIVE OUTPUT"
    assert len(captured_prompts) == 1
    assert "InvocationDelay" in captured_prompts[0]
    assert "attack" in captured_prompts[0]
