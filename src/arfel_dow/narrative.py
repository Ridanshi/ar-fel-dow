from typing import Callable, Optional


def _top_features_by_attribution(feature_attributions: dict[str, float], n: int = 2) -> list[str]:
    ranked = sorted(feature_attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [name for name, _ in ranked[:n]]


def _build_prompt(
    feature_values: dict[str, float],
    anomaly_score: float,
    feature_attributions: dict[str, float],
    predicted_label: str,
    financial_impact: Optional[float],
) -> str:
    top_features = _top_features_by_attribution(feature_attributions)
    factor_descriptions = ", ".join(
        f"{name}={feature_values.get(name)} (attribution {feature_attributions[name]:+.2f})"
        for name in top_features
    )
    lines = [
        f"Predicted label: {predicted_label}",
        f"Anomaly score: {anomaly_score:.2f}",
        f"Top contributing factors: {factor_descriptions}",
    ]
    if financial_impact is not None:
        lines.append(f"Estimated financial impact: ${financial_impact:.4f}")
    lines.append(
        "Write a one-paragraph, human-readable incident summary for a SOC/finance "
        "analyst based on the facts above."
    )
    return "\n".join(lines)


def _template_narrative(
    feature_values: dict[str, float],
    anomaly_score: float,
    feature_attributions: dict[str, float],
    predicted_label: str,
    financial_impact: Optional[float],
) -> str:
    top_features = _top_features_by_attribution(feature_attributions)
    factor_text = " and ".join(top_features)
    sentence = (
        f"Session classified as {predicted_label} with anomaly score {anomaly_score:.2f}. "
        f"Primary contributing factors: {factor_text}."
    )
    if financial_impact is not None:
        sentence += f" Estimated financial impact: ${financial_impact:.4f}."
    return sentence


def generate_narrative(
    feature_values: dict[str, float],
    anomaly_score: float,
    feature_attributions: dict[str, float],
    predicted_label: str,
    financial_impact: Optional[float] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Render a detection into a human-readable incident narrative.

    Attributions must be supplied by the caller (e.g. from a SHAP explainer
    wired up separately) — this function only renders text, it does not
    compute feature importance. Without llm_fn, falls back to a deterministic
    template (no network dependency, fully unit-testable). With llm_fn, the
    facts are assembled into a prompt and handed to the callable — the real
    LLM integration point, matching AR-FEL's LLM-driven post-hoc
    interpretability module, retargeted at DoW detections instead of IDS
    alerts."""
    if llm_fn is not None:
        prompt = _build_prompt(
            feature_values, anomaly_score, feature_attributions, predicted_label, financial_impact
        )
        return llm_fn(prompt)

    return _template_narrative(
        feature_values, anomaly_score, feature_attributions, predicted_label, financial_impact
    )
