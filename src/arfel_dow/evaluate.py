import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from arfel_dow.detector import StackedEnsembleDetector
from arfel_dow.surrogate import Surrogate
from arfel_dow.scaling import FeatureScaler
from arfel_dow.attacks_gradient import pgd_attack
from arfel_dow.attacks_gan import AdaptiveGANAttacker
from arfel_dow.adversarial_training import train_robust_ensemble
from arfel_dow.pricing import financial_robustness_gap, AWSLambdaPricing
from arfel_dow.baseline import RateLimitBaseline
from arfel_dow.federated import FederatedEnsembleTrainer
from arfel_dow.narrative import generate_narrative
from arfel_dow.weighting import balanced_sample_weight


def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) into disjoint train/test partitions. Every fitting step
    in this module (plain ensemble, surrogate, scaler, federated client
    partitions, robust-ensemble augmentation) uses only the train
    partition; every accuracy figure this module reports is measured on
    the test partition, which no fitting step ever sees.

    Replaces an earlier version of clean_accuracy that scored a model
    against the exact data it had just been fit on — training accuracy,
    not generalization accuracy, silently inflating the reported baseline.

    `groups`, if given, is a per-row group id (e.g. source IP on the real
    DoWTS dataset) — every row sharing a group id is forced onto the same
    side of the split (via sklearn's GroupShuffleSplit). Without this, a
    single IP contributing hundreds of near-duplicate rows gets scattered
    across both train and test under plain row shuffling, letting the
    model partially recognize an IP it already saw during training — a
    leakage source distinct from (and not fixed by) the attack-row
    train/test split in run_evaluation. groups=None falls back to plain
    row-level shuffling, unchanged from the original behavior.

    Returns (X_train, y_train, X_test, y_test, train_indices, test_indices);
    the index arrays let callers slice parallel per-row arrays (e.g.
    durations_ms) consistently with whichever split they're using."""
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_indices, test_indices = next(splitter.split(X, y, groups=groups))
    else:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X))
        split = int(len(X) * (1 - test_size))
        train_indices, test_indices = perm[:split], perm[split:]
    return (
        X[train_indices], y[train_indices],
        X[test_indices], y[test_indices],
        train_indices, test_indices,
    )


def run_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    durations_ms: np.ndarray,
    memory_mb: np.ndarray,
    epsilon: float,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> dict:
    """epsilon is in standardized (zero-mean, unit-variance) feature units
    — see scaling.py's docstring. Attacks are crafted in standardized space
    and inverse-transformed before being scored against the real ensemble,
    so epsilon is comparably meaningful across the real dataset's
    heterogeneous raw feature scales (single digits to the thousands).

    groups, if given (e.g. source IP), forces every row sharing a group id
    onto the same side of the train/test split — see _train_test_split's
    docstring for why this matters on the real dataset.

    (X, y) is split into train/test partitions (see _train_test_split).
    Every model is fit on the train partition only; every accuracy figure
    is measured on the test partition. Attack-class rows used to augment
    the robust ensemble's training set come from the train partition;
    attack-class rows that get attacked for evaluation come from the test
    partition — disjoint by construction, so the robust model is never
    scored on rows it was trained on (see train_robust_ensemble's
    docstring).

    (X, y) is expected to reflect the dataset's natural class distribution
    (not pre-downsampled) — every StackedEnsembleDetector fit below is
    weighted via balanced_sample_weight (see weighting.py), which trains
    on the full imbalanced data instead of discarding real rows."""
    X_train, y_train, X_test, y_test, _, test_indices = _train_test_split(
        X, y, test_size=0.3, random_state=random_state, groups=groups
    )

    plain_ensemble = StackedEnsembleDetector(random_state=random_state).fit(
        X_train, y_train, sample_weight=balanced_sample_weight(y_train)
    )
    y_pred_clean_test = plain_ensemble.predict(X_test)
    clean_accuracy = float((y_pred_clean_test == y_test).mean())
    # Blended accuracy on an imbalanced test set can look fine while one
    # class is barely being detected — exactly what happened on the real
    # DoWTS dataset (good benign recall, poor attack recall, masked by a
    # single accuracy number when the test set skews toward the class
    # being missed). Report both recalls explicitly rather than only the
    # blended figure.
    clean_attack_recall = float((y_pred_clean_test[y_test == 1] == 1).mean())
    clean_benign_recall = float((y_pred_clean_test[y_test == 0] == 0).mean())

    scaler = FeatureScaler().fit(X_train)
    surrogate = Surrogate(input_dim=X.shape[1], random_state=random_state)
    surrogate.fit(scaler.transform(X_train), plain_ensemble.predict_proba(X_train), epochs=200)

    attack_mask_train = y_train == 1
    X_attack_train, y_attack_train = X_train[attack_mask_train], y_train[attack_mask_train]

    attack_mask_test = y_test == 1
    X_attack_test, y_attack_test = X_test[attack_mask_test], y_test[attack_mask_test]
    attack_test_original_indices = test_indices[attack_mask_test]

    X_attack_test_scaled = scaler.transform(X_attack_test)
    X_adv_test_scaled = pgd_attack(
        surrogate, X_attack_test_scaled, y_attack_test, epsilon=epsilon, alpha=epsilon / 8, steps=15
    )
    X_adv_test = scaler.inverse_transform(X_adv_test_scaled)

    y_pred_clean_attack_rows = plain_ensemble.predict(X_attack_test)
    y_pred_adv_plain = plain_ensemble.predict(X_adv_test)
    adversarial_accuracy_plain = float((y_pred_adv_plain == y_attack_test).mean())

    robust_ensemble = train_robust_ensemble(
        X_train, y_train, X_attack_train, y_attack_train, surrogate, scaler,
        epsilon=epsilon, random_state=random_state,
    )
    y_pred_adv_robust = robust_ensemble.predict(X_adv_test)
    adversarial_accuracy_robust = float((y_pred_adv_robust == y_attack_test).mean())

    # Cross-attack-method validation: robust_ensemble was adversarially
    # trained only against PGD-crafted examples. Testing it against a
    # structurally different attack (GAN, trained on train-partition rows
    # only — never the held-out test rows) checks whether that training
    # generalizes to adversarial perturbation in general, or overfits to
    # PGD's specific perturbation pattern. Either answer is a real,
    # reportable finding, not a pass/fail bar.
    X_benign_train_scaled = scaler.transform(X_train[y_train == 0])
    X_attack_train_scaled = scaler.transform(X_attack_train)
    gan_attacker = AdaptiveGANAttacker(input_dim=X.shape[1], noise_dim=8, random_state=random_state)
    gan_attacker.fit(
        X_benign=X_benign_train_scaled, X_attack=X_attack_train_scaled, surrogate=surrogate, epochs=100
    )
    X_gan_adv_scaled = gan_attacker.generate(n_samples=len(X_attack_test))
    X_gan_adv = scaler.inverse_transform(X_gan_adv_scaled)
    y_gan_labels = np.ones(len(X_gan_adv), dtype=np.int64)

    adversarial_accuracy_plain_gan = float((plain_ensemble.predict(X_gan_adv) == y_gan_labels).mean())
    adversarial_accuracy_robust_gan = float((robust_ensemble.predict(X_gan_adv) == y_gan_labels).mean())

    baseline = RateLimitBaseline(delay_threshold=5.0, invocation_delay_column_index=1)
    baseline_clean_preds = baseline.predict(X_attack_test)
    baseline_clean_accuracy = float((baseline_clean_preds == y_attack_test).mean())
    baseline_adv_preds = baseline.predict(X_adv_test)
    baseline_adversarial_accuracy = float((baseline_adv_preds == y_attack_test).mean())

    durations_attack_test = durations_ms[attack_test_original_indices]
    memory_attack_test = memory_mb[attack_test_original_indices]

    financial_gap_plain = float(
        financial_robustness_gap(
            y_attack_test, y_pred_clean_attack_rows, y_pred_adv_plain, durations_attack_test, memory_attack_test
        )
    )
    financial_gap_robust = float(
        financial_robustness_gap(
            y_attack_test, y_pred_clean_attack_rows, y_pred_adv_robust, durations_attack_test, memory_attack_test
        )
    )

    return {
        "clean_accuracy": clean_accuracy,
        "clean_attack_recall": clean_attack_recall,
        "clean_benign_recall": clean_benign_recall,
        "adversarial_accuracy_plain": adversarial_accuracy_plain,
        "adversarial_accuracy_robust": adversarial_accuracy_robust,
        "adversarial_accuracy_plain_gan": adversarial_accuracy_plain_gan,
        "adversarial_accuracy_robust_gan": adversarial_accuracy_robust_gan,
        "baseline_clean_accuracy": baseline_clean_accuracy,
        "baseline_adversarial_accuracy": baseline_adversarial_accuracy,
        "financial_robustness_gap_plain": financial_gap_plain,
        "financial_robustness_gap_robust": financial_gap_robust,
    }


def _partition_for_clients(
    X: np.ndarray, y: np.ndarray, n_clients: int, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split (X, y) into n_clients simulated-tenant partitions. Shuffles
    first so each partition gets a mix of classes regardless of the
    caller's row ordering — contiguous slicing of unshuffled class-grouped
    data would otherwise hand some clients a single-class partition, which
    StackedEnsembleDetector's base learners cannot fit.

    Deliberately plain row-level shuffling, not group-aware — this splits
    the already-train-only partition among simulated tenants, not the
    outer train/test boundary the group-split in _train_test_split
    protects. A given source IP's rows landing in more than one simulated
    tenant isn't a generalization-validity problem the way it is across
    the train/test boundary."""
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(X))
    X_shuffled, y_shuffled = X[perm], y[perm]
    boundaries = np.linspace(0, len(X_shuffled), n_clients + 1, dtype=int)
    return [
        (X_shuffled[boundaries[i]:boundaries[i + 1]], y_shuffled[boundaries[i]:boundaries[i + 1]])
        for i in range(n_clients)
    ]


def _z_score_attributions(
    x_row: np.ndarray, benign_mean: np.ndarray, benign_std: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Deviation of a row from the benign-class distribution, in standard
    deviations per feature — a lightweight, dependency-free stand-in for a
    real SHAP explainer, matching generate_narrative's documented contract
    that attributions are supplied by the caller."""
    z_scores = (x_row - benign_mean) / benign_std
    return {name: float(z) for name, z in zip(feature_names, z_scores)}


def run_full_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    durations_ms: np.ndarray,
    memory_mb: np.ndarray,
    epsilon: float,
    feature_names: list[str],
    n_clients: int = 3,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> dict:
    """Full AR-FEL-DoW pipeline: the base run_evaluation results, plus
    federated ensemble aggregation (Task 12) and an LLM narrative for one
    sample detection (Task 13) — the five-pillar architecture (detector,
    adversarial attacks, adversarial training, federated learning, LLM
    interpretability) wired together in a single call. Uses the same
    train/test split discipline as run_evaluation (see _train_test_split):
    federated client models are trained only on the train partition,
    federated_clean_accuracy is measured on the held-out test partition.

    groups is forwarded to run_evaluation and to this function's own
    (deterministically identical) internal split — see run_evaluation's
    docstring."""
    base_results = run_evaluation(
        X, y, durations_ms, memory_mb, epsilon=epsilon, random_state=random_state, groups=groups
    )

    # Same deterministic split as run_evaluation (same X, y, random_state,
    # groups) — reused here rather than passed through base_results so
    # run_evaluation's tested return-key contract stays unchanged.
    X_train, y_train, X_test, y_test, _, test_indices = _train_test_split(
        X, y, test_size=0.3, random_state=random_state, groups=groups
    )

    client_partitions = _partition_for_clients(X_train, y_train, n_clients, random_state=random_state)
    federated_trainer = FederatedEnsembleTrainer().fit(client_partitions, random_state=random_state)
    federated_clean_accuracy = float((federated_trainer.predict(X_test) == y_test).mean())

    plain_ensemble = StackedEnsembleDetector(random_state=random_state).fit(
        X_train, y_train, sample_weight=balanced_sample_weight(y_train)
    )
    scaler = FeatureScaler().fit(X_train)
    surrogate = Surrogate(input_dim=X.shape[1], random_state=random_state)
    surrogate.fit(scaler.transform(X_train), plain_ensemble.predict_proba(X_train), epochs=200)

    attack_mask_test = y_test == 1
    X_attack_test, y_attack_test = X_test[attack_mask_test], y_test[attack_mask_test]
    attack_test_original_indices = test_indices[attack_mask_test]

    X_attack_test_scaled = scaler.transform(X_attack_test)
    X_adv_test_scaled = pgd_attack(
        surrogate, X_attack_test_scaled, y_attack_test, epsilon=epsilon, alpha=epsilon / 8, steps=15
    )
    X_adv_test = scaler.inverse_transform(X_adv_test_scaled)
    federated_adversarial_accuracy = float((federated_trainer.predict(X_adv_test) == y_attack_test).mean())

    benign_mean = X_train[y_train == 0].mean(axis=0)
    benign_std = X_train[y_train == 0].std(axis=0) + 1e-6

    sample_row = X_adv_test[0]
    sample_attributions = _z_score_attributions(sample_row, benign_mean, benign_std, feature_names)
    sample_proba = federated_trainer.predict_proba(sample_row.reshape(1, -1))[0]
    sample_label = "attack" if sample_proba.argmax() == 1 else "benign"

    pricing = AWSLambdaPricing()
    sample_original_index = attack_test_original_indices[0]
    sample_financial_impact = pricing.invocation_cost(
        duration_ms=float(durations_ms[sample_original_index]),
        memory_mb=float(memory_mb[sample_original_index]),
        n_invocations=1,
    )

    sample_narrative = generate_narrative(
        feature_values={name: float(v) for name, v in zip(feature_names, sample_row)},
        anomaly_score=float(sample_proba[1]),
        feature_attributions=sample_attributions,
        predicted_label=sample_label,
        financial_impact=sample_financial_impact,
    )

    return {
        **base_results,
        "federated_clean_accuracy": federated_clean_accuracy,
        "federated_adversarial_accuracy": federated_adversarial_accuracy,
        "sample_narrative": sample_narrative,
    }
