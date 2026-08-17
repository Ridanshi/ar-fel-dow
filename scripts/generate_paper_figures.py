"""Generate the 7 figures referenced in docs/paper/AR-FEL-DoW-paper.md.

Figure numbers below match each figure's first appearance in the paper's
document order (Section 4 architecture diagram, then Results 6.2/6.5/6.6/
6.7/6.8/6.9 in that order), not the order these functions are defined in.

Not part of the tested package — a one-shot generation script whose
output (PNG files) is committed and embedded in the paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import numpy as np

OUT_DIR = "docs/paper/figures"


def figure1_architecture():
    # Grid layout, generous spacing, no crossing arrows: a top chain
    # (dataset -> features -> scaler -> detector) feeds a central hub
    # (the detector); the surrogate/attack/adversarial-training chain
    # runs down the left column and loops back to the detector via a
    # single wide arc in the empty left margin; federated learning and
    # the baseline/financial-gap chain run down the right column;
    # the narrative layer sits directly below the detector.
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(-3.4, 14.2)
    ax.set_ylim(3.0, 13.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8eef7", ec="#2c3e50", fontsize=9.5):
        b = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.14",
            fc=fc, ec=ec, lw=1.4,
        )
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.5)
        return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)

    def arrow(p1, p2, **kw):
        kw.setdefault("arrowstyle", "-|>")
        kw.setdefault("mutation_scale", 15)
        kw.setdefault("lw", 1.3)
        kw.setdefault("color", "#2c3e50")
        a = FancyArrowPatch(p1, p2, **kw)
        ax.add_patch(a)

    # Row 1 (top): dataset -> feature engineering, centered
    dataset = box(1.0, 11.7, 3.2, 1.1, "DoWTS Dataset\n(187,088 rows,\nMendeley, CC BY 4.0)")
    features = box(5.0, 11.7, 3.7, 1.1, "Feature Engineering\n(19 columns + causal\nIP-frequency feature)")
    arrow((dataset[6], dataset[7]), (features[4], features[5]))

    # Row 2: scaler, centered under feature engineering
    scaler = box(5.0, 10.0, 3.7, 1.0, "FeatureScaler\n(standardized feature space)")
    arrow((features[0], features[1]), (scaler[2], scaler[3]))

    # Row 3: surrogate | detector (hub) | federated ensemble
    detector = box(3.9, 8.0, 5.9, 1.3, "Stacked Ensemble Detector\nXGBoost + RandomForest + CatBoost + MLP\n→ Logistic Regression meta-learner", fontsize=10)
    arrow((scaler[0], scaler[1]), (detector[0], detector[3]))

    surrogate = box(-1.6, 8.15, 3.1, 1.0, "Differentiable\nSurrogate (MLP)")
    arrow((detector[4], detector[5]), (surrogate[6], surrogate[7]))

    fed = box(10.4, 8.15, 3.3, 1.0, "Federated Ensemble\n(3 simulated tenants,\nsoft-vote aggregation)")
    arrow((detector[6], detector[7]), (fed[4], fed[5]))

    # Row 4: attack suite (left) | narrative layer (center) | rate-limit baseline (right)
    attacks = box(-1.8, 6.3, 3.5, 1.1, "Attack Suite\nFGSM / PGD / C&W\n+ GAN Adaptive Attacker")
    arrow((surrogate[0], surrogate[1]), (attacks[2], attacks[3]))

    narrative = box(5.0, 6.3, 3.7, 1.1, "LLM Narrative Layer\n(anomaly score + top\nfeature attributions)")
    arrow((detector[0], detector[1]), (narrative[2], narrative[3]))

    baseline = box(10.2, 6.3, 3.7, 1.1, "Rate-Limit Baseline\n(independent, practical-\ndeployment comparison)")

    # Row 5: adversarial training (left) | financial robustness gap (right)
    adv_train = box(-1.9, 4.4, 3.7, 1.1, "Adversarial Training\n(disjoint attack-row\ntrain/eval split)")
    arrow((attacks[0], attacks[1]), (adv_train[2], adv_train[3]))

    financial = box(10.2, 4.4, 3.7, 1.1, "Financial Robustness Gap\n(real AWS Lambda pricing)")
    arrow((baseline[0], baseline[1]), (financial[2], financial[3]))

    # Loop-back: adversarially-trained model replaces the plain detector.
    # Built as an explicit 3-segment path (left, up, right) through a
    # margin at x=-3.0 — clear of every box's left edge (leftmost box
    # starts at x=-1.9) — rather than relying on a connectionstyle to
    # infer a direction from just two points, which routed straight
    # through the box instead of around it. The endpoint's y must clear
    # the surrogate box's bottom edge (y=8.15): detector[1]+0.05=8.05
    # passes under it with margin, where the earlier +0.15 offset
    # (8.15) landed exactly on that edge and grazed straight through it.
    loop_start = (adv_train[4], adv_train[7])
    loop_end = (detector[4], detector[1] + 0.05)
    margin_x = -3.0
    loop_path = Path(
        [loop_start, (margin_x, loop_start[1]), (margin_x, loop_end[1]), loop_end],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO],
    )
    loop_arrow = FancyArrowPatch(
        path=loop_path, arrowstyle="-|>", mutation_scale=15, lw=1.3,
        color="#2c3e50", linestyle="dashed",
    )
    ax.add_patch(loop_arrow)

    ax.set_title("Figure 1. AR-FEL-DoW System Architecture", fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure1_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure2_ip_ablation():
    # Section 6.2, seed 42 single-run ablation.
    labels = ["Attack recall", "Benign recall", "Overall accuracy"]
    without_feat = [46.4, 82.8, 59.4]
    with_feat = [99.26, 99.55, 99.36]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - width / 2, without_feat, width, label="Without IP-frequency feature", color="#95a5a6")
    ax.bar(x + width / 2, with_feat, width, label="With IP-frequency feature", color="#2980b9")
    for i, (a, b) in enumerate(zip(without_feat, with_feat)):
        ax.annotate(f"{a:.1f}", xy=(x[i] - width / 2, a), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
        ax.annotate(f"{b:.1f}", xy=(x[i] + width / 2, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Figure 2. Contribution of the Causal IP Request-Frequency\nFeature (Section 6.2, seed 42)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure2_ip_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure3_baseline_comparison():
    # Section 6.5, mean +/- std across 5 seeds.
    rate_limit_mean = [0.0, 0.51]
    rate_limit_std = [0.0, 0.53]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(2)
    width = 0.35
    detector_vals = [99.42, 63.94]  # detector clean vs. detector under PGD (undefended), for direct comparability
    ax.bar(x - width / 2, detector_vals, width, label="AR-FEL-DoW detector (undefended)", color="#2980b9")
    ax.bar(x + width / 2, rate_limit_mean, width, yerr=rate_limit_std, capsize=5, label="Rate-limiting baseline", color="#7f8c8d")
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Under PGD attack"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Figure 3. Detector vs. Production Rate-Limiting Baseline\n(mean ± std, 5 seeds; Section 6.5)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for i, (a, b) in enumerate(zip(detector_vals, rate_limit_mean)):
        ax.annotate(f"{a:.1f}", xy=(x[i] - width / 2, a), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
        ax.annotate(f"{b:.1f}", xy=(x[i] + width / 2, b + rate_limit_std[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure3_baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure4_financial_gap():
    # Section 6.6, mean +/- std across 5 seeds, from results_multi_seed.json.
    conditions = ["Undefended", "Adversarially trained"]
    means = [7.1167, -0.1441]
    stds = [1.1256, 0.0249]
    colors = ["#c0392b", "#27ae60"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bars = ax.bar(conditions, means, yerr=stds, capsize=5, color=colors, width=0.5)
    ax.axhline(0, color="#2c3e50", linewidth=0.9)
    for b, m, s in zip(bars, means, stds):
        va = "bottom" if m >= 0 else "top"
        offset = 0.15 if m >= 0 else -0.15
        ax.annotate(f"\\${m:+.2f} ± \\${s:.2f}", xy=(b.get_x() + b.get_width() / 2, m + (s if m >= 0 else -s) + offset),
                    ha="center", va=va, fontsize=9.5)
    ax.set_ylabel("Financial Robustness Gap (USD, test-slice attack rows)")
    ax.set_title("Figure 4. Financial Robustness Gap: Undefended vs.\nAdversarially Trained Detector (mean ± std, 5 seeds)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure4_financial_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure5_results():
    # Mean +/- std across 5 seeds (random_state in {42..46}), from
    # scripts/run_multi_seed.py's results_multi_seed_summary.json.
    conditions = ["Clean", "PGD attack", "GAN attack"]
    undefended = [99.42, 63.94, 47.66]
    undefended_err = [0.05, 1.96, 4.51]
    defended = [99.42, 100.0, 86.65]
    defended_err = [0.05, 0.0, 2.18]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars1 = ax.bar(x - width / 2, undefended, width, yerr=undefended_err, capsize=4,
                    label="Undefended", color="#c0392b")
    bars2 = ax.bar(x + width / 2, defended, width, yerr=defended_err, capsize=4,
                    label="Adversarially Trained", color="#27ae60")

    ax.axhline(0.51, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Rate-limit baseline (~0.0-0.5%)")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Figure 5. Detector Accuracy Under Clean and Adversarial Conditions\n(mean ± std, 5 seeds)",
        fontsize=11.5, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 112)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bars, errs in ((bars1, undefended_err), (bars2, defended_err)):
        for b, err in zip(bars, errs):
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h + err), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure5_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure6_seed_stability():
    # Section 6.8, all 5 individual seeds, from results_multi_seed.json.
    seeds = [42, 43, 44, 45, 46]
    clean = [99.36, 99.43, 99.45, 99.49, 99.38]
    pgd_undef = [61.27, 62.52, 64.10, 66.95, 64.88]
    pgd_def = [100.0, 100.0, 100.0, 100.0, 100.0]
    gan_undef = [53.18, 52.69, 45.95, 44.62, 41.87]
    gan_def = [88.62, 88.74, 86.26, 82.75, 86.87]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(seeds, clean, marker="o", label="Clean accuracy", color="#2c3e50")
    ax.plot(seeds, pgd_undef, marker="s", label="PGD, undefended", color="#c0392b")
    ax.plot(seeds, pgd_def, marker="^", label="PGD, defended", color="#27ae60")
    ax.set_ylim(55, 102)
    ax.set_xlabel("Random seed")
    ax.set_xticks(seeds)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Gradient (PGD) attack", fontsize=10.5)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(linestyle=":", alpha=0.5)

    ax2 = axes[1]
    ax2.plot(seeds, clean, marker="o", label="Clean accuracy", color="#2c3e50")
    ax2.plot(seeds, gan_undef, marker="s", label="GAN, undefended", color="#c0392b")
    ax2.plot(seeds, gan_def, marker="^", label="GAN, defended", color="#27ae60")
    ax2.set_ylim(38, 102)
    ax2.set_xlabel("Random seed")
    ax2.set_xticks(seeds)
    ax2.set_title("Generative (GAN) attack", fontsize=10.5)
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(linestyle=":", alpha=0.5)

    fig.suptitle("Figure 6. Per-Seed Accuracy Across All 5 Independent Runs (Section 6.8)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure6_seed_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure7_catboost_isolation():
    # Section 6.9, isolated seed-42 substitution ablation. Percentage-scale
    # metrics only; the financial-gap delta ($6.345 vs $6.338) is a
    # different unit and is already reported in Table 8 directly above
    # this figure, so it is left out here rather than sharing an axis
    # where it would visually vanish against the ~99% bars.
    metrics = ["Clean\naccuracy", "PGD\nundefended", "GAN\nundefended"]
    catboost = [99.36, 61.27, 53.18]
    gbm = [99.37, 61.31, 53.50]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(x - width / 2, catboost, width, label="CatBoost (used in paper)", color="#8e44ad")
    ax.bar(x + width / 2, gbm, width, label="GradientBoostingClassifier", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Figure 7. CatBoost vs. GradientBoostingClassifier,\nIsolated Substitution Ablation (Section 6.9, seed 42)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for i, (a, b) in enumerate(zip(catboost, gbm)):
        ax.annotate(f"{a:g}", xy=(x[i] - width / 2, a), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
        ax.annotate(f"{b:g}", xy=(x[i] + width / 2, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure7_catboost_isolation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure1_architecture()
    figure2_ip_ablation()
    figure3_baseline_comparison()
    figure4_financial_gap()
    figure5_results()
    figure6_seed_stability()
    figure7_catboost_isolation()
    print("wrote figures 1-7")
