"""
src/visualize.py

Generates all thesis figures. Called from analyze_results.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_LABELS = {"full": "Full Evidence", "partial": "Partial Evidence", "none": "No Evidence"}
COLORS = {"full": "#2c7bb6", "partial": "#fdae61", "none": "#d7191c"}
CONDITIONS = ["full", "partial", "none"]


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def fig1_hallucination_rate(metrics_by_condition: dict, output_dir: str):
    """Bar chart: Hallucination Rate per Condition."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(CONDITIONS))
    rates = [metrics_by_condition[c]["hallucination_rate"] * 100 for c in CONDITIONS]
    bars = ax.bar(x, rates, color=[COLORS[c] for c in CONDITIONS], width=0.5, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("Hallucination Rate by Evidence Condition")
    ax.set_ylim(0, 105)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_hallucination_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig2_auroc_comparison(metrics_by_condition: dict, output_dir: str):
    """Grouped bar chart: AUROC per UE Method per Condition."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(CONDITIONS))
    width = 0.3

    vc_aurocs = [metrics_by_condition[c].get("auroc_verbalized_confidence") or 0 for c in CONDITIONS]
    sc_aurocs = [metrics_by_condition[c].get("auroc_selfcheck") or 0 for c in CONDITIONS]

    bars1 = ax.bar(x - width / 2, vc_aurocs, width, label="Verbalized Confidence", color="#4dac26", edgecolor="white")
    bars2 = ax.bar(x + width / 2, sc_aurocs, width, label="SelfCheckGPT", color="#b8e186", edgecolor="white")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC by UE Method and Evidence Condition")
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_auroc_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig3_calibration(results_by_condition: dict, output_dir: str, n_bins: int = 10):
    """Reliability diagram: one line per condition."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for cond in CONDITIONS:
        results = results_by_condition[cond]
        confidences = np.array([r.get("verbalized_confidence", 50) / 100.0 for r in results])
        accuracies = np.array([1 - r.get("is_hallucinated", 0) for r in results], dtype=float)

        bin_accs = []
        bin_confs_actual = []
        for i in range(n_bins):
            mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accs.append(accuracies[mask].mean())
                bin_confs_actual.append(bin_centers[i])

        ax.plot(bin_confs_actual, bin_accs, "o-", color=COLORS[cond],
                label=CONDITION_LABELS[cond], linewidth=1.5, markersize=5)

    ax.set_xlabel("Verbalized Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title("Reliability Diagram by Evidence Condition")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_calibration.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig4_confidence_distribution(results_by_condition: dict, output_dir: str):
    """Violin plot of verbalized confidence per condition."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    data = [
        [r.get("verbalized_confidence", 50) for r in results_by_condition[c]]
        for c in CONDITIONS
    ]
    parts = ax.violinplot(data, positions=range(len(CONDITIONS)), showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[CONDITIONS[i]])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Verbalized Confidence")
    ax.set_title("Distribution of Verbalized Confidence by Evidence Condition")
    ax.set_ylim(0, 105)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_confidence_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
