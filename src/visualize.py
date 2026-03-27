"""
src/visualize.py

Generates all thesis figures for 2x3 factorial design.
Called from analyze_results.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_LABELS = {"full": "Full", "partial": "Partial", "none": "None"}
PROMPT_COLORS = {"constrained": "#2c7bb6", "unconstrained": "#d7191c"}
CONDITION_COLORS = {"full": "#2c7bb6", "partial": "#fdae61", "none": "#d7191c"}
CONDITIONS = ["full", "partial", "none"]
PROMPT_TYPES = ["constrained", "unconstrained"]


def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def fig1_hallucination_rate(metrics_by_cell: dict, output_dir: str):
    """Grouped bar chart: Hallucination Rate (EM-based) over 2x3 design."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(CONDITIONS))
    width = 0.35

    for i, pt in enumerate(PROMPT_TYPES):
        rates = []
        ci_lo = []
        ci_hi = []
        for cond in CONDITIONS:
            key = (pt, cond)
            m = metrics_by_cell.get(key, {})
            hal = m.get("hallucination_rate_em", 0) * 100
            lo = m.get("hallucination_rate_em_ci_lower", 0) * 100
            hi = m.get("hallucination_rate_em_ci_upper", 0) * 100
            rates.append(hal)
            ci_lo.append(hal - lo)
            ci_hi.append(hi - hal)

        offset = (i - 0.5) * width
        yerr = [ci_lo, ci_hi]
        bars = ax.bar(x + offset, rates, width, label=pt.capitalize(),
                      color=PROMPT_COLORS[pt], edgecolor="white",
                      yerr=yerr, capsize=4, error_kw={"linewidth": 1})
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] + " Evidence" for c in CONDITIONS])
    ax.set_ylabel("Hallucination Rate (%) [EM-based]")
    ax.set_title("Hallucination Rate by Evidence Condition and Prompt Type")
    ax.set_ylim(0, 115)
    ax.legend(title="Prompt Type")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_hallucination_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig2_auroc_comparison(metrics_by_cell: dict, output_dir: str):
    """Grouped bar chart: AUROC per UE method (Unconstrained only, EM label)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(CONDITIONS))
    width = 0.25

    ue_methods = [
        ("auroc_vc_em", "Verbal. Confidence", "#4dac26"),
        ("auroc_sc_em", "SelfCheckGPT", "#b8e186"),
        ("auroc_judge_em", "LLM-as-Judge", "#f4a582"),
    ]

    for i, (field, label, color) in enumerate(ue_methods):
        aurocs = []
        for cond in CONDITIONS:
            key = ("unconstrained", cond)
            m = metrics_by_cell.get(key, {})
            val = m.get(field)
            aurocs.append(val if val is not None else 0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, aurocs, width, label=label, color=color, edgecolor="white")
        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] + " Evidence" for c in CONDITIONS])
    ax.set_ylabel("AUROC (vs. EM Label)")
    ax.set_title("AUROC by UE Method — Unconstrained Prompt")
    ax.set_ylim(0, 1.1)
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_auroc_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig3_calibration(grouped: dict, output_dir: str, n_bins: int = 10):
    """Reliability diagram: Constrained vs Unconstrained subplots, one line per condition."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ax, pt in zip(axes, PROMPT_TYPES):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

        for cond in CONDITIONS:
            key = (pt, cond)
            results = grouped.get(key, [])
            if not results:
                continue

            confidences = np.array([r.get("verbalized_confidence", 50) / 100.0 for r in results])
            # Use EM-based accuracy
            accuracies = np.array([1 - r.get("is_hallucinated_em", 0) for r in results], dtype=float)

            bin_accs = []
            bin_confs_actual = []
            for i in range(n_bins):
                mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_accs.append(accuracies[mask].mean())
                    bin_confs_actual.append(bin_centers[i])

            ax.plot(bin_confs_actual, bin_accs, "o-", color=CONDITION_COLORS[cond],
                    label=CONDITION_LABELS[cond] + " Evidence", linewidth=1.5, markersize=5)

        ax.set_xlabel("Verbalized Confidence")
        ax.set_ylabel("Actual Accuracy (EM)")
        ax.set_title(f"Reliability Diagram — {pt.capitalize()} Prompt")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_calibration.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig4_confidence_distribution(grouped: dict, output_dir: str):
    """Violin plot of confidence distribution over 2x3."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, pt in zip(axes, PROMPT_TYPES):
        data = []
        labels = []
        for cond in CONDITIONS:
            key = (pt, cond)
            results = grouped.get(key, [])
            confs = [r.get("verbalized_confidence", 50) for r in results]
            data.append(confs)
            labels.append(CONDITION_LABELS[cond] + "\nEvidence")

        if any(data):
            parts = ax.violinplot(data, positions=range(len(CONDITIONS)), showmedians=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(list(CONDITION_COLORS.values())[i])
                pc.set_alpha(0.7)

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Verbalized Confidence")
        ax.set_title(f"Confidence Distribution — {pt.capitalize()} Prompt")
        ax.set_ylim(0, 105)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_confidence_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
