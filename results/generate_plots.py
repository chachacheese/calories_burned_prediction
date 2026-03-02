import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Config (your current structure)
# ---------------------------
CSV_PATH = "results/model_comparison.csv"
OUT_DIR = "results/plots"

# Phase boundaries
PHASE_RANGES = {
    "Tree Models": (1, 4),
    "Polynomial Regression": (5, 10),
    "Gender Split": (11, 12),
}

# ---------------------------
# Green palette (Dark + Pastel)
# ---------------------------
GREEN_DARK = "#1b5e20"      # deep forest
GREEN_MAIN = "#2e7d32"      # strong green
GREEN_ACCENT = "#66bb6a"    # fresh green
GREEN_PASTEL = "#a5d6a7"    # pastel mint
GREEN_TEXT = "#1b5e20"      # text color

# Phase highlight backgrounds (very light)
SHADE_TREE = "#e8f5e9"      # soft green bg
SHADE_POLY = "#f1f8e9"      # soft lime bg
SHADE_GENDER = "#f9fbe7"    # very light yellow-green bg

# Neutral line / edge
EDGE = "#2c3e50"


# ---------------------------
# Utils
# ---------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Stage", "Model", "RMSE", "Improvement_pct", "Phase", "Key_Technique"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"model_comparison.csv is missing columns: {missing}")
    return df.sort_values("Stage").reset_index(drop=True)

def stage_to_phase(stage: int) -> str:
    for name, (s, e) in PHASE_RANGES.items():
        if s <= stage <= e:
            return name
    return "Unknown"

def fmt_stage_label(model_name: str) -> str:
    """
    Wrap labels similarly to your screenshots.
    """
    replacements = {
        "LGBM (baseline)": "Baseline\nLGBM",
        "Log transform": "Log\ntransform",
        "Feature Engineering": "Feature\nEngineering",
        "Optuna Tuning": "Optuna\nTuning",
        "Poly(deg3) Ridge": "Poly(deg3)\nRidge",
        "Ridge + KFold": "+ Ridge\n+ KFold",
        "Target Transform Stacking": "+ Target\nStacking",
        "Weight Optimization": "+ Weight\nOptimize",
        "Feature Pruning": "+ Feature\nPruning",
        "Round (integer)": "+ Round\n(Integer)",
        "Gender Split": "+ Gender\nSplit",
        "Grid Search (Final)": "+ Grid\nSearch",
    }
    return replacements.get(model_name, model_name)


# ---------------------------
# Plot 1: Stage-by-stage improvement bar chart
# ---------------------------
def plot_improvement_by_stage(df: pd.DataFrame, out_path: str) -> None:
    x = np.arange(len(df))
    improvements = df["Improvement_pct"].astype(float).values

    # Bar colors by phase: pastel -> dark -> accent
    colors = []
    for st in df["Stage"].astype(int).values:
        ph = stage_to_phase(int(st))
        if ph == "Tree Models":
            colors.append(GREEN_PASTEL)
        elif ph == "Polynomial Regression":
            colors.append(GREEN_MAIN)
        elif ph == "Gender Split":
            colors.append(GREEN_ACCENT)
        else:
            colors.append("#cccccc")

    plt.figure(figsize=(22, 7))
    bars = plt.bar(x, improvements, color=colors, edgecolor=EDGE, linewidth=1.0)

    # Value labels
    for rect, val in zip(bars, improvements):
        if np.isnan(val):
            continue
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.6,
            f"{int(round(val))}%",
            ha="center",
            va="bottom",
            fontsize=12,
            color=GREEN_TEXT,
            fontweight="bold",
        )

    labels = [fmt_stage_label(m) for m in df["Model"].tolist()]
    plt.xticks(x, labels, fontsize=11)
    plt.ylabel("Improvement (%)", fontsize=16, fontweight="bold", color=GREEN_TEXT)
    plt.title("Stage-by-Stage RMSE Improvement — Key Turning Points",
              fontsize=22, fontweight="bold", color=GREEN_TEXT)

    # Phase separators (after stage 4 and 10)
    idx_stage4 = df.index[df["Stage"] == 4].tolist()
    idx_stage10 = df.index[df["Stage"] == 10].tolist()
    for idx in idx_stage4:
        plt.axvline(idx + 0.5, color=GREEN_DARK, linestyle="--", linewidth=1.5, alpha=0.35)
    for idx in idx_stage10:
        plt.axvline(idx + 0.5, color=GREEN_DARK, linestyle="--", linewidth=1.5, alpha=0.35)

    ymax = max(45, np.nanmax(improvements) + 5)
    plt.ylim(0, ymax)

    def center_idx(stage_start: int, stage_end: int) -> float:
        i0 = df.index[df["Stage"] == stage_start][0]
        i1 = df.index[df["Stage"] == stage_end][0]
        return (i0 + i1) / 2

    plt.text(center_idx(1, 4), ymax * 0.92, "Phase 1\nTree Models",
             ha="center", va="top", fontsize=13, color=GREEN_DARK, fontweight="bold")
    plt.text(center_idx(5, 10), ymax * 0.92, "Phase 2\nPolynomial",
             ha="center", va="top", fontsize=13, color=GREEN_MAIN, fontweight="bold")
    plt.text(center_idx(11, 12), ymax * 0.92, "Phase 3\nGender",
             ha="center", va="top", fontsize=13, color=GREEN_ACCENT, fontweight="bold")

    plt.grid(axis="y", linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Plot 2: RMSE reduction by phase + contribution pie
# ---------------------------
def plot_phase_summary(df: pd.DataFrame, out_path: str) -> None:
    start_rmse = float(df.loc[df["Stage"] == 1, "RMSE"].iloc[0])

    phase_end_rmse = {}
    for ph_name, (s, e) in PHASE_RANGES.items():
        phase_end_rmse[ph_name] = float(df.loc[df["Stage"] == e, "RMSE"].iloc[0])

    end_tree = phase_end_rmse["Tree Models"]
    end_poly = phase_end_rmse["Polynomial Regression"]
    end_gender = phase_end_rmse["Gender Split"]

    red_tree = end_tree - start_rmse
    red_poly = end_poly - end_tree
    red_gender = end_gender - end_poly
    total_red = end_gender - start_rmse  # negative

    contrib_vals = np.array([abs(red_tree), abs(red_poly), abs(red_gender)], dtype=float)
    contrib_labels = [
        f"Phase 1: Tree\n({red_tree:+.3f})",
        f"Phase 2: Poly\n({red_poly:+.3f})",
        f"Phase 3: Gender\n({red_gender:+.3f})",
    ]
    contrib_colors = [GREEN_PASTEL, GREEN_MAIN, GREEN_ACCENT]

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])

    # Left: blocks
    ax1 = fig.add_subplot(gs[0, 0])

    phases = [
        ("Phase 1\nTree Models\n(Stage 1-4)", start_rmse, end_tree, GREEN_PASTEL),
        ("Phase 2\nPolynomial\n(Stage 5-10)", end_tree, end_poly, GREEN_MAIN),
        ("Phase 3\nGender Split\n(Stage 11-12)", end_poly, end_gender, GREEN_ACCENT),
    ]

    x_positions = np.arange(len(phases))
    width = 0.7

    for i, (label, y0, y1, c) in enumerate(phases):
        top = y0
        bottom = y1
        height = top - bottom
        ax1.bar(i, height, bottom=bottom, width=width, color=c, alpha=0.85,
                edgecolor=EDGE, linewidth=1.5)

        reduction = y1 - y0  # negative
        pct = (abs(reduction) / abs(total_red)) * 100 if total_red != 0 else 0

        ax1.text(i, bottom + height / 2,
                 f"{reduction:+.3f}\n({pct:.0f}%)",
                 ha="center", va="center",
                 fontsize=14, fontweight="bold", color=GREEN_TEXT)

        ax1.text(i, y0 + 0.03, f"{y0:.3f}", ha="center", va="bottom", fontsize=10, color="#607d6a")
        ax1.text(i, y1 - 0.03, f"{y1:.3f}", ha="center", va="top", fontsize=10, color="#607d6a")

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([p[0] for p in phases], fontsize=11)
    ax1.set_ylabel("RMSE", fontsize=16, fontweight="bold", color=GREEN_TEXT)
    ax1.set_title("RMSE Reduction by Phase", fontsize=20, fontweight="bold", color=GREEN_TEXT)
    ax1.set_ylim(0, max(start_rmse + 0.25, 2.5))
    ax1.grid(axis="y", linestyle="-", alpha=0.2)

    # Right: pie
    ax2 = fig.add_subplot(gs[0, 1])
    wedges, texts, autotexts = ax2.pie(
        contrib_vals,
        labels=contrib_labels,
        colors=contrib_colors,
        startangle=90,
        autopct=lambda p: f"{p:.1f}%",
        textprops={"fontsize": 12, "color": GREEN_TEXT},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        explode=(0.02, 0.04, 0.06),
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color(GREEN_TEXT)
        at.set_fontsize(14)

    ax2.set_title(f"Contribution to Total RMSE Reduction ({start_rmse:.2f} -> {end_gender:.2f})",
                  fontsize=18, fontweight="bold", color=GREEN_TEXT)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Plot 3: RMSE progression line plot with phase shading + annotations
# ---------------------------
def plot_rmse_progression(df: pd.DataFrame, out_path: str) -> None:
    stages = df["Stage"].astype(int).values
    rmse = df["RMSE"].astype(float).values
    labels = [fmt_stage_label(m) for m in df["Model"].tolist()]

    start_rmse = rmse[0]
    end_rmse = rmse[-1]
    improvement_pct_total = (1 - end_rmse / start_rmse) * 100 if start_rmse != 0 else 0

    plt.figure(figsize=(22, 7))
    ax = plt.gca()
    x = np.arange(len(df))

    # Phase shading
    def shade(stage_start: int, stage_end: int, color: str):
        i0 = df.index[df["Stage"] == stage_start][0]
        i1 = df.index[df["Stage"] == stage_end][0]
        ax.axvspan(i0 - 0.5, i1 + 0.5, color=color, alpha=0.65)

    shade(1, 4, SHADE_TREE)
    shade(5, 10, SHADE_POLY)
    shade(11, 12, SHADE_GENDER)

    # Line
    ax.plot(x, rmse, color=EDGE, linewidth=3, zorder=2)

    # Points colored by phase (greens only)
    for i, st in enumerate(stages):
        ph = stage_to_phase(int(st))
        if ph == "Tree Models":
            c = GREEN_PASTEL
        elif ph == "Polynomial Regression":
            c = GREEN_MAIN
        elif ph == "Gender Split":
            c = GREEN_ACCENT
        else:
            c = "#cccccc"
        ax.scatter([i], [rmse[i]], s=180, color=c, edgecolor=EDGE, linewidth=2, zorder=3)

    # Value labels
    for i, val in enumerate(rmse):
        ax.text(i, val + 0.05, f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=11, color=GREEN_TEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("RMSE", fontsize=16, fontweight="bold", color=GREEN_TEXT)
    ax.set_title(f"Calorie Prediction — RMSE Progression ({start_rmse:.2f} → {end_rmse:.2f}, -{improvement_pct_total:.0f}%)",
                 fontsize=22, fontweight="bold", color=GREEN_TEXT)
    ax.set_ylim(0, max(2.5, start_rmse + 0.25))
    ax.grid(axis="y", linestyle="-", alpha=0.2)

    # Legend for phase shading
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SHADE_TREE, alpha=0.65, label="Phase 1: Tree Models"),
        plt.Rectangle((0, 0), 1, 1, color=SHADE_POLY, alpha=0.65, label="Phase 2: Polynomial Regression"),
        plt.Rectangle((0, 0), 1, 1, color=SHADE_GENDER, alpha=0.65, label="Phase 3: Gender Split"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

    # Annotations (green)
    idx_poly = df.index[df["Stage"] == 5][0]
    ax.annotate(
        "Model Switch!\nTree -> Poly",
        xy=(idx_poly, rmse[idx_poly]),
        xytext=(idx_poly, rmse[idx_poly] + 0.65),
        arrowprops=dict(arrowstyle="->", color=GREEN_DARK, linewidth=2),
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=GREEN_DARK,
    )

    idx_gender = df.index[df["Stage"] == 11][0]
    ax.annotate(
        "Gender Split\nBreakthrough!",
        xy=(idx_gender, rmse[idx_gender]),
        xytext=(idx_gender, rmse[idx_gender] + 0.35),
        arrowprops=dict(arrowstyle="->", color=GREEN_DARK, linewidth=2),
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=GREEN_DARK,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    ensure_outdir(OUT_DIR)
    df = load_df(CSV_PATH)

    plot_improvement_by_stage(df, os.path.join(OUT_DIR, "improvement_by_stage.png"))
    plot_phase_summary(df, os.path.join(OUT_DIR, "phase_summary.png"))
    plot_rmse_progression(df, os.path.join(OUT_DIR, "rmse_progression.png"))

    print("[OK] Saved plots to:", OUT_DIR)

if __name__ == "__main__":
    main()
