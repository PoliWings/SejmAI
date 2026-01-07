import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import json
import os

TITLE_FONT_SIZE = 30
LABEL_FONT_SIZE = 24

CATEGORY_LABELS = {
    "economy": "Economy",
    "customary": "Customary",
    "foreign_policy": "Foreign\nPolicy",
    "system": "System",
    "climate_policy": "Climate\nPolicy"
}

PROMPT_ORDER = [
    "basic",
    "neutral",
    "left-wing",
    "right-wing",
]

def create_grouped_chart(analysis_data, ax):
    system_prompt_title = analysis_data.get("system_prompt", "Chart")
    chart_data = analysis_data.get("data", [])

    if not chart_data:
        ax.set_title(f"No data for {system_prompt_title}")
        return

    df_flat = pd.json_normalize(chart_data).set_index("category")

    df = pd.DataFrame(index=df_flat.index)
    df[("neutral", "leftism")] = df_flat["neutral_model.left_wing"]
    df[("neutral", "rightism")] = df_flat["neutral_model.right_wing"]
    df[("left-wing", "leftism")] = df_flat["left_wing_model.left_wing"]
    df[("left-wing", "rightism")] = df_flat["left_wing_model.right_wing"]
    df[("right-wing", "leftism")] = df_flat["right_wing_model.left_wing"]
    df[("right-wing", "rightism")] = df_flat["right_wing_model.right_wing"]

    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Model", "Ratio"])

    categories = df.index
    models = df.columns.get_level_values("Model").unique()

    x = np.arange(len(categories)) * 1.6

    bar_width = 0.24
    gap = 0.04
    offset = bar_width + gap

    positions = {
        "neutral": x - offset,
        "left-wing": x,
        "right-wing": x + offset,
    }

    colors = {
        "neutral": {"leftism": "#648FFC", "rightism": "#254EAD"},
        "left-wing": {"leftism": "#DC4494", "rightism": "#8F225B"},
        "right-wing": {"leftism": "#44B49C", "rightism": "#1E695B"},
    }

    for model in models:
        leftism = df[(model, "leftism")]
        rightism = df[(model, "rightism")]

        ax.bar(
            positions[model],
            leftism,
            width=bar_width,
            color=colors[model]["leftism"],
        )

        ax.bar(
            positions[model],
            rightism,
            width=bar_width,
            bottom=leftism,
            color=colors[model]["rightism"],
        )

    ax.set_title(system_prompt_title, fontsize=TITLE_FONT_SIZE, pad=10)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="red", linestyle="--", linewidth=2)

    labels = [CATEGORY_LABELS.get(cat, cat) for cat in categories]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22)

    ax.tick_params(axis="y", labelsize=20, pad=2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)


if __name__ == "__main__":

    filepath = os.path.join("output", "accuracies.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, "r") as f:
        all_data = json.load(f)

    results = all_data.get("analysis_results", [])
    if len(results) < 4:
        raise ValueError("Need at least 4 analysis_results")

    fig, axes = plt.subplots(
        2, 2, figsize=(22, 15),
        gridspec_kw={"hspace": 0.3, "wspace": 0.1}
    )
    axes = axes.flatten()

    results_by_prompt = {
        r["system_prompt"].lower(): r
        for r in results
    }

    ordered_results = [results_by_prompt[p] for p in PROMPT_ORDER]

    for ax, analysis in zip(axes, ordered_results):
        create_grouped_chart(analysis, ax)

    fig.suptitle(
        "Average Categories Scores for System Prompts", 
        fontsize=40,
        y=1
    )

    fig.text(
        0.07, 0.5, "Percentage (%)",
        va="center", rotation="vertical",
        fontsize=36
    )

    fig.text(
        0.5, 0.02, "Category",
        ha="center",
        fontsize=36
    )

    model_handles = [
        Patch(color="#648FFC", label="Neutral"),
        Patch(color="#DC4494", label="Left-wing"),
        Patch(color="#44B49C", label="Right-wing"),
    ]

    ratio_handles = [
        Patch(color="#9ca3af", label="Leftism"),
        Patch(color="#4b5563", label="Rightism"),
    ]

    fig.legend(
        handles=model_handles,
        title="Model",
        loc="lower center",
        ncol=3,
        fontsize=LABEL_FONT_SIZE,
        title_fontsize=TITLE_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.32, -0.1),
    )

    fig.legend(
        handles=ratio_handles,
        title="Ratio",
        loc="lower center",
        ncol=2,
        fontsize=LABEL_FONT_SIZE,
        title_fontsize=TITLE_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.72, -0.1),
    )

    os.makedirs("plots", exist_ok=True)
    output_path = "plots/system_prompt_all.png"

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Combined chart saved as: {output_path}")
