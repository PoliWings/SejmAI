import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import json
import os

TITLE_FONT_SIZE = 40
LABEL_FONT_SIZE = 32
LEGEND_FONT_SIZE = 24

CATEGORY_LABELS = {
    "economy": "Economy",
    "customary": "Customary",
    "foreign_policy": "Foreign",
    "system": "System",
    "climate_policy": "Climate"
}


def create_grouped_chart(analysis_data):

    system_prompt_title = analysis_data.get("system_prompt", "Chart")
    chart_data = analysis_data.get("data", [])

    if not chart_data:
        print(f"No data found for system_prompt: {system_prompt_title}")
        return

    df_flat = pd.json_normalize(chart_data)
    df_flat = df_flat.set_index("category")

    df = pd.DataFrame(index=df_flat.index)
    df[("neutral", "leftism")] = df_flat["neutral_model.left_wing"]
    df[("neutral", "rightism")] = df_flat["neutral_model.right_wing"]
    df[("left-wing", "leftism")] = df_flat["left_wing_model.left_wing"]
    df[("left-wing", "rightism")] = df_flat["left_wing_model.right_wing"]
    df[("right-wing", "leftism")] = df_flat["right_wing_model.left_wing"]
    df[("right-wing", "rightism")] = df_flat["right_wing_model.right_wing"]

    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Model", "Ratio"])
    df.index.name = "Category"

    categories = df.index
    models = df.columns.get_level_values("Model").unique()

    n_categories = len(categories)

    x = np.arange(n_categories) * 1.5

    bar_width = 0.25
    intra_group_gap = 0.05

    bar_plus_gap = bar_width + intra_group_gap

    positions = {"neutral": x - bar_plus_gap, "left-wing": x, "right-wing": x + bar_plus_gap}

    colors = {
        "neutral": {"leftism": "#648FFC", "rightism": "#254EAD"},
        "left-wing": {"leftism": "#DC4494", "rightism": "#8F225B"},
        "right-wing": {"leftism": "#44B49C", "rightism": "#1E695B"},
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    color_handles = {
        "leftism": Patch(color="#9ca3af", label="Leftism"),
        "rightism": Patch(color="#4b5563", label="Rightism"),
    }

    model_labels = {"neutral": "Neutral", "left-wing": "Left-wing", "right-wing": "Right-wing"}
    model_handles = [
        Patch(color=colors["neutral"]["leftism"], label=model_labels["neutral"]),
        Patch(color=colors["left-wing"]["leftism"], label=model_labels["left-wing"]),
        Patch(color=colors["right-wing"]["leftism"], label=model_labels["right-wing"]),
    ]

    for model in models:
        leftism_data = df[(model, "leftism")]
        rightism_data = df[(model, "rightism")]

        pos = positions[model]

        ax.bar(pos, leftism_data, width=bar_width, color=colors[model]["leftism"])

        ax.bar(pos, rightism_data, width=bar_width, bottom=leftism_data, color=colors[model]["rightism"])

    ax.set_title(f"System Prompt: {system_prompt_title}", fontsize=TITLE_FONT_SIZE, pad=10)
    ax.set_ylabel("Percentage (%)", fontsize=LABEL_FONT_SIZE)

    labels = [CATEGORY_LABELS.get(cat, cat) for cat in categories]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=LEGEND_FONT_SIZE)

    ax.tick_params(axis="y", labelsize=LEGEND_FONT_SIZE)

    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    ax.axhline(y=50, color="red", linestyle="--", linewidth=2)

    # leg1 = ax.legend(
    #     handles=model_handles,
    #     title="Model",
    #     loc="upper center",
    #     bbox_to_anchor=(0.3, -0.1),
    #     ncol=3,
    #     frameon=False,
    #     title_fontsize=LEGEND_FONT_SIZE,
    #     fontsize=LEGEND_FONT_SIZE,
    # )

    # leg2 = ax.legend(
    #     handles=color_handles.values(),
    #     title="Ratio",
    #     loc="upper center",
    #     bbox_to_anchor=(0.8, -0.1),
    #     ncol=2,
    #     frameon=False,
    #     title_fontsize=LEGEND_FONT_SIZE,
    #     fontsize=LEGEND_FONT_SIZE,
    # )

    # ax.add_artist(leg1)
    # ax.add_artist(leg2)

    # extra_artists = (leg1, leg2)

    fig.subplots_adjust(bottom=0.2)

    safe_title = system_prompt_title.replace(" ", "_").lower()

    os.makedirs("plots", exist_ok=True)

    output_filename = f"plots/system_prompt_{safe_title}.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')

    print(f"Chart successfully saved as: {output_filename}")
    plt.close(fig)


if __name__ == "__main__":

    filepath = os.path.join("output", "accuracies.json")

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
    else:
        with open(filepath, "r") as f:
            all_data = json.load(f)

        results = all_data.get("analysis_results", [])

        if not results:
            print("Error: No 'analysis_results' found in JSON file.")

        for analysis in results:
            create_grouped_chart(analysis)
