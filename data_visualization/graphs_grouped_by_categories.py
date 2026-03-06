import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os
import re

PROMPT_ORDER = ["BASIC", "LEFT", "NEUTRAL", "RIGHT"]
MODEL_ORDER = ["left", "neutral", "right"]

COLORS_MODEL = {
    "neutral": {"score": "#1C3D89", "rest": "#648FFC"},
    "left": {"score": "#6E1847", "rest": "#DC4494"},
    "right": {"score": "#154E45", "rest": "#44B49C"},
}

LEGEND_MODELS_AVG = {
    "neutral": Patch(color=COLORS_MODEL["neutral"]["rest"], label="Neutral model"),
    "left": Patch(color=COLORS_MODEL["left"]["rest"], label="Left-wing model"),
    "right": Patch(color=COLORS_MODEL["right"]["rest"], label="Right-wing model"),
}

LEGEND_RATIO = {
    "score": Patch(color="#2f3a45", label="Rightism"),
    "rest": Patch(color="#7b828a", label="Leftism"),
}

TITLE_FONT_SIZE = 40
LABEL_FONT_SIZE = 32
LEGEND_FONT_SIZE = 24


def parse_name(filename):
    m = re.match(r"prompt_(.+)_model_(.+)\.json", filename)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).lower()


def plot_categories_averages(data, out_dir="plots/averages"):
    os.makedirs(out_dir, exist_ok=True)

    categories = sorted({cat for file in data.values() for cat in file})

    for category in categories:

        values = {p: {m: None for m in MODEL_ORDER} for p in PROMPT_ORDER}

        for filename, content in data.items():
            prompt_type, model_bias = parse_name(filename)
            if not prompt_type:
                continue
            if category not in content:
                continue

            avg = content[category]["average"]

            if prompt_type in values and model_bias in values[prompt_type]:
                values[prompt_type][model_bias] = avg

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(PROMPT_ORDER)) * 1.2
        bw = 0.2
        gap = 0.05

        offsets = {
            "left": -bw - gap,
            "neutral": 0,
            "right": bw + gap,
        }

        for model in MODEL_ORDER:
            model_x = x + offsets[model]

            leftism_vals = []
            rightism_vals = []

            for prompt in PROMPT_ORDER:
                v = values[prompt][model]
                if v is None:
                    r, l = 0, 0
                else:
                    r = v
                    l = 100 - v

                leftism_vals.append(l)
                rightism_vals.append(r)

            ax.bar(model_x, leftism_vals, width=bw, color=COLORS_MODEL[model]["rest"])
            ax.bar(model_x, rightism_vals, width=bw, bottom=leftism_vals, color=COLORS_MODEL[model]["score"], alpha=0.8)

        ax.set_title(f"Average Score: {category}", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel("Percentage (%)", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("System Prompt Type", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(PROMPT_ORDER, fontsize=LEGEND_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=LEGEND_FONT_SIZE)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.axhline(50, color="red", linestyle="--")

        # leg1 = ax.legend(
        #     handles=list(LEGEND_MODELS_AVG.values()),
        #     title="Model type",
        #     loc="upper center",
        #     ncol=3,
        #     bbox_to_anchor=(0.3, -0.1),
        #     frameon=False,
        #     fontsize=LEGEND_FONT_SIZE,
        #     title_fontsize = LEGEND_FONT_SIZE,
        # )
        # leg2 = ax.legend(
        #     handles=list(LEGEND_RATIO.values()),
        #     title="Ratio",
        #     loc="upper center",
        #     ncol=2,
        #     bbox_to_anchor=(0.85, -0.1),
        #     frameon=False,
        #     fontsize=LEGEND_FONT_SIZE,
        #     title_fontsize = LEGEND_FONT_SIZE,
        # )
        # ax.add_artist(leg1)
        # ax.add_artist(leg2)

        # extra_artists = (leg1, leg2)

        fig.subplots_adjust(bottom=0.2)
        plt.savefig(f"{out_dir}/{category}.png", dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved average plot: {out_dir}/{category}.png")


def plot_general_average(score_data, out_file="plots/averages/general.png"):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    prompt_order = PROMPT_ORDER
    models = ["neutral", "left", "right"]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(prompt_order)) * 1.2
    bw = 0.2
    gap = 0.05
    offsets = {"left": -bw - gap, "neutral": 0, "right": bw + gap}

    for i, model in enumerate(models):
        model_x = x + offsets[model]

        leftism_values = score_data[:, i]
        rightism_values = 100 - leftism_values

        ax.bar(model_x, leftism_values, width=bw, color=COLORS_MODEL[model]["rest"])
        ax.bar(model_x, rightism_values, width=bw, bottom=leftism_values, color=COLORS_MODEL[model]["score"], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(prompt_order, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", fontsize=24)
    ax.set_xlabel("System Prompt Type", fontsize=24)
    ax.set_title("Overall Average Score", fontsize=TITLE_FONT_SIZE)
    ax.axhline(50, color="red", linestyle="--")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    leg1 = ax.legend(
        handles=list(LEGEND_MODELS_AVG.values()),
        title="Model type",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.3, -0.1),
        frameon=False,
        fontsize=14,
        title_fontsize = 14,
    )
    leg2 = ax.legend(
        handles=list(LEGEND_RATIO.values()),
        title="Ratio",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.85, -0.1),
        frameon=False,
        fontsize=14,
        title_fontsize = 14,
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    extra_artists = (leg1, leg2)


    fig.subplots_adjust(bottom=0.2)
    plt.savefig(out_file, dpi=200, bbox_inches='tight', bbox_extra_artists=extra_artists)
    plt.close()

    print(f"Saved overall average: {out_file}")


def plot_categories_std_dev(data, out_dir="plots/std_dev"):
    os.makedirs(out_dir, exist_ok=True)

    categories = sorted({category for file_data in data.values() for category in file_data})

    for category in categories:

        values = {p: {m: None for m in MODEL_ORDER} for p in PROMPT_ORDER}

        for filename, content in data.items():
            ptype, mbias = parse_name(filename)
            if not ptype:
                continue

            if category not in content:
                continue

            sd = content[category].get("std_dev")

            values[ptype][mbias] = sd

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(PROMPT_ORDER)) * 1.2
        bw = 0.2
        gap = 0.05
        offsets = {"left": -bw - gap, "neutral": 0, "right": bw + gap}

        for model in MODEL_ORDER:
            model_x = x + offsets[model]
            vals = [values[p][model] if values[p][model] is not None else 0 for p in PROMPT_ORDER]

            ax.bar(model_x, vals, width=bw, color=COLORS_MODEL[model]["rest"], alpha=0.9)

        ax.set_title(f"Standard Deviation: {category}", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel("Standard Deviation", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("System Prompt Type", fontsize=LABEL_FONT_SIZE)
        ax.set_ylim(0, 4)
        ax.set_xticks(x)
        ax.set_xticklabels(PROMPT_ORDER, fontsize=LEGEND_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=LEGEND_FONT_SIZE)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        ax.legend(
            handles=[Patch(color=COLORS_MODEL[m]["rest"], label=m.capitalize() + " model") for m in MODEL_ORDER],
            title="Model type",
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.1),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize = LEGEND_FONT_SIZE,
        )

        fig.subplots_adjust(bottom=0.2)
        plt.savefig(f"{out_dir}/{category}_std_dev.png", dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved std dev plot: {out_dir}/{category}_std_dev.png")


def plot_general_std_dev(std_data, out_file="plots/std_dev/general_std_dev.png"):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(PROMPT_ORDER)) * 1.2
    bw = 0.2
    gap = 0.05
    offsets = {"left": -bw - gap, "neutral": 0, "right": bw + gap}

    for i, model in enumerate(MODEL_ORDER):
        model_x = x + offsets[model]
        vals = std_data[:, i]

        ax.bar(model_x, vals, width=bw, color=COLORS_MODEL[model]["rest"], alpha=0.9)

    ax.set_title("Overall Standard Deviation", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("Standard Deviation", fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel("System Prompt Type", fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(PROMPT_ORDER, fontsize=LEGEND_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=LEGEND_FONT_SIZE)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    ax.legend(
        handles=[Patch(color=COLORS_MODEL[m]["rest"], label=m.capitalize() + " model") for m in MODEL_ORDER],
        title="Model type",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize = LEGEND_FONT_SIZE,
    )

    fig.subplots_adjust(bottom=0.2)
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved overall std dev: {out_file}")


def plot_all():
    with open("categories_statistics.json", "r", encoding="utf-8") as f:
        category_data = json.load(f)

    plot_categories_averages(category_data)

    general_data_averages = np.array(
        [[72.42, 82.85, 54.18], [88.82, 88.59, 81.41], [59.03, 71.68, 55.18], [18.18, 27.38, 21.38]]
    )
    plot_general_average(general_data_averages)

    plot_categories_std_dev(category_data)

    general_data_std_dev = np.array(
        [
            [0.47, 0.70, 0.97],
            [0.30, 0.32, 0.87],
            [0.36, 0.38, 0.49],
            [0.41, 0.62, 0.33],
        ]
    )
    plot_general_std_dev(general_data_std_dev)


if __name__ == "__main__":
    plot_all()
