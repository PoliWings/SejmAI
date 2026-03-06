import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

TITLE_FONT_SIZE = 28
LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18

question_numbers = list(range(1, 13))
percent_correct = [
    0.757575758, 0.818181818, 0.818181818, 0.727272727,
    0.424242424, 0.636363636, 0.818181818, 0.878787879,
    0.787878788, 0.393939394, 0.212121212, 0.666666667
]

correct_answer_by = [
    "Politician", "LLM", "Politician", "Politician", "LLM", "LLM",
    "Politician", "LLM", "Politician", "Politician", "LLM", "LLM"
]

color_map = {"Politician": "#4E79A7", "LLM": "#F28E2B"}
colors = [color_map[x] for x in correct_answer_by]

fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.bar(
    question_numbers,
    [p * 100 for p in percent_correct],
    color=colors,
    width=0.6
)

ax.set_title("Percentage of Correct Answers by Question", fontsize=TITLE_FONT_SIZE, pad=10)
ax.set_ylabel("Percentage (%)", fontsize=LABEL_FONT_SIZE)
ax.set_xlabel("Question Number", fontsize=LABEL_FONT_SIZE)

ax.set_xticks(question_numbers)
ax.set_xticklabels([str(q) for q in question_numbers], fontsize=LEGEND_FONT_SIZE)
ax.tick_params(axis="y", labelsize=LEGEND_FONT_SIZE)

ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)

legend_elements = [
    Patch(facecolor="#4E79A7", label="Politician"),
    Patch(facecolor="#F28E2B", label="LLM")
]
ax.legend(
    handles=legend_elements,
    title="Correct Answer",
    title_fontsize=LEGEND_FONT_SIZE,
    fontsize=LEGEND_FONT_SIZE,
    loc="upper right",
    frameon=False
)

os.makedirs("charts", exist_ok=True)
output_file = "charts/percentage_correct_colored.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"Bar chart saved as: {output_file}")
