import matplotlib.pyplot as plt
import os

labels = ["Politician", "LLM"]
values = [26, 7]

colors = ["#4E79A7", "#F28E2B"]

def make_autopct(labels):
    def my_autopct(pct):
        idx = make_autopct.idx
        s = f"{labels[idx]}\n{pct:.1f}%"
        make_autopct.idx += 1
        return s
    make_autopct.idx = 0
    return my_autopct

fig, ax = plt.subplots(figsize=(12, 12))
ax.pie(
    values,
    labels=None,
    autopct=make_autopct(labels),
    startangle=90,
    colors=colors,
    textprops=dict(color="white", fontsize=20, weight='bold')
)

ax.set_title("Answer Distribution – Question 11", fontsize=28, pad=10)

os.makedirs("charts", exist_ok=True)
output_file = "charts/question_11_pie_modern.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"Pie chart saved as: {output_file}")
