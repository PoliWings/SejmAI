import matplotlib.pyplot as plt
import json
import os

files = [("trainer_state_left.json", "Rightist Model"), ("trainer_state_right.json", "Leftist Model")]

os.makedirs("charts", exist_ok=True)

histories = []
for filename, _ in files:
    with open(filename) as f:
        data = json.load(f)
        histories.append(data["log_history"])

all_keys = set(histories[0][0].keys()) & set(histories[1][0].keys())
metrics = [key for key in all_keys if key != "step"]

for metric in metrics:
    plt.figure(figsize=(12, 4))
    
    for i, history in enumerate(histories):
        steps = [entry["step"] for entry in history]
        values = [entry.get(metric, None) for entry in history]
        label = files[i][1]
        plt.plot(steps, values, label=label)

    plt.title(f"{metric} comparison")
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = os.path.join("charts", f"{metric}_comparison.png")
    plt.savefig(filename)
    plt.close()
