import matplotlib.pyplot as plt
from matplotlib.patches import Patch

methods = [
    "Aether-V2",
    "Full-context",
    "A-Mem",
    "LangMem",
    "Zep",
    "OpenAI",
    "Mem0",
    "Mem0$^g$",
]

overall_j = [95.27, 72.90, 48.38, 58.10, 65.99, 52.90, 66.88, 68.44]

AETHER_COLOR = "#4a90d9"
OTHER_COLOR = "#f6b26b"
colors = [AETHER_COLOR if m == "Aether-V2" else OTHER_COLOR for m in methods]

plt.figure(figsize=(12, 7))
bars = plt.bar(methods, overall_j, color=colors, edgecolor="black", linewidth=0.8)

plt.title("Overall LLM-as-a-Judge (J) Score Comparison", fontsize=16, pad=14)
plt.ylabel("Overall J (%)", fontsize=12)
plt.xlabel("Method", fontsize=12)
plt.ylim(0, 105)
plt.grid(axis="y", linestyle="--", alpha=0.35)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, value in zip(bars, overall_j):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.2,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xticks(rotation=15, ha="right", fontsize=11)
plt.yticks(fontsize=11)

legend_handles = [
    Patch(facecolor=AETHER_COLOR, edgecolor="black", linewidth=0.8, label="Gemini-2.5-Flash"),
    Patch(facecolor=OTHER_COLOR, edgecolor="black", linewidth=0.8, label="GPT-4o-mini"),
]
ax.legend(
    handles=legend_handles,
    title="Model used",
    loc="upper right",
    frameon=True,
    fontsize=11,
    title_fontsize=12,
)

plt.tight_layout()
output_path = "./overall_j_bar_chart_orange_no_note.png"
plt.savefig(output_path, dpi=240, bbox_inches="tight")
plt.show()

print(output_path)