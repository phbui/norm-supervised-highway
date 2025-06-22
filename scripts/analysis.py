import os
import re
import ast
from collections import defaultdict
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})

RESULTS_DIR = "../results"
OUTPUT_DIR = "../results/plots"
OUTPUT_FILE = "comparison_grouped.png"
LEGEND_FILE = "legend_grouped.png"

PREFIX_LABELS = {
    "2_": "2L5V Model",
    "4_": "4L20V Model"
}
SCENARIO_LABELS = {
    "in_2_normalized": "2L5V Scenario",
    "in_4_normalized": "4L20V  Scenario",
    "in_tailgating_normalized": "Adversarial Scenario"
}

def list_files(directory, extension=".txt"):
    return sorted(f for f in os.listdir(directory) if f.endswith(extension))

def parse_results(path):
    data = {}
    mode = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line in ("WITH SUPERVISOR", "WITHOUT SUPERVISOR"):
                mode = line
                data[mode] = {}
            elif mode:
                # if line.startswith("Average collisions:"):
                #     m = re.search(r"Average collisions:\s*([\d\.]+)", line)
                #     if m: data[mode]["Collisions"] = float(m.group(1))
                if mode == "WITHOUT SUPERVISOR" and "Average unavoided violatoins by type:" in line:
                    d = ast.literal_eval(line.split(":",1)[1].strip())
                    for k, v in d.items(): data[mode][k] = float(v)
                elif mode == "WITH SUPERVISOR" and "Average unavoided violatoins by type:" in line:
                    d = ast.literal_eval(line.split(":",1)[1].strip())
                    for k, v in d.items(): data[mode][k] = float(v)
    return data

def analyze():
    files = list_files(RESULTS_DIR)
    if not files:
        print(f"No .txt files found in '{RESULTS_DIR}/'.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prefixes  = sorted({fn.split('_')[0] + '_' for fn in files})
    scenarios = sorted({fn.split('_',1)[1].rsplit('.txt',1)[0] for fn in files})
    modes     = ["WITHOUT SUPERVISOR", "WITH SUPERVISOR"]

    vals = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for fn in files:
        pf = fn.split('_',1)[0] + '_'
        sc = fn.split('_',1)[1].rsplit('.txt',1)[0]
        for mode, metrics in parse_results(os.path.join(RESULTS_DIR, fn)).items():
            for metric, v in metrics.items():
                vals[sc][mode][metric][pf] = v / 5 # per 100 episodes

    rows = len(scenarios)
    cols = len(prefixes)
    fig_width = cols * 8
    fig_height = rows * 4

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey='row'
    )

    colors = {
        "WITHOUT SUPERVISOR": "#1f77b4",
        "WITH SUPERVISOR": "#ff7f0e"
    }

    for i, sc in enumerate(scenarios):
        # Collect all metrics that appear at least once, excluding "Speeding"
        all_metrics = sorted({
            met for mode in modes
            for met in vals[sc][mode].keys()
            if met != "Speeding"
        })

        x = range(len(all_metrics))
        width = 0.35

        for j, pf in enumerate(prefixes):
            ax = axes[i][j]
            for k, mode in enumerate(modes):
                offset = (-0.5 + k) * width
                positions = [pos + offset for pos in x]
                heights = [vals[sc][mode].get(met, {}).get(pf, 0) for met in all_metrics]
                label = f"{mode}" if i == 0 and j == 0 else None
                ax.bar(positions, heights, width, label=label, color=colors[mode])

            ax.set_ylim(0, 11)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_xticks(list(x))
            ax.set_xticklabels([])

    for ax, col in zip(axes[0], prefixes):
        ax.set_title(PREFIX_LABELS[col], fontsize=24, pad=20)
    
    for ax, row in zip(axes[:, 0], scenarios):
        ax.set_ylabel(SCENARIO_LABELS[row], rotation=0, fontsize=24, ha='right')
        ax.yaxis.set_label_coords(-0.15, 0.5)  # Adjust y-label position

    axes[-1, 0].set_xticklabels(all_metrics, rotation=45, fontsize=24, ha='right')
    axes[-1, 1].set_xticklabels(all_metrics, rotation=45, fontsize=24, ha='right')

    fig.supylabel("Normalized Violation Rate", fontsize=28, fontweight='bold', ha='center', y=0.55)

    # Shared legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, [label.capitalize() for label in labels],
        ncol=1,
        fontsize=24,
        loc='upper center',
        bbox_to_anchor=(1.15, 0.6)
    )

    fig.suptitle("Distribution of Norm Violations by Model and Scenario", fontsize=32, fontweight='bold', x=0.65, y=1)

    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "comparison_filtered_default.eps")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')

    print(f"Saved final comparison plot (no Speeding) to {save_path}")


if __name__ == '__main__':
    analyze()
