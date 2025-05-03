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
    'legend.fontsize': 20,
    'legend.title_fontsize': 20
})

RESULTS_DIR = "results"
OUTPUT_DIR = "analysis"
OUTPUT_FILE = "comparison.png"
LEGEND_FILE = "legend.png"

PREFIX_LABELS = {
    "2_": "2 Lanes & 5 Vehicles Model",
    "4_": "4 Lanes & 20 Vehicles Model"
}
SCENARIO_LABELS = {
    "in_2": "2 Lanes & 5 Vehicles Scenario",
    "in_4": "4 Lanes & 20 Vehicles Scenario",
    "in_speeding": "Speeding Scenario",
    "in_switching": "Lane Switching Scenario",
    "in_tailgating": "Tailgating Scenario"
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
                if line.startswith("Average collisions:"):
                    m = re.search(r"Average collisions:\s*([\d\.]+)", line)
                    if m: data[mode]["Collisions"] = float(m.group(1))
                elif line.startswith("Average total unavoided violations:"):
                    m = re.search(r"Average total unavoided violations:\s*([\d\.]+)", line)
                    if m: data[mode]["TotalUnavoided"] = float(m.group(1))
                elif line.startswith("Average total avoided violations:"):
                    m = re.search(r"Average total avoided violations:\s*([\d\.]+)", line)
                    if m: data[mode]["TotalAvoided"] = float(m.group(1))
                elif mode == "WITHOUT SUPERVISOR" and "Average unavoided violatoins by type:" in line:
                    d = ast.literal_eval(line.split(":",1)[1].strip())
                    for k, v in d.items(): data[mode][k] = float(v)
                elif mode == "WITH SUPERVISOR" and "Average avoided violatoins by type:" in line:
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
                vals[sc][mode][metric][pf] = v

    max_metrics = max(len(vals[sc][mode]) for sc in scenarios for mode in modes)
    cols = len(modes)
    rows = len(scenarios)
    fig_width  = cols * max(6, max_metrics * 0.2)
    fig_height = rows * 6

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey='row',
        constrained_layout=True
    )

    for i, sc in enumerate(scenarios):
        sc_label = SCENARIO_LABELS.get(sc, sc)
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            metrics = sorted(vals[sc][mode].keys())
            x = range(len(metrics))
            n = len(prefixes)
            width = 0.8 / n

            for k, pf in enumerate(prefixes):
                heights   = [vals[sc][mode].get(met, {}).get(pf, 0) for met in metrics]
                positions = [pos + k * width for pos in x]
                label = PREFIX_LABELS.get(pf) if (i == 0 and j == 0) else None
                ax.bar(positions, heights, width, label=label)

            ax.set_yscale('log')
            centers = [pos + (n-1)*width/2 for pos in x]
            ax.set_xticks(centers)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.set_title(f"{sc_label}\n({mode})")
            if j == 0:
                ax.set_ylabel('Avg Values (log-scale)')

    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    fig.savefig(save_path, dpi=150)

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_fig = plt.figure(figsize=(4, 4))
    legend_fig.legend(handles, labels, title='Model Prefix', loc='center')
    legend_path = os.path.join(OUTPUT_DIR, LEGEND_FILE)
    legend_fig.savefig(legend_path, dpi=150, bbox_inches='tight')

    print(f"Saved comparison plot to {save_path}")
    print(f"Saved legend to {legend_path}")

if __name__ == '__main__':
    analyze()
