import os
import re
import ast
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUTPUT_DIR = "analysis"
OUTPUT_FILE = "comparison.png"

# Custom label mappings
PREFIX_LABELS = {
    "2_": "Trained for 2 Lanes & 5 Vehicles",
    "4_": "Trained for 4 Lanes & 20 Vehicles"
}
SCENARIO_LABELS = {
    "in_2": "2 Lanes & 5 Vehicles Scenario",
    "in_4": "4 Lanes & 20 Vehicles Scenario",
    "in_speeding": "Speeding Scenario",
    "in_switching": "Lane Switching Scenario",
    "in_tailgating": "Tailgating Scenario"
}


def list_files(directory, extension=".txt"):
    """Return sorted .txt files in a directory."""
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])


def parse_results(path):
    """Parse a results file and extract key metrics into a flat dict per mode."""
    data = {}
    mode = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line in ("WITH SUPERVISOR", "WITHOUT SUPERVISOR"):
                mode = line
                data[mode] = {}
            elif mode:
                # Average collisions
                if line.startswith("Average collisions:"):
                    m = re.search(r"Average collisions:\s*([\d\.]+)", line)
                    if m: data[mode]["Collisions"] = float(m.group(1))
                # Average total unavoided violations
                elif line.startswith("Average total unavoided violations:"):
                    m = re.search(r"Average total unavoided violations:\s*([\d\.]+)", line)
                    if m: data[mode]["TotalUnavoided"] = float(m.group(1))
                # Average total avoided violations
                elif line.startswith("Average total avoided violations:"):
                    m = re.search(r"Average total avoided violations:\s*([\d\.]+)", line)
                    if m: data[mode]["TotalAvoided"] = float(m.group(1))
                # Per-type unavoided violations
                elif mode == "WITHOUT SUPERVISOR" and "Average unavoided violatoins by type:" in line:
                    d = ast.literal_eval(line.split(":",1)[1].strip())
                    for k,v in d.items(): data[mode][k] = float(v)
                # Per-type avoided violations
                elif mode == "WITH SUPERVISOR" and "Average avoided violatoins by type:" in line:
                    d = ast.literal_eval(line.split(":",1)[1].strip())
                    for k,v in d.items(): data[mode][k] = float(v)
    return data


def analyze():
    files = list_files(RESULTS_DIR)
    if not files:
        print(f"No .txt files found in '{RESULTS_DIR}/'.")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect unique prefixes and scenarios
    prefixes = sorted({fn.split('_')[0] + '_' for fn in files})
    scenarios = sorted({fn.split('_',1)[1].rsplit('.txt',1)[0] for fn in files})
    modes = ["WITHOUT SUPERVISOR", "WITH SUPERVISOR"]

    # Organize values: vals[scenario][mode][metric][prefix]
    vals = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    metrics = set()

    for fn in files:
        pf = fn.split('_',1)[0] + '_'
        sc = fn.split('_',1)[1].rsplit('.txt',1)[0]
        data = parse_results(os.path.join(RESULTS_DIR, fn))
        for mode in modes:
            for metric, v in data.get(mode, {}).items():
                vals[sc][mode][metric][pf] = v
                metrics.add(metric)

    # Plot: one row per scenario, two columns per mode
    rows = len(scenarios)
    cols = len(modes)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4), squeeze=False)

    for i, sc in enumerate(scenarios):
        sc_label = SCENARIO_LABELS.get(sc, sc)
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            m_list = sorted(vals[sc][mode].keys())
            x = range(len(m_list))
            n = len(prefixes)
            width = 0.8 / n
            for k, pf in enumerate(prefixes):
                heights = [vals[sc][mode].get(met, {}).get(pf, 0) for met in m_list]
                positions = [pos + k*width for pos in x]
                pf_label = PREFIX_LABELS.get(pf, pf)
                label = pf_label if i==0 and j==0 else None
                ax.bar(positions, heights, width, label=label)

            ax.set_yscale('log')
            centers = [pos + (n-1)*width/2 for pos in x]
            ax.set_xticks(centers)
            ax.set_xticklabels(m_list, rotation=45, ha='right', fontsize=8)
            ax.set_title(f"{sc_label} ({mode})", fontsize=10)
            if j==0:
                ax.set_ylabel('Average Values (log-scale)', fontsize=9)
            if i==0 and j==0:
                ax.legend(title='Model Prefix', fontsize=8, title_fontsize=9)

    plt.tight_layout()
    # Save figure
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    fig.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")


if __name__ == '__main__':
    analyze()
