#!/bin/bash

# Map from numeric input to label
label() {
    if [ "$1" -eq 0 ]; then
        echo "2L5V"
    else
        echo "4L20V"
    fi
}

# Format: method value model config
experiments=(
    "fixed    0.3    0 0"
    "fixed    1      0 0"
    "fixed    3      0 0"
    "fixed    10     0 0"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for args in "${experiments[@]}"; do
    read method value model config <<< "$args"
    model_label=$(label "$model")
    config_label=$(label "$config")

    sub_dir="${method}"  # infer subdirectory from method name

    if [ "$method" == "baseline" ]; then
        out_path="results/${sub_dir}/${model_label}_${config_label}"
        echo "Running baseline ${model_label} in ${config_label} -> $out_path.txt"
        python "$SCRIPT_DIR/experiment_run.py" <<EOF
$model
$config
$out_path
EOF
    else
        out_path="results/${sub_dir}/${model_label}_${config_label}_${value}"
        echo "Running ${method} ${model_label} in ${config_label} with value=${value} -> $out_path.txt"
        python "$SCRIPT_DIR/experiment_run.py" "$method" "$value" <<EOF
$model
$config
$out_path
EOF
    fi
done
