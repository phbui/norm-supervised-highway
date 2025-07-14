#!/bin/bash

# Constants for experiments and episodes
NUM_EXPERIMENTS=5
NUM_EPISODES=100

# Set to true to overwrite existing results
FORCE_WRITE=true

experiments=(
    # BASELINES
    # <profile>  <mode>         <method>  <value>  <model>  <env>
    #" cautious   unsupervised   nan           nan  2L5V     2L5V "
    #" cautious   filter_only    nan           nan  2L5V     2L5V "
    " cautious   naive_augment  nan           nan  2L5V     2L5V "
    #" efficient  unsupervised   nan           nan  2L5V     2L5V "
    #" efficient  filter_only    nan           nan  2L5V     2L5V "
    " efficient  naive_augment  nan           nan  2L5V     2L5V "

    # ADAPTIVE
    # <profile>  <mode>         <method>  <value>  <model>  <env>
    #" cautious   default        adaptive    0.001  2L5V     2L5V "
    #" cautious   default        adaptive    0.003  2L5V     2L5V "
    #" cautious   default        adaptive    0.010  2L5V     2L5V "
    #" efficient  default        adaptive    0.001  2L5V     2L5V "
    #" efficient  default        adaptive    0.003  2L5V     2L5V "
    #" efficient  default        adaptive    0.010  2L5V     2L5V "

    # FIXED
    # <profile>  <mode>         <method>  <value>  <model>  <env>
    #" cautious   default        fixed           1  2L5V     2L5V "
    #" cautious   default        fixed           3  2L5V     2L5V "
    #" cautious   default        fixed          10  2L5V     2L5V "
    #" efficient  default        fixed           1  2L5V     2L5V "
    #" efficient  default        fixed           3  2L5V     2L5V "
    #" efficient  default        fixed          10  2L5V     2L5V "
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create results directory if it doesn't exist
mkdir -p results

for args in "${experiments[@]}"; do
    read profile mode method value model env <<< "$args"

    # Create subdirectory based on mode (use method for default mode)
    if [ "$mode" == "default" ]; then
        sub_dir="${method}"
    else
        sub_dir="${mode}"
    fi

    # Create subdirectory with profile prefix
    mkdir -p "results/${profile}/${sub_dir}"

    # Build output filename
    if [ "$mode" == "default" ]; then
        out_path="results/${profile}/${sub_dir}/${model}_${env}_${value}.csv"
        if [ "$FORCE_WRITE" = true ] && [ -f "$out_path" ]; then
            echo "Overwriting existing results: $out_path"
        elif [ -f "$out_path" ]; then
            echo "Skipping existing results: $out_path"
            continue
        fi
        echo "Running ${profile} ${mode} ${method} ${model} in ${env} with value=${value} -> $out_path"
        python "$SCRIPT_DIR/test.py" \
            --profile "$profile" \
            --mode "$mode" \
            --method "$method" \
            --value "$value" \
            --model "$model" \
            --env "$env" \
            --experiments "$NUM_EXPERIMENTS" \
            --episodes "$NUM_EPISODES" \
            --output "$out_path"
    else
        out_path="results/${profile}/${sub_dir}/${model}_${env}.csv"
        if [ "$FORCE_WRITE" = true ] && [ -f "$out_path" ]; then
            echo "Overwriting existing results: $out_path"
        elif [ -f "$out_path" ]; then
            echo "Skipping existing results: $out_path"
            continue
        fi
        echo "Running ${profile} ${mode} ${model} in ${env} -> $out_path"
        python "$SCRIPT_DIR/test.py" \
            --profile "$profile" \
            --mode "$mode" \
            --model "$model" \
            --env "$env" \
            --experiments "$NUM_EXPERIMENTS" \
            --episodes "$NUM_EPISODES" \
            --output "$out_path"
    fi
done

echo "All experiments completed!"
