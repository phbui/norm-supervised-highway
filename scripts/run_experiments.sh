#!/bin/bash

# Constants for experiments and episodes
NUM_EXPERIMENTS=5
NUM_EPISODES=100

# Set to true to overwrite existing results
FORCE_WRITE=false

experiments=(
    # BASELINES
    # <profile>  <mode>         <method>  <value>  <filter>
    " cautious   nop            nan           nan   false "  # Cautious unsupervised
    " cautious   nop            nan           nan   true  "
    " cautious   naive_augment  nan           nan   true  "
    " efficient  nop            nan           nan   false "  # Efficient unsupervised
    " efficient  nop            nan           nan   true  " 
    " efficient  naive_augment  nan           nan   true  " 

    # ADAPTIVE
    # <profile>  <mode>         <method>  <value>  <filter>
    " cautious   default        adaptive    0.005   true "
    " cautious   default        adaptive    0.010   true "
    " cautious   default        adaptive    0.050   true "
    " efficient  default        adaptive    0.005   true "
    " efficient  default        adaptive    0.010   true "
    " efficient  default        adaptive    0.050   true "

    # FIXED
    # <profile>  <mode>         <method>  <value>  <filter>
    " cautious   default        fixed         0.5   true "
    " cautious   default        fixed           1   true "
    " cautious   default        fixed           5   true "
    " efficient  default        fixed         0.5   true "
    " efficient  default        fixed           1   true "
    " efficient  default        fixed           5   true "

    # ABLATIONS
    # <profile>  <mode>         <method>  <value>  <filter>
    " cautious   naive_augment  nan           nan   false "
    " efficient  naive_augment  nan           nan   false "
    " cautious   default        adaptive    0.005   false "
    " cautious   default        adaptive    0.010   false "
    " cautious   default        adaptive    0.050   false "
    " efficient  default        adaptive    0.005   false "
    " efficient  default        adaptive    0.010   false "
    " efficient  default        adaptive    0.050   false "
    " cautious   default        fixed         0.5   false "
    " cautious   default        fixed           1   false "
    " cautious   default        fixed           5   false "
    " efficient  default        fixed         0.5   false "
    " efficient  default        fixed           1   false "
    " efficient  default        fixed           5   false "
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create results directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/results"

for args in "${experiments[@]}"; do
    read profile mode method value filter <<< "$args"

    # Create subdirectory based on mode and filter status
    if [ "$mode" == "default" ]; then
        if [ "$filter" == "true" ]; then 
            filter_suffix="_filtered"
        else 
            filter_suffix="_unfiltered"
        fi
        sub_dir="${method}${filter_suffix}"
    elif [ "$mode" == "naive_augment" ]; then
        if [ "$filter" == "true" ]; then 
            filter_suffix="_filtered"
        else 
            filter_suffix="_unfiltered"
        fi
        sub_dir="${mode}${filter_suffix}"
    elif [ "$mode" == "nop" ]; then
        if [ "$filter" == "true" ]; then 
            sub_dir="filter_only"
        else 
            sub_dir="unsupervised"
        fi
    else
        sub_dir="${mode}"
    fi

    # Create subdirectory with profile prefix
    mkdir -p "$PROJECT_ROOT/results/${profile}/${sub_dir}"

    # Build output filename
    if [ "$mode" == "default" ]; then
        out_path="$PROJECT_ROOT/results/${profile}/${sub_dir}/2L5V_2L5V_${value}.csv"
        if [ "$FORCE_WRITE" = true ] && [ -f "$out_path" ]; then
            echo "Overwriting existing results: $out_path"
        elif [ -f "$out_path" ]; then
            echo "Skipping existing results: $out_path"
            continue
        fi
        echo "Running ${profile} ${mode} ${method} with value=${value} filter=${filter} -> $out_path"
        python "$SCRIPT_DIR/test.py" \
            --profile "$profile" \
            --mode "$mode" \
            --method "$method" \
            --value "$value" \
            --filter \
            --experiments "$NUM_EXPERIMENTS" \
            --episodes "$NUM_EPISODES" \
            --output "$out_path"
    else
        out_path="$PROJECT_ROOT/results/${profile}/${sub_dir}/2L5V_2L5V.csv"
        if [ "$FORCE_WRITE" = true ] && [ -f "$out_path" ]; then
            echo "Overwriting existing results: $out_path"
        elif [ -f "$out_path" ]; then
            echo "Skipping existing results: $out_path"
            continue
        fi
        echo "Running ${profile} ${mode} filter=${filter} -> $out_path"
        if [ "$filter" == "true" ]; then
            python "$SCRIPT_DIR/test.py" \
                --profile "$profile" \
                --mode "$mode" \
                --filter \
                --experiments "$NUM_EXPERIMENTS" \
                --episodes "$NUM_EPISODES" \
                --output "$out_path"
        else
            python "$SCRIPT_DIR/test.py" \
                --profile "$profile" \
                --mode "$mode" \
                --no-filter \
                --experiments "$NUM_EXPERIMENTS" \
                --episodes "$NUM_EPISODES" \
                --output "$out_path"
        fi
    fi
done

echo "All experiments completed!"
