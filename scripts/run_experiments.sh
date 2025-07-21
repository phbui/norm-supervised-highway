#!/bin/bash

# Constants for experiments and episodes
NUM_EXPERIMENTS=5
NUM_EPISODES=100

# Set to true to overwrite existing results
FORCE_WRITE=false

experiments=(
    # BASELINES
    # <profile>  <method>  <value>  <filter>
    " cautious   nop           nan  no-filter "  # Cautious unsupervised
    " cautious   nop           nan  filter    "  # Cautious filter-only
    " cautious   naive         nan  filter    "  # Cautious naive augment
    " efficient  nop           nan  no-filter "  # Efficient unsupervised
    " efficient  nop           nan  filter    "  # Efficient filter-only
    " efficient  naive         nan  filter    "  # Efficient naive augment

    # ADAPTIVE
    # <profile>  <method>  <value>  <filter>
    " cautious   adaptive     0.01  filter    "
    " cautious   adaptive     0.10  filter    "
    " cautious   adaptive     1.00  filter    "
    " efficient  adaptive     0.01  filter    "
    " efficient  adaptive     0.10  filter    "
    " efficient  adaptive     1.00  filter    "

    # FIXED
    # <profile>  <method>  <value>  <filter>
    " cautious   fixed        0.01  filter    "
    " cautious   fixed        0.10  filter    "
    " cautious   fixed        1.00  filter    "
    " efficient  fixed        0.01  filter    "
    " efficient  fixed        0.10  filter    "
    " efficient  fixed        1.00  filter    "

    # GREEDY
    # <profile>  <method>  <value>  <filter>
    " cautious   projection    nan  filter    "
    " efficient  projection    nan  filter    "

    # LOG SPACED TRIALS
    # <profile>  <method>  <value>  <filter>
    " cautious   adaptive   0.0316  filter    "
    " cautious   adaptive   0.3162  filter    "
    " cautious   adaptive   3.1623  filter    "
    " cautious   adaptive   10.000  filter    "

    # ABLATIONS
    # <profile>  <method>  <value>  <filter>
    " cautious   naive         nan  no-filter "
    " efficient  naive         nan  no-filter "
    " cautious   adaptive     0.01  no-filter "
    " cautious   adaptive     0.10  no-filter "
    " cautious   adaptive     1.00  no-filter "
    " efficient  adaptive     0.01  no-filter "
    " efficient  adaptive     0.10  no-filter "
    " efficient  adaptive     1.00  no-filter "
    " cautious   fixed        0.01  no-filter "
    " cautious   fixed        0.10  no-filter "
    " cautious   fixed        1.00  no-filter "
    " efficient  fixed        0.01  no-filter "
    " efficient  fixed        0.10  no-filter "
    " efficient  fixed        1.00  no-filter "
    " cautious   projection    nan  no-filter "
    " efficient  projection    nan  no-filter "
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create results directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/results"

for args in "${experiments[@]}"; do
    read profile method value filter <<< "$args"

    if [ "$filter" == "filter" ]; then
        filter_arg="--filter"
    else
        filter_arg="--no-filter"
    fi

    # Create subdirectory based on method and filter status
    if [ "$method" == "nop" ]; then
        if [ "$filter" == "filter" ]; then 
            sub_dir="filter_only"
        else 
            sub_dir="unsupervised"
        fi
    else
        if [ "$filter" == "filter" ]; then 
            filter_suffix="_filtered"
        else 
            filter_suffix="_unfiltered"
        fi
        sub_dir="${method}${filter_suffix}"
    fi

    # Create subdirectory with profile prefix
    mkdir -p "$PROJECT_ROOT/results/${profile}/${sub_dir}"

    # Build output filename
    if [ "$method" == "adaptive" ] || [ "$method" == "fixed" ]; then
        out_path="$PROJECT_ROOT/results/${profile}/${sub_dir}/2L5V_2L5V_${value}.csv"
        if [ "$FORCE_WRITE" = true ] && [ -f "$out_path" ]; then
            echo "Overwriting existing results: $out_path"
        elif [ -f "$out_path" ]; then
            echo "Skipping existing results: $out_path"
            continue
        fi
        echo "Running ${profile} ${method} with value=${value} filter=${filter} -> $out_path"
        python "$SCRIPT_DIR/test.py" \
            --profile "$profile" \
            --method "$method" \
            --value "$value" \
            "$filter_arg" \
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
        echo "Running ${profile} ${method} filter=${filter} -> $out_path"
        python "$SCRIPT_DIR/test.py" \
            --profile "$profile" \
            --method "$method" \
            "$filter_arg" \
            --experiments "$NUM_EXPERIMENTS" \
            --episodes "$NUM_EPISODES" \
            --output "$out_path"
    fi
done

echo "All experiments completed!"
