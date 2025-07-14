#!/usr/bin/env python3
"""
Analysis script for norm supervisor experiment results.
Scans CSV files in results directory and generates summary statistics.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import argparse


def list_csv_files(directory):
    """List all CSV files in the given directory and subdirectories."""
    if not os.path.exists(directory):
        return []
    
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                # Get relative path from the results directory
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                csv_files.append(rel_path)
    
    return sorted(csv_files)


def parse_configuration_from_filename(filename):
    """Extract configuration from CSV filename."""
    # Expected format: profile/method/model_env.csv or profile/method/model_env_value.csv
    # Examples: 
    # - cautious/unsupervised/2L5V_2L5V.csv
    # - cautious/adaptive/2L5V_2L5V_0.003.csv
    
    # Remove .csv extension
    filename = filename.replace('.csv', '')
    
    # Split by path separators
    parts = filename.split('/')
    
    if len(parts) != 3:
        return None
    
    profile = parts[0]
    method = parts[1]
    
    # Parse the model_env part (with optional value)
    model_env_part = parts[2]
    if '_' not in model_env_part:
        return None
    
    # Check if there's a value at the end (format: model_env_value)
    if model_env_part.count('_') >= 2:
        # Has value: split on last underscore
        last_underscore_idx = model_env_part.rfind('_')
        model_env = model_env_part[:last_underscore_idx]
        value_str = model_env_part[last_underscore_idx + 1:]
        
        # Parse value
        try:
            value = float(value_str) if value_str != 'None' else None
        except ValueError:
            return None
    else:
        # No value: just model_env
        model_env = model_env_part
        value = None
    
    # Parse model and environment from model_env
    # Format is model_env (e.g., "2L5V_2L5V" or "4L20V_4L20V")
    if '_' not in model_env:
        return None
    
    model_env_underscore_idx = model_env.rfind('_')
    model = model_env[:model_env_underscore_idx]
    env = model_env[model_env_underscore_idx + 1:]
    
    return {
        'profile': profile,
        'method': method,
        'value': value,
        'model': model,
        'env': env,
        'mode': 'default'  # Default mode since it's not in filename
    }


def load_and_group_data(results_dir):
    """Load all CSV files and group by configuration."""
    csv_files = list_csv_files(results_dir)
    
    if not csv_files:
        print(f"No CSV files found in '{results_dir}'")
        return {}
    
    grouped_data = defaultdict(list)
    
    for filename in csv_files:
        config = parse_configuration_from_filename(filename)
        if config is None:
            print(f"Warning: Could not parse configuration from filename: {filename}")
            continue
        
        filepath = os.path.join(results_dir, filename)
        try:
            df = pd.read_csv(filepath)
            if not df.empty:
                # Add configuration info to each row
                for key, value in config.items():
                    df[key] = value
                grouped_data[tuple(sorted(config.items()))].append(df)
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")
    
    return grouped_data


def calculate_statistics(group_data):
    """Calculate mean and standard deviation for all numeric columns."""
    # Combine all dataframes in the group
    combined_df = pd.concat(group_data, ignore_index=True)
    
    # Get numeric columns (excluding configuration columns)
    config_cols = ['profile', 'method', 'value', 'model', 'env', 'mode']
    numeric_cols = [col for col in combined_df.columns 
                   if col not in config_cols and combined_df[col].dtype in ['float64', 'int64']]
    
    stats = {}
    for col in numeric_cols:
        values = combined_df[col].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            stats[col] = (mean_val, std_val)
        else:
            stats[col] = (np.nan, np.nan)
    
    return stats, len(combined_df)


def format_statistic(mean_val, std_val, metric_name=None):
    """Format mean and standard deviation as 'mean (std)'."""
    if pd.isna(mean_val) or pd.isna(std_val):
        return "-"
    
    # Multiply rates by 100 for better readability
    if metric_name and ('rate' in metric_name or 'violations' in metric_name):
        mean_val *= 100
        std_val *= 100
    
    return f"{mean_val:.2f} ({std_val:.2f})"


def get_lane_columns(stats):
    """Get lane time columns from statistics."""
    return sorted([col for col in stats.keys() if col.startswith('lane_') and col.endswith('_time_mean')])


def generate_markdown_tables(grouped_data):
    """Generate markdown tables from grouped data, one per model-environment combination."""
    
    # Define metric categories and their display names
    metric_categories = {
        'Episode Metrics': ['episode_length_mean', 'collision_count'],
        'Safety Metrics': ['mean_ttc', 'mean_following_distance', 'mean_speed'],
        'Violation Rates': [
            'speed_violations_rate_mean', 'tailgating_violations_rate_mean',
            'braking_violations_rate_mean', 'lane_change_tailgating_violations_rate_mean',
            'lane_change_braking_violations_rate_mean', 'collision_violations_rate_mean',
            'lane_change_collision_violations_rate_mean'
        ],
        'Cost Metrics': ['cost_rate_mean', 'avoided_cost_rate_mean'],
        'Lane Usage': []  # Will be populated dynamically
    }
    
    # Create display names mapping (replace underscores with spaces and remove "mean")
    display_names = {}
    for category, metrics in metric_categories.items():
        for metric in metrics:
            display_name = metric.replace('_', ' ')
            # Remove "mean" from the end
            if display_name.endswith(' mean'):
                display_name = display_name[:-5]  # Remove " mean"
            display_names[metric] = display_name
    
    # Add lane usage metrics to display names
    all_lane_cols = set()
    for group_data in grouped_data.values():
        stats, _ = calculate_statistics(group_data)
        lane_cols = get_lane_columns(stats)
        all_lane_cols.update(lane_cols)
    
    metric_categories['Lane Usage'] = sorted(all_lane_cols)
    for lane_col in all_lane_cols:
        display_name = lane_col.replace('_', ' ')
        # Remove "mean" from the end
        if display_name.endswith(' mean'):
            display_name = display_name[:-5]  # Remove " mean"
        display_names[lane_col] = display_name

    # Build header row
    header_cols = ['Method']
    for category, metrics in metric_categories.items():
        header_cols.extend([display_names.get(metric, metric) for metric in metrics])
    
    # Group data by model-environment combination first, then by profile
    model_env_groups = defaultdict(lambda: defaultdict(dict))
    for config_tuple, group_data in grouped_data.items():
        config_dict = dict(config_tuple)
        model_env_key = (config_dict['model'], config_dict['env'])
        profile = config_dict['profile']
        model_env_groups[model_env_key][profile][config_tuple] = group_data
    
    tables = []
    for (model, env), profile_configs in sorted(model_env_groups.items()):
        # Build table rows for this model-environment combination
        rows = []
        
        for profile, configs in sorted(profile_configs.items()):
            # Add separator row if not the first profile
            if rows:  # Add empty row before new section
                rows.append([''] * len(header_cols))
            
            # Add section header
            section_header = f"**{profile.title()} Profile**"
            separator_row = [section_header] + [''] * (len(header_cols) - 1)
            rows.append(separator_row)
            
            for config_tuple, group_data in sorted(configs.items()):
                config_dict = dict(config_tuple)
                stats, n_experiments = calculate_statistics(group_data)
                
                # Create configuration name (just method and value)
                if config_dict['method'] in ['adaptive', 'fixed'] and config_dict['value'] is not None:
                    config_name = f"{config_dict['method'].title()} ({config_dict['value']})"
                else:
                    config_name = f"{config_dict['method'].title()}"
                
                # Build data row
                row = [config_name]
                for category, metrics in metric_categories.items():
                    for metric in metrics:
                        if metric in stats:
                            mean_val, std_val = stats[metric]
                            row.append(format_statistic(mean_val, std_val, metric))
                        else:
                            row.append("-")
                
                rows.append(row)
        
        tables.append({
            'model': model,
            'env': env,
            'header_cols': header_cols,
            'rows': rows
        })
    
    return tables


def generate_summary_table(grouped_data):
    """Generate a summary table showing experiment counts and total episodes for each configuration."""
    summary_rows = []
    
    # Collect all unique configurations
    all_configs = []
    for config_tuple, group_data in grouped_data.items():
        config_dict = dict(config_tuple)
        all_configs.append(config_dict)
    
    # Sort configurations by profile, model, env, method, value
    all_configs.sort(key=lambda x: (x['profile'], x['model'], x['env'], x['method'], x['value'] or 0))
    
    for config in all_configs:
        # Find the corresponding group data
        config_tuple = tuple(sorted(config.items()))
        if config_tuple in grouped_data:
            group_data = grouped_data[config_tuple]
            stats, n_experiments = calculate_statistics(group_data)
            
            # Calculate total episodes
            total_episodes = 0
            for df in group_data:
                if 'num_episodes' in df.columns:
                    total_episodes += df['num_episodes'].sum()
            
            # Create configuration name
            if config['method'] in ['adaptive', 'fixed'] and config['value'] is not None:
                method_name = f"{config['method'].title()} ({config['value']})"
            else:
                method_name = f"{config['method'].title()}"
            
            # Create row
            row = [
                config['profile'].title(),
                f"{config['model']} {config['env']}",
                method_name,
                str(n_experiments),
                str(total_episodes)
            ]
            summary_rows.append(row)
    
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results from CSV files')
    parser.add_argument('--results-dir', default=None, 
                       help='Directory containing CSV result files (default: script_dir/../results)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for analysis files (default: script_dir/../analysis)')
    
    args = parser.parse_args()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Set default paths relative to project root
    if args.results_dir is None:
        args.results_dir = os.path.join(project_root, 'results')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'analysis')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and group data
    print(f"Scanning for CSV files in {args.results_dir}...")
    grouped_data = load_and_group_data(args.results_dir)
    
    if not grouped_data:
        print("No valid data found. Exiting.")
        return
    
    print(f"Found {len(grouped_data)} configuration groups")
    
    # Generate markdown tables
    print("Generating markdown tables...")
    tables = generate_markdown_tables(grouped_data)
    
    # Generate summary table
    print("Generating summary table...")
    summary_rows = generate_summary_table(grouped_data)
    
    # Write output (will overwrite existing file)
    output_file = os.path.join(args.output_dir, 'summary.md')
    
    # Write overview (overwrites any existing file)
    with open(output_file, 'w') as f:
        f.write("# Experiment Results Summary\n")
    
    # Write each table
    for table_data in tables:
        model = table_data['model']
        env = table_data['env']
        header_cols = table_data['header_cols']
        rows = table_data['rows']
        
        with open(output_file, 'a') as f:
            f.write(f"\n## {model} {env} Results\n\n")
            f.write("*Violation and cost rates are normalized by episode length and given per 100 "
                    "episodes, and lane times are given as a proportion of the total episode length\n\n")
            f.write("| " + " | ".join(header_cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(header_cols)) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
    
    # Write summary table
    with open(output_file, 'a') as f:
        f.write("\n## Experiment Summary\n\n")
        f.write("| Profile | Model-Environment | Method | Experiments | Total Episodes |\n")
        f.write("|---------|-------------------|--------|-------------|----------------|\n")
        for row in summary_rows:
            f.write("| " + " | ".join(row) + " |\n")
    
    print(f"Analysis complete! Results written to {output_file}")


if __name__ == '__main__':
    main()
