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

import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt


def list_csv_files(directory):
    """List all CSV files in the given directory and subdirectories, skipping any 'ignore' directories."""
    if not os.path.exists(directory):
        return []
    
    csv_files = []
    for root, dirs, files in os.walk(directory):
        # Skip any directory named 'ignore'
        if 'ignore' in dirs:
            dirs.remove('ignore')
        if os.path.basename(root) == 'ignore':
            continue
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
    summary_metric_categories = {
        'Episode Metrics': ['episode_length_mean', 'km_per_collision'],  # Replace collisions_per_km
        'Safety Metrics': ['mean_ttc', 'mean_following_distance', 'mean_speed'],
        'Cost Metrics': ['cost_rate_mean', 'avoided_cost_rate_mean'],
        'Lane Usage': []  # Will be populated dynamically
    }
    
    details_metric_categories = {
        'Violation Rates': [
            'speed_violations_rate_mean', 'tailgating_violations_rate_mean',
            'braking_violations_rate_mean', 'lane_change_tailgating_violations_rate_mean',
            'lane_change_braking_violations_rate_mean', 'collision_violations_rate_mean',
            'lane_change_collision_violations_rate_mean'
        ],
        'Cost Metrics': ['cost_rate_mean', 'avoided_cost_rate_mean']
    }
    
    # Create display names mapping (replace underscores with spaces and remove "mean")
    display_names = {}
    for category, metrics in summary_metric_categories.items():
        for metric in metrics:
            display_name = metric.replace('_', ' ')
            # Remove "mean" from the beginning or end
            if display_name.startswith('mean '):
                display_name = display_name[5:]
            if display_name.endswith(' mean'):
                display_name = display_name[:-5]  # Remove " mean"
            display_names[metric] = display_name
    
    for category, metrics in details_metric_categories.items():
        for metric in metrics:
            display_name = metric.replace('_', ' ')
            # Remove "mean" from the beginning or end
            if display_name.startswith('mean '):
                display_name = display_name[5:]
            if display_name.endswith(' mean'):
                display_name = display_name[:-5]  # Remove " mean"
            display_names[metric] = display_name
    
    # Add lane usage metrics to display names
    all_lane_cols = set()
    for group_data in grouped_data.values():
        stats, _ = calculate_statistics(group_data)
        lane_cols = get_lane_columns(stats)
        all_lane_cols.update(lane_cols)
    
    summary_metric_categories['Lane Usage'] = sorted(all_lane_cols)
    for lane_col in all_lane_cols:
        display_name = lane_col.replace('_', ' ')
        # Remove "mean" from the end
        if display_name.endswith(' mean'):
            display_name = display_name[:-5]  # Remove " mean"
        display_names[lane_col] = display_name

    # Add display name for new metric
    display_names['km_per_collision'] = 'km per collision'

    # Build header rows
    summary_header_cols = ['method']
    for category, metrics in summary_metric_categories.items():
        summary_header_cols.extend([display_names.get(metric, metric) for metric in metrics])
    
    details_header_cols = ['method']
    for category, metrics in details_metric_categories.items():
        details_header_cols.extend([display_names.get(metric, metric) for metric in metrics])
    
    # Group data by model-environment combination first, then by profile
    model_env_groups = defaultdict(lambda: defaultdict(dict))
    for config_tuple, group_data in grouped_data.items():
        config_dict = dict(config_tuple)
        model_env_key = (config_dict['model'], config_dict['env'])
        profile = config_dict['profile']
        model_env_groups[model_env_key][profile][config_tuple] = group_data
    
    summary_tables = []
    details_tables = []
    
    for (model, env), profile_configs in sorted(model_env_groups.items()):
        # Build table rows for this model-environment combination
        summary_rows = []
        details_rows = []
        
        for profile, configs in sorted(profile_configs.items()):
            # Add separator row if not the first profile
            if summary_rows:  # Add empty row before new section
                summary_rows.append([''] * len(summary_header_cols))
                details_rows.append([''] * len(details_header_cols))
            
            # Add section header
            section_header = f"**{profile.title()} Profile**"
            summary_separator_row = [section_header] + [''] * (len(summary_header_cols) - 1)
            details_separator_row = [section_header] + [''] * (len(details_header_cols) - 1)
            summary_rows.append(summary_separator_row)
            details_rows.append(details_separator_row)
            
            # Sort configs by the new method_sort_key
            sorted_configs = sorted(configs.items(), key=lambda item: method_sort_key(dict(item[0])))
            for config_tuple, group_data in sorted_configs:
                config_dict = dict(config_tuple)
                stats, n_experiments = calculate_statistics(group_data)
                
                # Create configuration name (just method and value)
                if config_dict['method'] in ['adaptive', 'fixed'] and config_dict['value'] is not None:
                    config_name = f"{config_dict['method'].title()} ({config_dict['value']})"
                else:
                    config_name = f"{config_dict['method'].title()}"
                
                # Build summary data row
                summary_row = [config_name]
                for category, metrics in summary_metric_categories.items():
                    for metric in metrics:
                        if metric == 'km_per_collision':
                            # Compute km per collision for each experiment, then mean and std
                            km_per_collision_list = []
                            policy_period = 1/5 # TODO: Use environment config
                            for df in group_data:
                                if 'collision_count' in df.columns and 'mean_speed' in df.columns and 'episode_length_mean' in df.columns and 'num_episodes' in df.columns:
                                    for idx, row in df.iterrows():
                                        collisions = row.get('collision_count', 0)
                                        speed = row.get('mean_speed', np.nan)
                                        ep_length = row.get('episode_length_mean', np.nan)
                                        num_episodes = row.get('num_episodes', np.nan)
                                        if not np.isnan(speed) and not np.isnan(ep_length) and not np.isnan(num_episodes) and collisions > 0:
                                            distance = speed * ep_length * policy_period * num_episodes
                                            km_per_collision = (distance / collisions) / 1000
                                            km_per_collision_list.append(km_per_collision)
                            if km_per_collision_list:
                                mean_val = np.mean(km_per_collision_list)
                                std_val = np.std(km_per_collision_list)
                                summary_row.append(f"{mean_val:.2f} ({std_val:.2f})")
                            else:
                                summary_row.append("-")
                        elif metric in stats:
                            mean_val, std_val = stats[metric]
                            summary_row.append(format_statistic(mean_val, std_val, metric))
                        else:
                            summary_row.append("-")
                
                # Build details data row
                details_row = [config_name]
                for category, metrics in details_metric_categories.items():
                    for metric in metrics:
                        if metric in stats:
                            mean_val, std_val = stats[metric]
                            details_row.append(format_statistic(mean_val, std_val, metric))
                        else:
                            details_row.append("-")
                
                summary_rows.append(summary_row)
                details_rows.append(details_row)
        
        summary_tables.append({
            'model': model,
            'env': env,
            'header_cols': summary_header_cols,
            'rows': summary_rows
        })
        
        details_tables.append({
            'model': model,
            'env': env,
            'header_cols': details_header_cols,
            'rows': details_rows
        })
    
    return summary_tables, details_tables


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


def generate_lane_preference_plot(grouped_data, output_dir):
    """Generate horizontal bar chart showing lane preferences with center line at 0.5."""
    
    # Create plots subdirectory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Group data by model-environment configuration
    model_env_groups = defaultdict(lambda: defaultdict(dict))
    for config_tuple, group_data in grouped_data.items():
        config_dict = dict(config_tuple)
        model_env_key = (config_dict['model'], config_dict['env'])
        profile = config_dict['profile']
        model_env_groups[model_env_key][profile][config_tuple] = group_data
    
    # Generate one plot per model-environment configuration
    for (model, env), profile_configs in sorted(model_env_groups.items()):
        # Collect lane preference data for this model-environment
        preference_data = []
        seen_baselines = set()  # Track seen baseline configurations
        
        # Group data by profile and include headers
        profile_groups = defaultdict(list)
        for profile, configs in profile_configs.items():
            for config_tuple, group_data in configs.items():
                config_dict = dict(config_tuple)
                stats, n_experiments = calculate_statistics(group_data)
                
                # Get lane 1 time (assuming 2-lane scenario)
                lane_1_key = 'lane_1_time_mean'
                if lane_1_key in stats:
                    mean_val, std_val = stats[lane_1_key]
                    
                    # Create configuration name
                    if config_dict['method'] in ['adaptive', 'fixed'] and config_dict['value'] is not None:
                        method_name = f"{config_dict['method'].title()} ({config_dict['value']})"
                    else:
                        method_name = f"{config_dict['method'].title()}"
                    
                    # For baseline methods (unsupervised, filter_only), only include one version
                    if config_dict['method'] in ['unsupervised', 'filter_only']:
                        baseline_key = f"{config_dict['method']}"
                        if baseline_key in seen_baselines:
                            continue
                        seen_baselines.add(baseline_key)
                        # Use just the method name for baselines
                        label = method_name
                    else:
                        # For supervised methods, just use the method name (profile will be grouped visually)
                        label = method_name
                    
                    profile_groups[profile].append({
                        'label': label,
                        'mean': mean_val,
                        'std': std_val,
                        'profile': config_dict['profile'],
                        'method': method_name,
                        'original_method': config_dict['method'],
                        'original_value': config_dict.get('value', 0)
                    })
        
        if not profile_groups:
            print(f"No lane preference data found for {model} {env}. Skipping plot generation.")
            continue
        
        # Build final data with headers in correct positions
        preference_data = []
        
        # Sort data within each profile group first (excluding baselines)
        for profile in profile_groups:
            # Sort each profile group by the same criteria
            def profile_sort_key(data):
                method = data['original_method']
                value = data['original_value']
                
                # Skip baselines in profile sorting since they're handled separately
                if method in ['unsupervised', 'filter_only']:
                    return (999, 0, 0)  # Put baselines at the end for now
                elif method == 'naive_augment':
                    return (0, 0, 0)  # Naive_Augment always first
                elif method == 'adaptive':
                    return (1, 0, value or 0)  # Adaptive next, sorted by value
                else:  # fixed
                    return (2, 0, value or 0)  # Fixed after, sorted by value
            
            profile_groups[profile].sort(key=profile_sort_key)
        
        # Add baselines first (unsupervised and filter_only methods from any profile)
        baseline_data = []
        for profile in profile_groups:
            for data in profile_groups[profile]:
                if data['original_method'] in ['unsupervised', 'filter_only']:
                    data['profile'] = 'no_profile'
                    baseline_data.append(data)
        
        # Sort baselines: unsupervised first, then filter_only
        baseline_data.sort(key=lambda x: (0 if x['original_method'] == 'unsupervised' else 1, 0))
        # Add 'No Profile' header before baselines
        if baseline_data:
            header_data = {
                'label': 'No Profile',
                'mean': np.nan,
                'std': np.nan,
                'profile': 'no_profile',
                'method': 'header',
                'original_method': 'header',
                'original_value': 0
            }
            preference_data.append(header_data)
        preference_data.extend(baseline_data)
        
        # Add cautious profile with header
        if 'cautious' in profile_groups:
            # Add header first
            header_data = {
                'label': 'Cautious Profile',
                'mean': np.nan,
                'std': np.nan,
                'profile': 'cautious',
                'method': 'header',
                'original_method': 'header',
                'original_value': 0
            }
            preference_data.append(header_data)
            # Then add the data (excluding baselines)
            cautious_data = [data for data in profile_groups['cautious'] 
                           if data['original_method'] not in ['unsupervised', 'filter_only']]
            preference_data.extend(cautious_data)
        
        # Add efficient profile with header
        if 'efficient' in profile_groups:
            # Add header first
            header_data = {
                'label': 'Efficient Profile',
                'mean': np.nan,
                'std': np.nan,
                'profile': 'efficient',
                'method': 'header',
                'original_method': 'header',
                'original_value': 0
            }
            preference_data.append(header_data)
            # Then add the data (excluding baselines)
            efficient_data = [data for data in profile_groups['efficient'] 
                           if data['original_method'] not in ['unsupervised', 'filter_only']]
            preference_data.extend(efficient_data)
        
        if not preference_data:
            print(f"No lane preference data found for {model} {env}. Skipping plot generation.")
            continue
        
        # Invert the order so methods appear from bottom to top
        preference_data.reverse()

        # After reversing preference_data and before plotting:
        # Identify y-index ranges for each profile group (excluding headers and baselines)
        group_ranges = {}
        current_profile = None
        start_idx = 0
        for i, data in enumerate(preference_data):
            if data['method'] == 'header':
                if current_profile:
                    group_ranges[current_profile] = (start_idx, i)
                    start_idx = i + 1
            current_profile = data['profile']

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(preference_data) * 0.4)))

        # Shade profile regions light blue, light orange, light gray
        profile_colors = {'cautious': '#e6f2ff', 'efficient': '#fff7e6', 'no_profile': '#f0f0f0'}
        for profile, (ymin, ymax) in group_ranges.items():
            if profile in profile_colors:
                ax.axhspan(ymin-0.5, ymax+0.5, facecolor=profile_colors[profile], alpha=0.4, zorder=0)
        
        # Set up the plot
        y_positions = np.arange(len(preference_data))
        center_line = 0.5
        
        # Define colors for left vs right preference
        left_color = '#ff7f0e'   # Orange for left preference
        right_color = '#1f77b4'  # Blue for right preference
        
        # Plot bars
        for i, data in enumerate(preference_data):
            mean_val = data['mean']
            
            # Skip plotting bars for header entries
            if np.isnan(mean_val) or data['method'] == 'header':
                continue
                
            # Determine bar direction and color
            if mean_val < center_line:
                # Left preference - bar extends left from center
                bar_width = center_line - mean_val
                bar_start = mean_val  # Start at the mean value, extend to center
                color = left_color
            else:
                # Right preference - bar extends right from center
                bar_width = mean_val - center_line
                bar_start = center_line  # Start at center, extend to mean
                color = right_color
            
            # Plot the bar
            ax.barh(i, bar_width, left=bar_start, height=0.6, color=color, alpha=0.7)
            
            # Add better error bars if std is available
            if not np.isnan(data['std']):
                std_val = data['std']
                # Create horizontal error bar
                if mean_val < center_line:
                    # Left preference: error bar extends left from mean
                    error_x = [max(0, mean_val - std_val), mean_val + std_val]
                else:
                    # Right preference: error bar extends right from mean
                    error_x = [mean_val - std_val, min(1, mean_val + std_val)]
                error_y = [i, i]
                ax.plot(error_x, error_y, 'k-', linewidth=2, alpha=0.8)
                # Add error bar caps
                ax.plot([error_x[0], error_x[0]], [i-0.1, i+0.1], 'k-', linewidth=2, alpha=0.8)
                ax.plot([error_x[1], error_x[1]], [i-0.1, i+0.1], 'k-', linewidth=2, alpha=0.8)
        
        # Add center line
        ax.axvline(x=center_line, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        # Customize the plot
        ax.set_yticks(y_positions)
        
        # Create labels with proper bold formatting for headers
        labels = []
        for data in preference_data:
            if data['method'] == 'header':
                # Make headers bold using matplotlib's weight parameter
                labels.append(data['label'])
            else:
                labels.append(data['label'])
        
        ax.set_yticklabels(labels)
        
        # Apply bold formatting specifically to header tick labels
        for i, data in enumerate(preference_data):
            if data['method'] == 'header':
                ax.get_yticklabels()[i].set_weight('bold')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Lane Preference')
        ax.set_title(f'Lane Preference Analysis - {model} {env}')
        
        # Add visual grouping for profiles
        # Find boundaries between different profiles
        current_profile = None
        for i, data in enumerate(preference_data):
            if current_profile != data['profile']:
                if current_profile is not None:
                    # Add separator line between profile groups
                    ax.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=1, alpha=0.3)
                current_profile = data['profile']
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=left_color, alpha=0.7, label='Left Lane Preference'),
            Patch(facecolor=right_color, alpha=0.7, label='Right Lane Preference'),
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout and save
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f'lane_preference_{model}_{env}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Lane preference plot saved to: {plot_file}")


def method_sort_key(config):
    method = config.get('method', '').lower()
    value = config.get('value', 0)
    # Unsupervised first
    if method == 'unsupervised':
        return (0, 0, 0)
    # Filter Only second
    elif method == 'filter_only':
        return (1, 0, 0)
    # Naive Augment third
    elif method == 'naive_augment':
        return (2, 0, 0)
    # Adaptive next, sorted by value
    elif method == 'adaptive':
        return (3, 0, float(value) if value is not None else 0)
    # Fixed next, sorted by value
    elif method == 'fixed':
        return (4, 0, float(value) if value is not None else 0)
    # Default fallback
    else:
        return (99, 0, 0)


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
    summary_tables, details_tables = generate_markdown_tables(grouped_data)
    
    # Generate summary table
    print("Generating summary table...")
    summary_rows = generate_summary_table(grouped_data)
    
    # Generate lane preference plot
    print("Generating lane preference plot...")
    generate_lane_preference_plot(grouped_data, args.output_dir)
    
    # Write summary.md (without violation rates)
    summary_file = os.path.join(args.output_dir, 'summary.md')
    
    # Write overview (overwrites any existing file)
    with open(summary_file, 'w') as f:
        f.write("# Experiment Results Summary\n")
    
    # Write each summary table
    for table_data in summary_tables:
        model = table_data['model']
        env = table_data['env']
        header_cols = table_data['header_cols']
        rows = table_data['rows']
        
        with open(summary_file, 'a') as f:
            f.write(f"\n## {model} {env} Results\n\n")
            f.write("Cost rates are normalized by episode length multiplied by 100 "
                    "for convenience, and lane times are given as a proportion of the total episode "
                    "length. For each metric, the mean and standard deviation between experiments is "
                    "given in the format \"mean (std)\". \n\n")
            f.write("| " + " | ".join(header_cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(header_cols)) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
    
    # Write summary table
    with open(summary_file, 'a') as f:
        f.write("\n## Experiment Summary\n\n")
        f.write("| Profile | Model-Environment | Method | Experiments | Total Episodes |\n")
        f.write("|---------|-------------------|--------|-------------|----------------|\n")
        for row in summary_rows:
            f.write("| " + " | ".join(row) + " |\n")
    
    # Write details.md (only violation rates)
    details_file = os.path.join(args.output_dir, 'details.md')
    
    # Write overview (overwrites any existing file)
    with open(details_file, 'w') as f:
        f.write("# Norm Violation Details\n")
    
    # Write each details table
    for table_data in details_tables:
        model = table_data['model']
        env = table_data['env']
        header_cols = table_data['header_cols']
        rows = table_data['rows']
        
        with open(details_file, 'a') as f:
            f.write(f"\n## {model} {env} Violation Rates\n\n")
            f.write("Violation rates are normalized by episode length multiplied by 100 "
                    "for convenience. For each metric, the mean and standard deviation between experiments is "
                    "given in the format \"mean (std)\". \n\n")
            f.write("| " + " | ".join(header_cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(header_cols)) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
    
    print(f"Analysis complete! Results written to:")
    print(f"  Summary: {summary_file}")
    print(f"  Details: {details_file}")


if __name__ == '__main__':
    main()
