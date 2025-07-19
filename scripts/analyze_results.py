#!/usr/bin/env python3
"""
Analysis script for norm supervisor experiment results.
Scans CSV files in results directory and generates summary statistics.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
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
    """Parse configuration from filename format: profile/method/model_env_value.csv"""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    
    model = parts[0]
    env = parts[1]
    value = None
    if len(parts) > 2:
        try:
            value = float(parts[2])
        except ValueError:
            value = None
    
    # Get method from parent directory
    method = os.path.basename(os.path.dirname(filename))
    # Remove _filtered/_unfiltered suffix for label purposes
    if method.endswith('_filtered'):
        method_base = method[:-9]
        filtered = True
    elif method.endswith('_unfiltered'):
        method_base = method[:-11]
        filtered = False
    else:
        method_base = method
        filtered = False
    
    # Get profile from grandparent directory
    profile = os.path.basename(os.path.dirname(os.path.dirname(filename)))
    
    return {
        'profile': profile,
        'method': method_base,
        'filtered': filtered,
        'value': value,
        'model': model,
        'env': env,
        'mode': method_base if method_base in ['default', 'naive', 'nop'] else 'default',
        'filename': filename
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


def format_statistic(mean_val, std_val, n_experiments, metric_name=None):
    """Format mean and standard error as 'mean ± SE'."""
    if pd.isna(mean_val) or pd.isna(std_val) or n_experiments <= 1:
        return "-"
    
    # Calculate standard error
    se_val = std_val / np.sqrt(n_experiments)
    return f"{mean_val:.2f} ± {se_val:.2f}"


def format_median_iqr(median_val, q1_val, q3_val, iqr_val):
    """Format median and IQR as "median (Q1Q3)"."""
    if pd.isna(median_val) or pd.isna(q1_val) or pd.isna(q3_val) or pd.isna(iqr_val):
        return "-"
    
    return f"{median_val:.2f} ({q1_val:.2f} - {q3_val:.2f})"


def format_collision_rate(group_data):
    """Format collision rate per hour."""
    total_distance = 0
    total_collisions = 0
    total_time_seconds = 0
    policy_period = 1  # TODO: Use environment config
    
    for df in group_data:
        if (
            'total_collisions' in df.columns
            and 'mean_speed' in df.columns
            and 'mean_episode_length' in df.columns
            and 'num_episodes' in df.columns
        ):
            for idx, row in df.iterrows():
                collisions = row.get('total_collisions', 0)
                speed = row.get('mean_speed', np.nan)
                ep_length = row.get('mean_episode_length', np.nan)
                num_episodes = row.get('num_episodes', np.nan)
                if not np.isnan(speed) and not np.isnan(ep_length) and not np.isnan(num_episodes):
                    distance = speed * ep_length * policy_period * num_episodes
                    total_distance += distance
                    total_collisions += collisions
                    total_time_seconds += ep_length * policy_period * num_episodes
    
    if total_collisions == 0:
        return "0.00"
    else:
        # Calculate collision rate per hour
        total_time_hours = total_time_seconds / 3600
        collision_rate = total_collisions / total_time_hours
        # NOTE: We can model the collision data as a binomial distribution where each episode is a
        # trial that either ends with success (no collision) or failure (collision).
        n = sum([df['num_episodes'].sum() for df in group_data]) # n  = total number of trials
        p = total_collisions / n                                 # p  = probability of a collision
        se_p = np.sqrt(p * (1 - p) / n)                          # SE = sqrt(n*p*(1-p)) / n
        # NOTE: We can propagate the standard error to a linear function of p:
        # f(p) = (p * 3600) / (ep_length * policy_period)
        # |df/dp| = 3600 / (ep_length * policy_period)
        # se_f = |df/dp| * se_p = (3600 * se_p) / (ep_length * policy_period)
        mean_episode_length = total_time_seconds / n if n > 0 else 1
        se_collision_rate = (3600 * se_p) / (mean_episode_length * policy_period)
        return f"{collision_rate:.2f} ± {se_collision_rate:.2f}"


def get_lane_columns(stats):
    """Get lane time columns from statistics."""
    return sorted([col for col in stats.keys() if col.startswith('lane_') and col.endswith('_preference')])


def compute_success_rate(group_data):
    """Compute success rate as percentage of episodes without collision."""
    total_collisions = 0
    total_episodes = 0
    for df in group_data:
        if 'total_collisions' in df.columns and 'num_episodes' in df.columns:
            total_collisions += df['total_collisions'].sum()
            total_episodes += df['num_episodes'].sum()
    if total_episodes == 0:
        return "-"
    success_rate = 100 * (total_episodes - total_collisions) / total_episodes
    return f"{success_rate:.2f}"


def create_config_name(config_dict):
    """Create a readable configuration name."""
    if config_dict['method'] in ['adaptive', 'fixed'] and config_dict['value'] is not None:
        # Format value as power of 10
        value = config_dict['value']
        if value == 0.01:
            value_str = "10⁻²"
        elif value == 0.0316:
            value_str = "10⁻¹·⁵"
        elif value == 0.10:
            value_str = "10⁻¹"
        elif value == 0.3162:
            value_str = "10⁻⁰·⁵"
        elif value == 1.00:
            value_str = "10⁰"
        else:
            value_str = str(value)
        return f"{config_dict['method'].title()} ({value_str})"
    elif config_dict['method'] == 'naive':
        return f"{config_dict['method'].title()}"
    else:
        return f"{config_dict['method'].title()}"


def method_sort_key(config):
    """Sort key for methods."""
    method = config.get('method', '').lower()
    value = config.get('value', 0)
    # Unsupervised first
    if method == 'unsupervised':
        return (0, 0, 0)
    # Filter Only second
    elif method == 'filter_only':
        return (1, 0, 0)
    # Naive third
    elif method == 'naive':
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


def generate_markdown_tables(grouped_data):
    """Generate markdown tables from grouped data, one per model-environment combination."""
    
    # Define metric categories and their display names
    summary_metric_categories = {
        'Episode Metrics': ['mean_episode_length', 'collision_rate'],
        'Safety Metrics': ['mean_speed'],
        'Cost Metrics': ['cost_rate', 'avoided_cost_rate'],
        'Lane Usage': []
    }
    
    details_metric_categories = {
        'Violation Rates': [
            'speed_violation_rate', 'tailgating_violation_rate',
            'braking_violation_rate', 'lane_keeping_violation_rate',
            'lane_change_tailgating_violation_rate', 'lane_change_braking_violation_rate',
            'collision_violation_rate', 'lane_change_collision_violation_rate'
        ],
        'Cost Metrics': ['cost_rate', 'avoided_cost_rate']
    }
    
    # Create display names mapping
    display_names = {
        # Summary metrics
        'mean_episode_length'    : 'Episode Length (s)',
        'collision_rate'         : 'Collision Rate (hr⁻¹)',
        'mean_speed'             : 'Speed (m/s)',
        'tet_2s'                 : 'TET 2s (%)',
        'tet_3s'                 : 'TET 3s (%)',
        'teud_2v'                : 'TEUD 2v (%)',
        'teud_3v'                : 'TEUD 3v (%)',
        'cost_rate'              : 'Cost Rate (hr⁻¹)',
        'avoided_cost_rate'      : 'Avoided Cost Rate (hr⁻¹)',
        
        # Details metrics
        'speed_violation_rate'                 : 'Speed Violations (hr⁻¹)',
        'tailgating_violation_rate'            : 'Tailgating Violations (hr⁻¹)',
        'braking_violation_rate'               : 'Braking Violations (hr⁻¹)',
        'lane_keeping_violation_rate'          : 'LaneKeeping Violations (hr⁻¹)',
        'lane_change_tailgating_violation_rate': 'Lane Change Tailgating Violations (hr⁻¹)',
        'lane_change_braking_violation_rate'   : 'Lane Change Braking Violations (hr⁻¹)',
        'collision_violation_rate'             : 'Collision Violations (hr⁻¹)',
        'lane_change_collision_violation_rate' : 'Lane Change Collision Violations (hr⁻¹)',
        
        # Lane usage metrics
        'lane_0_preference': 'Left Lane Preference (%)',
        'lane_1_preference': 'Right Lane Preference (%)'
    }
    
    # Add lane usage metrics to display names
    all_lane_cols = set()
    for group_data in grouped_data.values():
        stats, _ = calculate_statistics(group_data)
        lane_cols = get_lane_columns(stats)
        all_lane_cols.update(lane_cols)
    
    summary_metric_categories['Lane Usage'] = sorted(all_lane_cols)

    # Build header rows
    summary_header_cols = ['Method']
    for category, metrics in summary_metric_categories.items():
        for metric in metrics:
            col_name = display_names.get(metric, metric)
            summary_header_cols.append(col_name)

    details_header_cols = ['Method']
    for category, metrics in details_metric_categories.items():
        for metric in metrics:
            col_name = display_names.get(metric, metric)
            details_header_cols.append(col_name)
    
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
        summary_configs = []
        details_rows = []
        
        for profile, configs in sorted(profile_configs.items()):
            # Add separator row if not the first profile
            if summary_rows:
                summary_rows.append([''] * len(summary_header_cols))
                summary_configs.append(None)
                details_rows.append([''] * len(details_header_cols))
            
            # Add section header
            section_header = f"**{profile.title()} Profile**"
            summary_separator_row = [section_header] + [''] * (len(summary_header_cols) - 1)
            details_separator_row = [section_header] + [''] * (len(details_header_cols) - 1)
            summary_rows.append(summary_separator_row)
            summary_configs.append(None)
            details_rows.append(details_separator_row)
            
            # Sort configs by method_sort_key
            sorted_configs = sorted(configs.items(), key=lambda item: method_sort_key(dict(item[0])))
            
            for config_tuple, group_data in sorted_configs:
                config_dict = dict(config_tuple)
                stats, n_experiments = calculate_statistics(group_data)
                config_name = create_config_name(config_dict)
                
                # Build summary data row
                summary_row = [config_name]
                for category, metrics in summary_metric_categories.items():
                    for metric in metrics:
                        if metric == 'collision_rate':
                            summary_row.append(format_collision_rate(group_data))
                        elif metric in ['tet_2s', 'tet_3s', 'teud_2v', 'teud_3v', 'lane_0_preference', 'lane_1_preference']:
                            if metric in stats:
                                mean_val, std_val = stats[metric]
                                mean_val *= 100
                                std_val *= 100
                                summary_row.append(format_statistic(mean_val, std_val, n_experiments, metric))
                            else:
                                summary_row.append("-")
                        elif metric in stats:
                            mean_val, std_val = stats[metric]
                            if metric in ['cost_rate', 'avoided_cost_rate']:
                                mean_val *= 3600
                                std_val *= 3600
                            summary_row.append(format_statistic(mean_val, std_val, n_experiments, metric))
                        else:
                            summary_row.append("-")
                
                summary_rows.append(summary_row)
                summary_configs.append(config_dict)
                
                # Build details data row
                details_row = [config_name]
                for category, metrics in details_metric_categories.items():
                    for metric in metrics:
                        if metric in stats:
                            mean_val, std_val = stats[metric]
                            if 'violation_rate' in metric or 'cost_rate' in metric:
                                mean_val *= 3600
                                std_val *= 3600
                            details_row.append(format_statistic(mean_val, std_val, n_experiments, metric))
                        else:
                            details_row.append("-")
                details_rows.append(details_row)
        
        summary_tables.append({
            'model': model,
            'env': env,
            'header_cols': summary_header_cols,
            'rows': summary_rows,
            'configs': summary_configs
        })
        details_tables.append({
            'model': model,
            'env': env,
            'header_cols': details_header_cols,
            'rows': details_rows,
            'configs': summary_configs
        })
    
    return summary_tables, details_tables


def generate_summary_table(grouped_data):
    """Generate a summary table showing experiment counts and total episodes for each configuration."""
    summary_rows = []
    seen_keys = set()
    
    # Collect all unique configurations
    all_configs = []
    for config_tuple, group_data in grouped_data.items():
        config_dict = dict(config_tuple)
        key = (config_dict['profile'], config_dict['model'], config_dict['env'], 
               config_dict['method'], config_dict['value'], config_dict['filtered'])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        # Only include in summary if it's a main result (not ablation)
        if config_dict['method'] in ['unsupervised', 'filter_only', 'nop'] or config_dict['filtered'] == True:
            all_configs.append(config_dict)
    
    # Sort configurations
    all_configs.sort(key=lambda x: (x['profile'], x['model'], x['env'], x['method'], x['value'] or 0))
    
    for config in all_configs:
        config_tuple = tuple(sorted(config.items()))
        if config_tuple in grouped_data:
            group_data = grouped_data[config_tuple]
            stats, n_experiments = calculate_statistics(group_data)
            
            # Calculate total episodes
            total_episodes = 0
            for df in group_data:
                if 'num_episodes' in df.columns:
                    total_episodes += df['num_episodes'].sum()
            
            # Calculate success rate
            success_rate = compute_success_rate(group_data)
            method_name = create_config_name(config)
            
            # Create row
            row = [
                config['profile'].title(),
                f"{config['model']} {config['env']}",
                method_name,
                str(n_experiments),
                str(total_episodes),
                success_rate
            ]
            summary_rows.append(row)
    
    # Reverse the order so most interesting results appear at the top
    return summary_rows[::-1]


def write_table_section(f, title, header_cols, rows):
    """Write a table section to the file."""
    f.write(f"\n### {title}\n\n")
    f.write("| " + " | ".join(header_cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(header_cols)) + "|\n")
    for row in rows:
        f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")


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
    
    for (model, env), profile_configs in sorted(model_env_groups.items()):
        # Collect lane preference data for this model-environment
        preference_data = []
        seen_baselines = set()
        profile_groups = defaultdict(list)
        
        for profile, configs in profile_configs.items():
            for config_tuple, group_data in configs.items():
                config_dict = dict(config_tuple)
                # Skip ablation studies
                if config_dict['method'] not in ['unsupervised', 'filter_only'] and not config_dict['filtered']:
                    continue

                stats, n_experiments = calculate_statistics(group_data)
                lane_1_key = 'lane_1_preference'
                if lane_1_key in stats:
                    mean_val, std_val = stats[lane_1_key]
                    method_name = create_config_name(config_dict)
                    
                    # For baseline methods, only include one version
                    if config_dict['method'] in ['unsupervised', 'filter_only']:
                        baseline_key = f"{config_dict['method']}"
                        if baseline_key in seen_baselines:
                            continue
                        seen_baselines.add(baseline_key)
                        label = method_name
                    else:
                        label = method_name
                    
                    profile_groups[profile].append({
                        'label': label,
                        'mean': mean_val,
                        'std': std_val,
                        'profile': config_dict['profile'],
                        'method': method_name,
                        'original_method': config_dict['method'],
                        'original_value': config_dict.get('value', 0),
                        'n_experiments': n_experiments
                    })
        
        if not profile_groups:
            print(f"No lane preference data found for {model} {env}. Skipping plot generation.")
            continue
        
        # Build final data with headers in correct positions
        preference_data = []
        
        # Sort data within each profile group
        for profile in profile_groups:
            def profile_sort_key(data):
                method = data['original_method']
                value = data['original_value']
                if method in ['unsupervised', 'filter_only']:
                    return (999, 0, 0)
                elif method == 'naive':
                    return (0, 0, 0)
                elif method == 'adaptive':
                    return (1, 0, value or 0)
                else:  # fixed
                    return (2, 0, value or 0)
            profile_groups[profile].sort(key=profile_sort_key)
        
        # Add baselines first
        baseline_data = []
        for profile in profile_groups:
            for data in profile_groups[profile]:
                if data['original_method'] in ['unsupervised', 'filter_only']:
                    data['profile'] = 'no_profile'
                    baseline_data.append(data)
        
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
                'original_value': 0,
                'n_experiments': 1
            }
            preference_data.append(header_data)
        preference_data.extend(baseline_data)
        
        # Add cautious profile with header
        if 'cautious' in profile_groups:
            header_data = {
                'label': 'Cautious Profile',
                'mean': np.nan,
                'std': np.nan,
                'profile': 'cautious',
                'method': 'header',
                'original_method': 'header',
                'original_value': 0,
                'n_experiments': 1
            }
            preference_data.append(header_data)
            cautious_data = [data for data in profile_groups['cautious'] 
                           if data['original_method'] not in ['unsupervised', 'filter_only']]
            preference_data.extend(cautious_data)
        
        # Add efficient profile with header
        if 'efficient' in profile_groups:
            header_data = {
                'label': 'Efficient Profile',
                'mean': np.nan,
                'std': np.nan,
                'profile': 'efficient',
                'method': 'header',
                'original_method': 'header',
                'original_value': 0,
                'n_experiments': 1
            }
            preference_data.append(header_data)
            efficient_data = [data for data in profile_groups['efficient'] 
                           if data['original_method'] not in ['unsupervised', 'filter_only']]
            preference_data.extend(efficient_data)
        
        if not preference_data:
            print(f"No lane preference data found for {model} {env}. Skipping plot generation.")
            continue
        
        # Invert the order so methods appear from bottom to top
        preference_data.reverse()
        
        # Identify y-index ranges for each profile group
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
        
        # Shade profile regions
        profile_colors = {'cautious': '#e6f2ff', 'efficient': '#fff7e6', 'no_profile': '#f0f0f0'}
        for profile, (ymin, ymax) in group_ranges.items():
            if profile in profile_colors:
                ax.axhspan(ymin-0.5, ymax+0.5, facecolor=profile_colors[profile], alpha=0.4, zorder=0)
        
        # Set up the plot
        y_positions = np.arange(len(preference_data))
        center_line = 0.0
        left_color = '#ff7f0e'
        right_color = '#1f77b4'
        
        # Plot bars
        for i, data in enumerate(preference_data):
            mean_val = data['mean']
            if np.isnan(mean_val) or data['method'] == 'header':
                continue
            
            # Convert lane preference from [0,1] to [-1,1] scale
            scaled_mean = (mean_val - 0.5) * 2
            if not np.isnan(data['std']):
                n_experiments = data.get('n_experiments', 1)
                se_val = data['std'] / np.sqrt(n_experiments)
                scaled_se = se_val * 2
            else:
                scaled_se = np.nan
            
            # Determine bar direction and color
            if scaled_mean < center_line:
                bar_width = abs(scaled_mean)
                bar_start = scaled_mean
                color = left_color
            else:
                bar_width = scaled_mean
                bar_start = center_line
                color = right_color
            
            # Plot the bar
            ax.barh(i, bar_width, left=bar_start, height=0.6, color=color, alpha=0.7)
            
            # Add error bars if std is available
            if not np.isnan(scaled_se):
                if scaled_mean < center_line:
                    error_x = [max(-1, scaled_mean - scaled_se), scaled_mean + scaled_se]
                else:
                    error_x = [scaled_mean - scaled_se, min(1, scaled_mean + scaled_se)]
                error_y = [i, i]
                ax.plot(error_x, error_y, 'k-', linewidth=2, alpha=0.8)
                ax.plot([error_x[0], error_x[0]], [i-0.1, i+0.1], 'k-', linewidth=2, alpha=0.8)
                ax.plot([error_x[1], error_x[1]], [i-0.1, i+0.1], 'k-', linewidth=2, alpha=0.8)
        
        # Add center line
        ax.axvline(x=center_line, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        # Customize the plot
        ax.set_yticks(y_positions)
        labels = []
        for data in preference_data:
            if data['method'] == 'header':
                label = data['label']
            else:
                label = data['label']
            label = label.replace('_', '-')
            labels.append(label)
        ax.set_yticklabels(labels)
        
        # Apply bold formatting to header tick labels
        for i, data in enumerate(preference_data):
            if data['method'] == 'header':
                ax.get_yticklabels()[i].set_weight('bold')
        
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Lane Preference')
        ax.set_title(f'Lane Preference Analysis - {model} {env}')
        
        # Add visual grouping for profiles
        current_profile = None
        for i, data in enumerate(preference_data):
            if current_profile != data['profile']:
                if current_profile is not None:
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


def generate_adaptive_trend_plot(grouped_data, output_dir, adaptive_values):
    """Generate line plots showing how metrics evolve with adaptive method values."""
    # Create plots subdirectory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Group data by model-environment-profile combination
    model_env_profile_groups = defaultdict(lambda: defaultdict(dict))
    for config_tuple, group_data in grouped_data.items():
        config_dict = dict(config_tuple)
        if config_dict['method'] != 'adaptive' or not config_dict['filtered']:
            continue
        
        model_env_key = (config_dict['model'], config_dict['env'])
        profile = config_dict['profile']
        value = config_dict.get('value')
        
        if value is not None and value in adaptive_values:
            model_env_profile_groups[model_env_key][profile][value] = group_data
    
    # Generate plots for each model-environment combination
    for (model, env), profile_configs in sorted(model_env_profile_groups.items()):
        if 'cautious' not in profile_configs:
            continue
        
        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Only plot cautious profile
        profile = 'cautious'
        profile_configs = profile_configs[profile]
        
        # Plot data for cautious profile
        x_values = []
        y_collision_values = []
        y_cost_values = []
        y_collision_errors = []
        y_cost_errors = []
        
        # Collect data points for adaptive values
        for value in sorted(adaptive_values):
            if value in profile_configs:
                group_data = profile_configs[value]
                stats, n_experiments = calculate_statistics(group_data)
                
                # Get collision rate per hour
                collision_rate_str = format_collision_rate(group_data)
                if collision_rate_str != "0.00":
                    try:
                        # Parse collision rate from format "X.XX ± Y.YY" or "X.XX"
                        if "±" in collision_rate_str:
                            parts = collision_rate_str.split("±")
                            collision_rate = float(parts[0].strip())
                            collision_error = float(parts[1].strip())
                        else:
                            collision_rate = float(collision_rate_str)
                            collision_error = 0  # No error info available
                        y_collision_values.append(collision_rate)
                        x_values.append(value)
                        y_collision_errors.append(collision_error)
                    except ValueError:
                        pass
                
                # Get cost rate (raw, before 3600 multiplication)
                if 'cost_rate' in stats:
                    mean_val, std_val = stats['cost_rate']
                    if not pd.isna(mean_val):
                        # Convert to per-hour units
                        cost_rate = mean_val * 3600
                        y_cost_values.append(cost_rate)
                        if len(x_values) < len(y_cost_values):  # In case collision rate was missing
                            x_values.append(value)
                        # Compute standard error: std / sqrt(n), then convert to per-hour
                        se_val = std_val / np.sqrt(n_experiments) if n_experiments > 1 else 0
                        se_val *= 3600  # Convert error to per-hour units
                        y_cost_errors.append(se_val)
        
        if x_values:  # Only plot if we have data
            # Plot collision rate on top subplot
            if y_collision_values:
                ax1.errorbar(x_values, y_collision_values, yerr=y_collision_errors, 
                           marker='s', color='black', linestyle=':', label='Collision Rate',
                           capsize=3, capthick=1, linewidth=2, markersize=6)
            
            # Plot cost rate on bottom subplot
            if y_cost_values:
                ax2.errorbar(x_values, y_cost_values, yerr=y_cost_errors, 
                           marker='s', color='black', linestyle=':', label='Cost Rate',
                           capsize=3, capthick=1, linewidth=2, markersize=6)
        
        # Customize plots
        ax2.set_xlabel('Adaptive Value', color='black')
        ax1.set_ylabel('Collision Rate (hr⁻¹)', color='black')
        ax2.set_ylabel('Cost Rate (hr⁻¹)', color='black') # Changed to raw cost rate
        
        # Set subplot titles
        ax1.set_title('Collision Rate', fontsize=12, color='black')
        ax2.set_title('Cost Rate', fontsize=12, color='black')
        
        # Set main title
        fig.suptitle(f'Adaptive Method Trends - {model} {env} (Cautious Profile)', fontsize=14)
        
        # Use log scale for x-axis if values span multiple orders of magnitude
        if max(adaptive_values) / min(adaptive_values) > 10:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        
        # Set y-axis limits
        ax1.set_ylim(bottom=0)  # Start from 0 for collision rate
        if y_cost_values:
            max_cost = max(y_cost_values) * 1.1  # Add 10% padding
            ax2.set_ylim(0, max_cost)
        else:
            ax2.set_ylim(0, 1)  # Default if no data
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f'adaptive_trends_{model}_{env}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Adaptive trends plot saved to: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results from CSV files')
    parser.add_argument('--results-dir', default=None, 
                       help='Directory containing CSV result files (default: script_dir/../results)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for analysis files (default: script_dir/../analysis)')
    parser.add_argument('--fixed-values', default='0.01, 0.10, 1.00',
                       help='Comma-separated list of allowed values for fixed method (e.g., 0.01,0.05,0.1)')
    parser.add_argument('--adaptive-values', default='0.01, 0.10, 1.00',
                       help='Comma-separated list of allowed values for adaptive method (e.g., 0.01,0.05,0.1)')
    parser.add_argument('--plot-adaptive-values', default='0.01, 0.0316, 0.10, 0.3162, 1.00, 3.1623, 10.000',
                       help='Comma-separated list of adaptive values to include in trend plots')
    
    args = parser.parse_args()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Set default paths relative to project root
    if args.results_dir is None:
        args.results_dir = os.path.join(project_root, 'results')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'analysis')
    
    # Parse allowed values for fixed and adaptive methods
    def parse_value_list(val):
        if val is None or val.strip() == '':
            return None
        return [float(x) for x in val.split(',') if x.strip() != '']
    allowed_fixed_values = parse_value_list(args.fixed_values)
    allowed_adaptive_values = parse_value_list(args.adaptive_values)
    plot_adaptive_values = parse_value_list(args.plot_adaptive_values)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and group data
    print(f"Scanning for CSV files in {args.results_dir}...")
    grouped_data = load_and_group_data(args.results_dir)
    
    if not grouped_data:
        print("No valid data found. Exiting.")
        return
    
    print(f"Found {len(grouped_data)} configuration groups")
    
    # Filter grouped_data based on allowed values
    def config_is_allowed(config):
        method = config.get('method', '')
        value = config.get('value', None)
        if method == 'fixed' and allowed_fixed_values is not None:
            return value in allowed_fixed_values
        if method == 'adaptive' and allowed_adaptive_values is not None:
            return value in allowed_adaptive_values
        return True  # All other methods always included
    
    filtered_grouped_data = {cfg: data for cfg, data in grouped_data.items() 
                           if config_is_allowed(dict(cfg))}
    
    # Generate markdown tables
    print("Generating markdown tables...")
    summary_tables, details_tables = generate_markdown_tables(filtered_grouped_data)
    
    # Generate summary table
    print("Generating summary table...")
    summary_rows = generate_summary_table(filtered_grouped_data)
    
    # Generate lane preference plot
    print("Generating lane preference plot...")
    generate_lane_preference_plot(filtered_grouped_data, args.output_dir)
    
    # Generate adaptive trends plot
    if plot_adaptive_values:
        print("Generating adaptive trends plot...")
        generate_adaptive_trend_plot(grouped_data, args.output_dir, plot_adaptive_values)
    
    # Write summary.md
    summary_file = os.path.join(args.output_dir, 'summary.md')
    with open(summary_file, 'w') as f:
        f.write("# Experiment Results Summary\n")
    
    # Write each summary table
    for table_data in summary_tables:
        model = table_data['model']
        env = table_data['env']
        header_cols = table_data['header_cols']
        rows = table_data['rows']
        configs = table_data.get('configs', [None] * len(rows))
        
        # Find the index of 'Collision Rate (hr⁻¹)' in the header
        try:
            collision_idx = header_cols.index('Collision Rate (hr⁻¹)')
        except ValueError:
            collision_idx = 1
        
        # Insert 'Success Rate (%)' before collision rate in header
        extended_header_cols = header_cols[:collision_idx] + ['Success Rate (%)'] + header_cols[collision_idx:]
        
        # Split rows into main and ablation
        main_rows = []
        ablation_rows = []
        seen_main_methods = set()
        
        for row, config in zip(rows, configs):
            if config is None:
                main_rows.append(row[:collision_idx] + [''] + row[collision_idx:])
                ablation_rows.append(row[:collision_idx] + [''] + row[collision_idx:])
                continue
            
            method_key = (config['profile'], config['method'], config.get('value', None))
            config_tuple = tuple(sorted(config.items()))
            group_data = filtered_grouped_data.get(config_tuple, None)
            success_rate = compute_success_rate(group_data) if group_data is not None else "-"
            new_row = row[:collision_idx] + [success_rate] + row[collision_idx:]
            
            if config['method'] in ['unsupervised', 'filter_only', 'nop']:
                if method_key not in seen_main_methods:
                    main_rows.append(new_row)
                    seen_main_methods.add(method_key)
            elif config['filtered'] == False:
                ablation_rows.append(new_row)
            else:
                if method_key not in seen_main_methods:
                    main_rows.append(new_row)
                    seen_main_methods.add(method_key)
        
        with open(summary_file, 'a') as f:
            f.write(f"\n## {model} {env} Results\n\n")
            write_table_section(f, "Main Experimental Results", extended_header_cols, main_rows)
            if any(cell for row in ablation_rows for cell in row if cell and not cell[0:2] == '**'):
                write_table_section(f, "Ablation Studies", extended_header_cols, ablation_rows)
    
    # Write summary table
    with open(summary_file, 'a') as f:
        f.write("\n## Experiment Summary\n\n")
        f.write("| Profile | Model-Environment | Method | Experiments | Total Episodes |\n")
        f.write("|---------|-------------------|--------|-------------|----------------|\n")
        for row in summary_rows:
            f.write("| " + " | ".join(row[:5]) + " |\n")

    # Write ablation studies table
    ablation_configs = []
    for config_tuple, group_data in filtered_grouped_data.items():
        config_dict = dict(config_tuple)
        if config_dict.get('method', 'nop') not in ['unsupervised', 'filter_only', 'nop'] and not config_dict.get('filtered', True):
            ablation_configs.append(config_dict)
    
    ablation_configs.sort(key=lambda x: (x['profile'], x['model'], x['env'], x['method'], x['value'] or 0))
    
    if ablation_configs:
        with open(summary_file, 'a') as f:
            f.write("\n## Ablation Studies\n\n")
            f.write("| Profile | Model-Environment | Method | Experiments | Total Episodes |\n")
            f.write("|---------|-------------------|--------|-------------|----------------|\n")
            for config in ablation_configs[::-1]:
                config_tuple = tuple(sorted(config.items()))
                if config_tuple in filtered_grouped_data:
                    group_data = filtered_grouped_data[config_tuple]
                    stats, n_experiments = calculate_statistics(group_data)
                    
                    total_episodes = 0
                    for df in group_data:
                        if 'num_episodes' in df.columns:
                            total_episodes += df['num_episodes'].sum()
                    
                    method_name = create_config_name(config)
                    row = [
                        config['profile'].title(),
                        f"{config['model']} {config['env']}",
                        method_name,
                        str(n_experiments),
                        str(total_episodes)
                    ]
                    f.write("| " + " | ".join(row) + " |\n")
    
    # Write details.md
    details_file = os.path.join(args.output_dir, 'details.md')
    with open(details_file, 'w') as f:
        f.write("# Norm Violation Details\n")
    
    # Write each details table
    for table_data in details_tables:
        model = table_data['model']
        env = table_data['env']
        header_cols = table_data['header_cols']
        rows = table_data['rows']
        configs = table_data.get('configs', [None] * len(rows))
        
        # Split rows into main and ablation
        main_rows = []
        ablation_rows = []
        seen_main_methods = set()
        
        for row, config in zip(rows, configs):
            if config is None:
                main_rows.append(row)
                ablation_rows.append(row)
                continue
            
            method_key = (config['profile'], config['method'], config.get('value', None))
            
            if config['method'] in ['unsupervised', 'filter_only', 'nop']:
                if method_key not in seen_main_methods:
                    main_rows.append(row)
                    seen_main_methods.add(method_key)
            elif config['filtered'] == False:
                ablation_rows.append(row)
            else:
                if method_key not in seen_main_methods:
                    main_rows.append(row)
                    seen_main_methods.add(method_key)
        
        with open(details_file, 'a') as f:
            f.write(f"\n## {model} {env} Violation Rates\n\n")
            f.write("For each metric, the mean and standard error between experiments are given in "
                    "the format \"mean ± SE\". \n\n")
            write_table_section(f, "Main Experimental Results", header_cols, main_rows)
            if any(cell for row in ablation_rows for cell in row if cell and not cell.startswith('**')):
                write_table_section(f, "Ablation Studies", header_cols, ablation_rows)
    
    print(f"Analysis complete! Results written to:")
    print(f"  Summary: {summary_file}")
    print(f"  Details: {details_file}")


if __name__ == '__main__':
    main()
