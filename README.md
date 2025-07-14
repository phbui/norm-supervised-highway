# Normative Supervision for Behavior Preferencing in Reinforcement Learning

This project implements a normative supervisor which augments the actions of a pretrained DQN agent to enforce driving norms and safety constraints. The agent is trained and tested in the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv/tree/master) simulation environment using the configurations under the `configs/` directory.

## Table of Contents
- [1. Getting Started](#1-getting-started)
    - [1.1. Prerequisites](#11-prerequisites)
    - [1.2. Installation](#12-installation)
- [2. Project Structure](#2-project-structure)
- [3. Usage](#3-usage)
    - [3.1. Training Models](#31-training-models)
    - [3.2. Running Experiments](#32-running-experiments)
    - [3.3. Analyzing Results](#33-analyzing-results)
- [4. License](#4-license)

## 1. Getting Started

### 1.1. Prerequisites

- `python >= 3.9`
- `virtualenv` (recommended)

### 1.2. Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install .  # Or for editable installation: "pip install -e ."
```

## 2. Project Structure

```
norm-supervised-highway/
├── norm_supervisor/          # Main package
│   ├── supervisor.py         # Core normative supervisor
│   ├── metrics.py            # Safety metrics calculation
│   ├── consts.py             # Constants and utilities
│   └── norms/                # Norm implementations
│       ├── norms.py          # Driving norm definitions
│       ├── abstract.py       # Abstract norm base classes
│       ├── prediction.py     # Action outcome prediction
│       └── profiles/         # Driving behavior profiles
├── scripts/                  # Main execution scripts
│   ├── train.py              # Train DQN models
│   ├── test.py               # Run experiments with supervision
│   └── analyze_results.py    # Analyze experiment results
├── scripts/old/              # Legacy scripts (for reference)
├── configs/                  # Configuration files
│   ├── training/             # Training configurations
│   └── environment/          # Environment configurations
├── models/                   # Trained model storage
├── results/                  # Experiment results
└── pyproject.toml            # Package configuration
```

## 3. Usage

### 3.1. Training Models

```bash
python scripts/train.py
```

Select training and environment configurations when prompted.

### 3.2. Running Experiments

```bash
python scripts/test.py
```

Choose models, environment configurations, and supervision parameters. Results of the test are written to `results/` by default. Use `--help` to read all of the command-line options.

### 3.3. Analyzing Results

```bash
python scripts/analyze_results.py
```

Processes CSV files in `results/` to generate summary statistics, which are written to `analysis/` by default. Use `--help` to read all of the command-line options.

## 4. License

Released under the MIT License.
