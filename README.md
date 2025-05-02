# Norm‑Supervised‑Highway

**Metrics‑Driven Normative Supervision for Safe RL in Autonomous Driving**

## Overview

This project implements a non‑learned supervisory layer around a pretrained DQN agent to enforce driving norms:
- Speed limits  
- Safe following distance  
- Gentle braking  
- Safe lane changes  

Results in the HighwayEnv simulator show 85–90% fewer collisions in training scenarios and robust zero‑shot performance in novel traffic settings.

## Installation

1. Clone the repository: https://github.com/phbui/norm-supervised-highway.git  
2. Create and activate a Python virtual environment  
3. Install dependencies from `requirements.txt`  

## Usage

- **Train without supervision**: run the training script with environment `highway-v0`, DQN algorithm, and desired timesteps  
- **Evaluate with supervision**: run the evaluation script pointing to the trained model and norm configuration  
- **Zero‑shot tests**: supply a custom scenario file to the evaluation script  

## Configuration

Adjust norm thresholds and toggles in `config/default_norms.yml`, including:  
- `speed_limit` (max vehicle speed in m/s)  
- `ttc_threshold` (time‑to‑collision in seconds)  
- `response_time`, `a_max`, `a_min` for safe following distance  

## Outputs

Evaluation logs, metrics (collision counts, avoided vs. unavoided violations), and plots are saved under `experiments/results/`.

## License

Released under the MIT License.
