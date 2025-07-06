# Metrics‑Driven Normative Supervision for Safe Reinforcement Learning in Autonomous Driving

This project implements a metrics-driven normative supervisor, which augments the actions of a pretrained DQN agent to enforce driving norms and safety constraints. The agent is trained and tested in the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv/tree/master) simulation environment using the configurations under the `configs` directory.

## Getting Started

After cloning the repository, set up a virtual environment and install the required packages:
 
     virtualenv venv
     source venv/bin/activate
     pip install -e .

## Usage

- To run one of our pretrained agents without normative supervision, execute the `run.py` script.

- To reproduce the nominal and zero-shot results from our paper, including the normative supervisor, run the `experiment_run.py` script and select your desired model and scenario configuration when prompted.

- Finally, to run the adversarial scenario, use the `adversarial_test.py` script.

- Plots can be generated using the `analysis.py` script.

## Results

You can specify a directory for the results when you run the experiment scripts. The results generated for our paper are in the `results` directory. Generated plots will be written to the `results/plots` directory by default.


## License

Released under the MIT License.
