# MAD — Minimum Action Distance

MAD is a framework for learning Minimum Action Distance functions in continuous control environments. It supports four distance metric types and two planning modes, all evaluated on [OGBench](https://github.com/seohongpark/ogbench) environments (AntMaze, PointMaze).

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd MinimumActionDistance

# Install in editable mode (recommended)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

**Dependencies:** `torch~=2.6`, `gymnasium~=1.0`, `wandb~=0.19`, `torchqmet~=0.1`, `scipy`, `python-dotenv`

## Setup

Copy the environment file and fill in your Weights & Biases entity:

```bash
cp .env.example .env
# Edit .env and set WANDB_ENTITY=your_wandb_username
```

## Distance Metrics

| Key | Class | Description |
|-----|-------|-------------|
| `mad` | `MadDist` | MAD quasimetric distance |
| `tdmad` | `TDMadDist` | Temporal-difference MAD |
| `hilbert` | `HilbertDistance` | Hilbert representation distance |
| `qrl` | `TorchScriptDistanceWrapper` | QRL model (trained from `https://github.com/quasimetric-learning/quasimetric-rl`) |

## Running Experiments

All experiments use a config file: a Python module with a `CONFIG` dict, an `ENVIRONMENTS` or `RUNS` list, and a `SEEDS` list.

### Train a distance metric

```bash
python Runs/eval_distance.py Runs/configs/dist_mad.py
python Runs/eval_distance.py Runs/configs/dist_mador.py
python Runs/eval_distance.py Runs/configs/dist_tdmad.py
python Runs/eval_distance.py Runs/configs/dist_hilbert.py
```

### Evaluate planning

Requires a pre-trained model in `trained_models/`.

```bash
# Random-sampling planner (OGBench)
python Runs/eval_planning.py Runs/configs/plan_ogbench_sampling_mad.py

# Dataset-guided planner (OGBench)
python Runs/eval_planning.py Runs/configs/plan_ogbench_dataset_mad.py
```

## Tests

Smoke tests cover the full training and planning pipelines using a lightweight environment and a stub distance model (no GPU or pre-trained weights required).

```bash
python -m unittest discover -s UnitTests -v
```

Each test loads a real config, overrides it to run for a single step/episode, and verifies the pipeline completes without error.

## Model Checkpoints

Trained models are saved to:
```
trained_models/{exp_name}_{env_name}_seed_{seed}.pt
```

The `trained_models/` directory is excluded from git (see `.gitignore`). Pre-trained checkpoints will be released separately.

## Project Structure

```
Distances/       # Distance metric implementations (MadDist, TDMadDist, HilbertDist)
Models/          # Neural network models (encoders, distance heads)
Agents/          # Planning agents (PlanningAgent, DatasetGuidedPlanningAgent)
Planning/        # Environment wrappers for forward simulation
Envs/            # Gymnasium / OGBench environment wrappers
DistExpReplay/   # Experience replay buffer
Runs/            # Entry-point scripts and config files
Utils/           # Shared utilities (metrics, logging, plotting)
UnitTests/       # Smoke tests for training and planning pipelines
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{steccanella2025learning,
  title={Learning the minimum action distance},
  author={Steccanella, Lorenzo and Evans, Joshua B and {\c{S}}im{\c{s}}ek, {\"O}zg{\"u}r and Jonsson, Anders},
  journal={arXiv preprint arXiv:2506.09276},
  year={2025}
}
```

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
