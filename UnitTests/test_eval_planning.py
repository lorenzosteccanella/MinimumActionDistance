"""Smoke tests for eval_planning.py.

Each test loads a plan_*.py config, overrides it to run a single episode on
a single environment/seed, and verifies the pipeline completes without error
and that wandb.log is called.

Pre-trained model files are NOT required — the distance is replaced with a
stub that returns random distances so the test exercises the full agent/env
loop without needing real weights.

Run with:
    python -m pytest UnitTests/test_eval_planning.py -v
"""
import copy
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

from Runs.eval_planning import run_evaluation
from Utils import load_config


def _smoke_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["track"] = False
    cfg["render_mode"] = None  # disable cv2 rendering in headless tests
    cfg["num_eval_episodes"] = 1
    cfg["num_samples"] = 4
    cfg["lookahead"] = 2
    cfg["num_cores"] = 1
    cfg["max_episode_steps"] = 5
    return cfg


class _StubDistance:
    """Returns random distances so planning can run without real weights."""

    class _StubModel:
        def eval(self): return self
        def to(self, device): return self

    model = _StubModel()

    def eval_embed_state(self, s: torch.Tensor) -> torch.Tensor:
        return torch.rand(s.shape[0], 16)

    def eval_z_dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return torch.rand(z1.shape[0])

    def eval_dist(self, s1, s2) -> torch.Tensor:
        if isinstance(s1, np.ndarray):
            s1 = torch.from_numpy(s1).float()
        n = s1.shape[0] if s1.dim() > 1 else 1
        return torch.rand(n)


class TestEvalPlanningSmoke(unittest.TestCase):

    def _run(self, config_path: str):
        mod = load_config(config_path)
        config = _smoke_config({**mod.CONFIG, **mod.RUNS[0], "seed": mod.SEEDS[0]})

        mock_run = MagicMock()
        mock_run.summary = {}
        mock_summary = {}

        with patch("Runs.eval_planning._load_distance", return_value=_StubDistance()), \
             patch("wandb.log") as mock_log, \
             patch("wandb.init"), \
             patch("wandb.finish"), \
             patch("wandb.run", mock_run), \
             patch("wandb.summary", mock_summary, create=True):
            run_evaluation(config)

        self.assertTrue(mock_log.called, "wandb.log was never called — metrics not logged")

    def test_plan_ogbench_sampling_mad(self):
        self._run("Runs/configs/plan_ogbench_sampling_mad.py")

    def test_plan_ogbench_dataset_mad(self):
        self._run("Runs/configs/plan_ogbench_dataset_mad.py")

    def test_plan_ogbench_dataset_tdmad(self):
        self._run("Runs/configs/plan_ogbench_dataset_tdmad.py")

    def test_plan_ogbench_sampling_tdmad(self):
        self._run("Runs/configs/plan_ogbench_sampling_tdmad.py")

    def test_plan_ogbench_dataset_hilbert(self):
        self._run("Runs/configs/plan_ogbench_dataset_hilbert.py")


if __name__ == "__main__":
    unittest.main()
