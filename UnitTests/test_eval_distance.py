"""Smoke tests for eval_distance.py.

Each test loads a dist_*.py config, overrides it to run for a single gradient
step on a single lightweight environment, and verifies that the pipeline
completes without error and that wandb.log is called.

Run with:
    python -m pytest UnitTests/test_eval_distance.py -v
"""
import copy
import unittest
from unittest.mock import patch, MagicMock

from dotenv import load_dotenv
load_dotenv()

from Envs.GridWorlds.EmptyGridWorld import EmptyGridWorld
from Runs.eval_distance import evaluate
from Utils import load_config

_SMOKE_ENV = [("EmptyGridWorld", lambda: EmptyGridWorld())]
_SMOKE_SEEDS = [0]


def _smoke_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["gradient_steps"] = 1
    cfg["track"] = False
    cfg["debug"] = False
    return cfg


class TestEvalDistanceSmoke(unittest.TestCase):

    def _run(self, config_path: str):
        mod = load_config(config_path)
        config = _smoke_config(mod.CONFIG)

        mock_run = MagicMock()
        mock_run.summary = {}

        with patch("wandb.init"), \
             patch("wandb.finish"), \
             patch("wandb.log") as mock_log, \
             patch("wandb.define_metric"), \
             patch("wandb.run", mock_run), \
             patch("wandb.Html", side_effect=lambda x: x):
            evaluate(_SMOKE_SEEDS, _SMOKE_ENV, config)

        self.assertTrue(mock_log.called, "wandb.log was never called — metrics not logged")

    def test_dist_mad(self):
        self._run("Runs/configs/dist_mad.py")

    def test_dist_tdmad(self):
        self._run("Runs/configs/dist_tdmad.py")

    def test_dist_hilbert(self):
        self._run("Runs/configs/dist_hilbert.py")

    def test_dist_mador(self):
        self._run("Runs/configs/dist_mador.py")


if __name__ == "__main__":
    unittest.main()
