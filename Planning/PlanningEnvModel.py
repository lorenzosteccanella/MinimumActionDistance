from gymnasium import Env
from typing import Generic, TypeVar
from copy import deepcopy

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class PlanningEnvModel(Generic[StateType, ActionType]):
    def __init__(self, env: Env):
        """
        Initialize the PlanningEnvModel.

        Args:
            sim_env: A simulation environment of type gymnasium.Env.
        """
        self.sim_env = deepcopy(env)

    def predict(self, state: StateType, action: ActionType) -> tuple[StateType, float]:
        """
        Predict the next state and reward given the current state and action.

        Args:
            state: The current state.
            action: The action to be taken.

        Returns:
            A tuple containing the next state and the reward.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
