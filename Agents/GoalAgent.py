from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

import numpy as np

StateType = TypeVar("StateType")
GoalType = TypeVar("GoalType")


class GoalAgent(ABC, Generic[StateType, GoalType]):
    @abstractmethod
    def act(self, state: StateType, goal: GoalType) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Select an action given the current state and goal.

        Returns:
            (action, predicted_state) where predicted_state is the agent's
            estimate of where the action sequence will lead (may be None).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset any episode-level state. Called at the start of each episode."""
        pass
