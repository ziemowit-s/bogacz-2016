from typing import Optional

from actor_uncertainty_2016.au.auaction import AUAction
from core.state import State


class AUState(State):
    def __init__(self, num_of_actions: int = 1, alfa: float = 0.1,
                 beta: Optional[float] = None, actor_only: bool = False):
        super().__init__(num_of_actions=num_of_actions, alfa=alfa, beta=beta, actor_only=actor_only,
                         action_type=AUAction)

    def act(self, a=None, b=None):
        super().act(a=a, b=b)

