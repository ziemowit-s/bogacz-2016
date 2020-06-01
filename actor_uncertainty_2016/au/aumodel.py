from typing import Optional

from core.model import Model


class AUModel(Model):
    def __init__(self, num_of_states: int = 1, num_of_actions: int = 1, alfa: float = 0.1,
                 beta: Optional[float] = None, with_critic: bool = False):
        """
        Actor Uncertainty model
        """
        super().__init__(num_of_states=num_of_states, num_of_actions=num_of_actions, alfa=alfa,
                         beta=beta, with_critic=with_critic)

    def act(self, state: int = 0, a: float = 1, b: float = 1):
        super().act(state=state, a=a, b=b)
