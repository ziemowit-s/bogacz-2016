from typing import Optional

from au.austate import AUState


class AUModel:
    def __init__(self, num_of_states: int = 1, num_of_actions: int = 1, alfa: float = 0.1,
                 beta: Optional[float] = None, actor_only: bool = False):
        """
        Actor Uncertainty model
        """
        self.states = []
        for _ in range(num_of_states):
            state = AUState(num_of_actions=num_of_actions, alfa=alfa, beta=beta,
                            actor_only=actor_only)
            self.states.append(state)

    def reward(self, reward, state: int = 0, action: int = 0):
        self.states[state].reward(reward=reward, action=action)

    def act(self, state: int = 0, a: float = 1, b: float = 1):
        probas = self.states[state].act(a=a, b=b)
        return probas