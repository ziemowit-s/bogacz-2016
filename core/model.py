from typing import Optional

from actor_uncertainty_2016.au.auaction import AUAction
from core.state import State


class Model:
    def __init__(self, num_of_states: int = 1, num_of_actions: int = 1, **kwargs):
        """
        Actor Uncertainty model
        """
        self.states = []
        for _ in range(num_of_states):
            state = State(num_of_actions=num_of_actions, action_type=AUAction, **kwargs)
            self.states.append(state)

    def reward(self, reward, state: int = 0, action: int = 0):
        self.states[state].reward(reward=reward, action=action)

    def act(self, state: int = 0, **kwargs):
        probas = self.states[state].act(**kwargs)
        return probas