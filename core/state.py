import numpy as np
from typing import TypeVar, Type

from core.action import Action

T = TypeVar('T', bound=Action)


class State:
    def __init__(self, action_type: Type[T], num_of_actions: int = 1, **kwargs):

        self.actions = []
        for _ in range(num_of_actions):
            a = action_type(**kwargs)
            self.actions.append(a)

    def reward(self, reward, action: int = 0):
        self.actions[action].reward(r=reward)

    def act(self, **kwargs):
        probas = []
        for act in self.actions:
            pi = act.act(**kwargs)
            probas.append(pi)

        ps_sum = np.sum(probas)
        return [p / ps_sum for p in probas]

    def __repr__(self):
        return str(self.act())
