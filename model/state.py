import numpy as np
from collections import namedtuple

from model.action import Action

ActionValue = namedtuple("ActionValue", "name g n")


class State:
    def __init__(self, num_of_actions: int = 1, alfa: float = 0.1,
                 beta: float = None, actor_only: bool = False):

        self.actions = []
        for _ in range(num_of_actions):
            a = Action(alfa=alfa, beta=beta, actor_only=actor_only)
            self.actions.append(a)

    def reward(self, reward, action: int = 0):
        self.actions[action].reward(r=reward)

    def act(self, a=None, b=None):
        probas = []
        for act in self.actions:
            pi = act.act(a=a, b=b)
            probas.append(pi)

        ps_sum = np.sum(probas)
        return [p / ps_sum for p in probas]

    def get_values(self):
        result = []
        for i, a in enumerate(self.actions):
            values = a.get_values()
            a_val = ActionValue(name=i, g=values[0], n=values[1])
            result.append(a_val)
        return result

    def __repr__(self):
        return str(self.act())
