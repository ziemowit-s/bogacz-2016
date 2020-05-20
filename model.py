import numpy as np
from action import Action


class Model:
    def __init__(self, num_of_actions=1, alfa=0.1, beta=None, actor_only=False):
        self._actions = [Action(alfa=alfa, beta=beta, actor_only=actor_only) for _ in range(num_of_actions)]

    def act(self, r):
        acs = []
        for action in self._actions:
            action.compute(r, a=r, b=r)

            q = action._g - action._n
            s = action._g + action._n
            u = (action.a + action.b) * q - (action.b - action.a) * s

            pi = np.exp(0.5 * u)
            acs.append(pi)

        ps_sum = np.sum(acs)
        return [p / ps_sum for p in acs]