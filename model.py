import numpy as np
from action import Action


class Model:
    def __init__(self, num_of_actions=1, alfa=0.1, beta=None, actor_only=False):
        """
        We assume that the model can be in one of n states and each state is associated with a
        single correct actions. So num_of_actions = num_of states
        :param num_of_actions:
        :param alfa:
        :param beta:
        :param actor_only:
        """
        self._actions = [Action(alfa=alfa, beta=beta, actor_only=actor_only) for _ in range(num_of_actions)]

    def act(self, r, action_index=0):
        acs = []
        for i, action in enumerate(self._actions):
            if i == action_index:
                action.compute(r, a=1, b=1)
            else:
                action.compute(-r, a=1, b=1)

            q = action._g - action._n
            s = action._g + action._n
            u = (action.a + action.b) * q - (action.b - action.a) * s

            pi = np.exp(0.5 * u)
            acs.append(pi)

        ps_sum = np.sum(acs)
        return [p / ps_sum for p in acs]