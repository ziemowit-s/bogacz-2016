from typing import Optional

import numpy as np


class AUAction:
    def __init__(self, alfa:float = 0.1, beta: float = 0.1, actor_only: bool = True,
                 a: int = 1, b: int = 1):
        """
        Actor Uncertainty action
        :param alfa:
        :param beta:
        :param actor_only:
        """
        self._alfa = alfa
        self._beta = beta
        self.actor_only = actor_only

        self._v = 0
        self._g = 0
        self._n = 0

        self._s = 0
        self._q = 0

        self.a = a
        self.b = b

    def v(self, r: float):
        self._v = self._v + self._alfa * (r - self._v)

    def g(self, r):
        self._g = self._g + self._alfa * np.max([r - self._get_comparator(), 0]) - self._beta * self._g

    def n(self, r):
        self._n = self._n + self._alfa * np.abs(np.min([r - self._get_comparator(), 0])) - self._beta * self._n

    def q(self, r):
        self._q = self._g - self._n  # q + self._alfa * (r - self._v) - self._alfa * q

    def s(self, r):
        self._s = self._g + self._n  # s + self._alfa * np.abs(r - self._v) - self._alfa * s

    def reward(self, r):
        self.g(r)
        self.n(r)

        self.v(r)
        self.q(r)
        self.s(r)

    def act(self, a=None, b=None):
        if a:
            self.a = a
        if b:
            self.b = b

        u = (self.a + self.b) * self._q - (self.b - self.a) * self._s
        pi = np.exp(0.5 * u)
        return pi

    def _get_comparator(self):
        if self.actor_only:
            return self._q
        else:
            return self._v

