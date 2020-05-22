from typing import Optional

import numpy as np


class AUAction:
    def __init__(self, alfa:float = 0.1, beta: Optional[float] = None, actor_only: bool = True,
                 a: int = 1, b: int = 1):
        """
        Actor Uncertainty action
        :param alfa:
        :param beta:
        :param actor_only:
        """
        self._alfa = alfa
        if not beta:
            self._beta = self._alfa / np.sqrt(2 * np.pi)

        self.actor_only = actor_only
        self._v = 0
        self._g = 0
        self._n = 0

        self._s = 0
        self._q = 0

        self.a = 1
        self.b = 1

    def v(self, r: float):
        self._v = self._v + self._alfa * (r - self._v)

    def g(self, r):
        if self.actor_only:
            self._g = self._g + self._alfa * np.max([r - self._q, 0]) - self._beta * self._g
        else:
            self._g = self._g + self._alfa * np.max([r - self._v, 0]) - self._alfa * self._g

    def n(self, r):
        if self.actor_only:
            self._n = self._n + self._alfa * np.abs(np.min([r - self._q, 0])) - self._beta * self._n
        else:
            self._n = self._n + self._alfa * np.abs(np.min([r - self._v, 0])) - self._alfa * self._n

    def q(self, r):
        q = self._g - self._n
        self._q = q + self._alfa * (r - self._v) - self._alfa * q

    def s(self, r):
        s = self._g + self._n
        self._s = s + self._alfa * np.abs(r - self._v) - self._alfa * s

    def reward(self, r):
        self.v(r)
        self.q(r)

        self.g(r)
        self.n(r)

        self.s(r)

    def act(self, a=None, b=None):
        if a:
            self.a = a
        if b:
            self.b = b
        q = self._g - self._n
        s = self._g + self._n

        u = (self.a + self.b) * q - (self.b - self.a) * s

        pi = np.exp(0.5 * u)
        return pi

