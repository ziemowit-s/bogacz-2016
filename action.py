import numpy as np


class Action:
    def __init__(self, alfa=0.1, beta=None, actor_only=True):
        self._alfa = alfa
        if not beta:
            self._beta = self._alfa / np.sqrt(2 * np.pi)

        self.actor_only = actor_only
        self._v = 0
        self._g = 0
        self._n = 0

        self._s = 0
        self._q = 0

        self.a = 0
        self.b = 0

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

    def compute(self, r, a, b):
        self.a = a
        self.b = b

        self.v(r)
        self.q(r)

        self.g(r)
        self.n(r)

        self.s(r)