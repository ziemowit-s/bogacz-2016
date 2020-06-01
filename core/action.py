import abc


class Action:
    def __init__(self):
        pass

    @abc.abstractmethod
    def reward(self, r):
        raise NotImplementedError()

    @abc.abstractmethod
    def act(self, **kwargs):
        raise NotImplementedError()
