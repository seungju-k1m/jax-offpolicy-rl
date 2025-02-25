from abc import ABC


class BaseAgent(ABC):
    def sample(self, *args, **kwargs):
        raise NotImplementedError("`sample` should be implemented.")
