from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import List
import warnings
import torch

class Aggregator(metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        pass
    @abstractmethod
    def aggregate(self, grads: List[torch.Tensor], epoch) -> torch.Tensor:
        pass

class AggregatorFactory:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class: Aggregator):
            if name in cls.registry:
                warnings.warn(f'Register {name} already exists. Will replace it')
            cls.registry[name] = wrapped_class
        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs):
        exec_class = cls.registry[name]
        executer = exec_class(**kwargs)
        return executer
