import torch
from abc import ABC, abstractmethod

class Op(ABC):
    def __init__(self, name, method, **kwargs):
        self.name = name
        self.method = method
        self.kwargs = kwargs

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

