from abc import ABC, abstractmethod

class Part(ABC):
    @abstractmethod
    def build_model(self):
        pass