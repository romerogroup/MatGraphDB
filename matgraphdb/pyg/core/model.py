import abc

import torch


class BaseModel(torch.nn.Module, abc.ABC):
    def __init__(self, task_type: str = "regression"):
        super().__init__()
        self.task_type = task_type

    @abc.abstractmethod
    def forward(self, data):
        pass

    @abc.abstractmethod
    def compute_loss(self, data, outputs, loss_fn):
        pass
