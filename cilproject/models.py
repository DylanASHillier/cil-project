"""Defines any models used in the project."""
from torch import nn
import abc


class PerPhaseModel(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Aggregator(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, common_embedding, per_phase_embeddings):
        pass


class CommonModel(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class CombinedModel(nn.Module):
    """A model used to combine together different subcomponents,

    handling the logic of passing the data between them, and
    whether or not they exist."""

    def __init__(
        self,
        per_phase_model: PerPhaseModel,
        aggregator: Aggregator,
        common_model: CommonModel,
    ):
        super().__init__()
        self.per_phase_model = per_phase_model
        self.aggregator = aggregator
        self.common_model = common_model

    def forward(self, *args, **kwargs):
        common_embedding = self.common_model(*args, **kwargs)
        per_phase_embeddings = self.per_phase_model(*args, **kwargs)
        return self.aggregator(common_embedding, per_phase_embeddings)
