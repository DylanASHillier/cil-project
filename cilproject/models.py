"""Defines any models used in the project."""
import torch
import torch.utils.data
from torch import nn
import tqdm
import abc
from typing import Sequence
from collections import defaultdict
import timm.layers.classifier as timm_classifier


class PerPhaseModel(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class LinearPerPhaseModel(PerPhaseModel):
    def __init__(self, num_classes: int, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class NNPerPhaseModel(PerPhaseModel):
    def __init__(
        self, num_classes: int, embedding_size: int, hidden_size: int, num_layers: int
    ):
        super().__init__()
        self.linear = nn.Linear(embedding_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return self.final(x)


class NormMlpClassifierHead(PerPhaseModel):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.norm_layer = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.norm_layer(x)
        x = torch.tanh(x)
        return self.linear(x)


class NormMLPResidualHead(PerPhaseModel):
    def __init__(self, base_model, embedding_size: int):
        super().__init__()
        self.base_model = base_model
        self.residual_model = NormMlpClassifierHead(
            in_features=embedding_size,
            num_classes=embedding_size,
        )

    def forward(self, x):
        with torch.no_grad():
            x_base = self.base_model(x)
        return x_base + self.residual_model(x)


class Aggregator(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, common_embedding, per_phase_embeddings):
        pass


class ConcatAggregator(Aggregator):
    def forward(self, common_embedding, per_phase_embeddings):
        if common_embedding is None:
            return torch.cat(per_phase_embeddings, dim=1)
        elif per_phase_embeddings is None:
            return common_embedding
        return torch.cat([common_embedding] + per_phase_embeddings, dim=1)


class AddAggregator(Aggregator):
    def forward(self, common_embedding, per_phase_embeddings):
        if common_embedding is None:
            return sum(per_phase_embeddings)
        elif per_phase_embeddings is None:
            return common_embedding
        return common_embedding + sum(per_phase_embeddings)


class CommonModel(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.TensorType:
        pass


class IdModel(CommonModel):
    def forward(self, x):
        return x


class CombinedModel(nn.Module):
    """A model used to combine together different subcomponents,

    handling the logic of passing the data between them, and
    whether or not they exist."""

    def __init__(
        self,
        per_phase_models: Sequence[PerPhaseModel] | None,
        aggregator: Aggregator,
        common_model: CommonModel | None = None,
    ):
        super().__init__()
        if per_phase_models is not None:
            self.per_phase_models = nn.ModuleList(per_phase_models)
        else:
            self.per_phase_models = None
        self.aggregator: Aggregator = aggregator
        self.common_model = common_model
        self.curr_phase = 0

    def forward(self, *args, **kwargs):
        if self.common_model is None:
            common_embedding = None
        else:
            common_embedding = self.common_model(*args, **kwargs)
        if self.per_phase_models is None:
            per_phase_embeddings = None
        else:
            per_phase_embeddings = [
                per_phase_model(*args, **kwargs)
                for per_phase_model in self.per_phase_models
            ]
        return self.aggregator(common_embedding, per_phase_embeddings)


def get_model(
    per_phase_model: str | None,
    aggregator: str,
    common_model: str | None = None,
    embedding_size: int = 1000,
):
    """Returns the model."""
    if common_model == "id":
        common = IdModel()
    elif common_model is None:
        common = None
    else:
        raise ValueError(f"Unknown common model {common_model}.")
    if per_phase_model == "linear":
        phase_models = [
            LinearPerPhaseModel(num_classes=10, embedding_size=embedding_size)
            for _ in range(10)
        ]
    elif per_phase_model == "nn":
        phase_models = [
            NNPerPhaseModel(
                num_classes=10,
                embedding_size=embedding_size,
                hidden_size=100,
                num_layers=1,
            )
            for _ in range(10)
        ]
    elif per_phase_model == "timm_classifier":
        phase_models = [
            NormMlpClassifierHead(
                in_features=embedding_size,
                num_classes=10,
            )
            for _ in range(10)
        ]
    elif per_phase_model == "timm_classifier_residual":
        phase_models = [
            NormMLPResidualHead(
                base_model=common,
                embedding_size=embedding_size,
            )
            for _ in range(10)
        ]
    elif per_phase_model is None:
        phase_models = None
    else:
        raise ValueError(f"Unknown per phase model {per_phase_model}.")
    if aggregator == "concat":
        agg = ConcatAggregator()
    elif aggregator == "add":
        agg = AddAggregator()
    else:
        raise ValueError(f"Unknown aggregator {aggregator}.")
    return CombinedModel(phase_models, agg, common)


def get_model_embedded_history(
    model: CombinedModel,
    history,
    device: str = "cpu",
):
    """Returns the history with the model's embeddings added."""
    model.eval()
    new_dict = defaultdict(list)
    for label, embeddings in history.items():
        if len(embeddings) == 0:
            print(f"Skipping {label}")
        with torch.no_grad():
            new_dict[label] = (
                model(
                    torch.stack(embeddings).to(device),
                )
                .cpu()
                .numpy()
            )
    return new_dict
