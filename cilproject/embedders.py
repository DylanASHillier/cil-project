"""Stores code for instantiating embedders."""
from torch import nn
import torch
import timm


class FrozenEmbedder(nn.Module):
    """An embedder that does not update its weights during training."""

    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def __call__(self, *args, **kwargs):
        return self.embedder(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the embedder."""
        return self.embedder(*args, **kwargs)

    def train(self, mode=True):
        pass


class TimmEmbedder(nn.Module):
    """An embedder that uses a pretrained timm model."""

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        # self.model.reset_classifier(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        """Forward pass through the embedder."""
        return self.model(x)

    def train(self, mode=True):
        self.model.train(mode)


class EmbedderCache(nn.Module):
    """A cache for the embedder."""

    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        self.cache = {}

    def forward(self, x: torch.Tensor):
        """Forward pass through the embedder."""
        if x.device not in self.cache:
            self.cache[x.device] = self.embedder(x)
        return self.cache[x.device]

    def train(self, mode=True):
        self.embedder.train(mode)


def get_embedder(device: str, model_name: str):
    """Returns the embedder."""
    return TimmEmbedder(model_name).to(device)
