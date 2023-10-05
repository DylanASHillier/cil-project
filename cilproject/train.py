"""Contains training methods"""
import tqdm
import cilproject.models as models
import torch.utils.data
import torch.nn as nn
import torch.optim
import typing
import random


def train_model_cross_entropy(
    model: models.CombinedModel,
    data,
    phase: int,
    history: dict[int, list[torch.Tensor]],
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 5e-3,
    **kwargs,
):
    """Trains the model using training kwargs on the current epoch"""
    assert model.per_phase_models is not None
    model.curr_phase = phase
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history_data = [(x, y) for y in history for x in history[y]]
    data = data + history_data
    for _ in range(epochs):
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
        )
        agg_loss = 0
        acc = 0
        for x, y in tqdm.tqdm(dataloader):
            x, y = x.to("mps"), y.to("mps")
            optimizer.zero_grad()
            preds = model.per_phase_models[phase - 1](x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            agg_loss += loss.item()
            acc += (preds.argmax(dim=1) == y).float().mean().item()
        print(f"Loss: {agg_loss}")
        print(f"Acc: {acc}")
    return model


def train_model_contrastive(
    model: models.CombinedModel,
    data,
    phase: int,
    history: dict[int, list[torch.Tensor]],
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 5e-3,
    **kwargs,
):
    """Trains the model using contrastive learning.

    In particular, we use the simclr objective:
    $$
    \mathcal{L} = \frac{1}{2N}\sum_{i=1}^N \sum_{j=1}^N \left[
    -\log \frac{\exp(z_i \cdot z_j / \tau)}{\sum_{k=1}^{2N} \exp(z_i \cdot z_k / \tau)}
    \right]
    $$

    Where $z_i$ is the embedding of the $i$th image, and $N$ is the batch size.
    """
    assert model.per_phase_models is not None
    model.curr_phase = phase
    model.train()
    history_data = [(x.to("cpu"), y) for y in history for x in history[y]]
    data = data + history_data
    xs = [x for x, _ in data]
    ys = [y for _, y in data]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        agg_loss = 0
        data = _contrastive_pairs(x=xs, y=ys)
        # put data into a dataloader that works with such triplets
        dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(*data),
            batch_size=batch_size,
            shuffle=True,
        )
        for x1, x2, y in tqdm.tqdm(dataloader):
            x1, x2, y = x1.to("mps"), x2.to("mps"), y.to("mps")
            optimizer.zero_grad()
            yhat1 = model.per_phase_models[phase - 1](x1)
            yhat2 = model.per_phase_models[phase - 1](x2)
            # find the similarity between the two embeddings
            pred = torch.sum(yhat1 * yhat2, dim=1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            agg_loss += loss.item()
        print(f"Loss: {agg_loss}")
    return model


def _contrastive_pairs(x, y) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates all the pairs for contrastive learning.
    Suppose x_i is from class y_i, with n instaces for class y_i.
    Then we should have n-1 positive pairs, and only n-1 negative pairs
    to ensure balance."""
    pos_pairs = []
    neg_pairs = []

    for i in range(len(x)):
        for j in range(i, len(x)):
            if i != j and y[i] == y[j]:
                pos_pairs.append((x[i], x[j], 1))
            elif i != j and y[i] != y[j]:
                neg_pairs.append((x[i], x[j], 0))
    random.shuffle(neg_pairs)
    neg_pairs = neg_pairs[: len(pos_pairs)]
    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    # transform into tensors
    tensor_x1 = torch.stack([x1 for x1, _, _ in pairs])
    tensor_x2 = torch.stack([x2 for _, x2, _ in pairs])
    tensor_y = torch.stack([torch.tensor(y, dtype=torch.float) for _, _, y in pairs])
    return tensor_x1, tensor_x2, tensor_y
