"""Stores the classification heads and sklearn training code."""
import sklearn.linear_model as linear_model
import torch


def get_classifier(classifier_name: str, device: str | None, num_classes: int | None):
    """Returns a classifier."""
    if classifier_name == "linear":
        return linear_model.LogisticRegression(multi_class="multinomial", max_iter=1000)
    else:
        raise ValueError(f"Unknown classifier {classifier_name}.")


def train_classifier(classifier_name, history, device="cpu"):
    """Trains a classifier."""
    classifier = get_classifier(classifier_name, device, len(history))
    X = []
    y = []
    for label, embeddings in history.items():
        X.extend(embeddings)
        y.extend([label] * len(embeddings))
    classifier.fit(X, y)
    return classifier
