"""Contains methods for making/saving predictions with the models."""
import numpy as np
import torch


def save_predictions(dataset, embedder, classifier, pred_path, device="cpu", phase=1):
    """Saves predictions for the given dataset."""
    lines = []
    for x, label in dataset:
        x = x.to(device)
        with torch.no_grad():
            embedding = embedder(x.unsqueeze(0)).squeeze()
        embedding = embedding.cpu().detach().numpy()
        pred_classes = classifier.predict_proba(embedding.reshape(1, -1))[0]
        lines.append(f"{label} {str(np.argmax(pred_classes))}")
    with open(f"{pred_path}/result_{phase}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved predictions for phase {phase}.")
