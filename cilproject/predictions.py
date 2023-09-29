"""Contains methods for making/saving predictions with the models."""
import numpy as np
import torch


def save_predictions(dataset, embedder, classifier, pred_path, device="cpu", phase=1):
    """Saves predictions for the given dataset."""
    preds = []
    for i, (x, _) in enumerate(dataset):
        x = x.to(device)
        with torch.no_grad():
            embedding = embedder(x.unsqueeze(0)).squeeze()
        embedding = embedding.cpu().detach().numpy()
        pred_classes = classifier.predict_proba(embedding.reshape(1, -1))[0]
        preds.append(np.argmax(pred_classes))
    with open(f"{pred_path}/result_{phase}.txt", "w") as f:
        f.write("\n".join([str(x) for x in preds]))
    print(f"Saved predictions for phase {phase}.")
