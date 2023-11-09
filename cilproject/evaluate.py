"""Stores methods for evaluation."""
import numpy as np
import torch, os
from visualize_heatmap import gen_heatmap_and_save


def evaluate_classifier(
    val_data: dict, classifier, imgs=None, embedding_model=None, phase=None
):
    X = np.concatenate(list(val_data.values()))
    y = np.concatenate([[k] * len(v) for k, v in val_data.items()])
    for k, v in val_data.items():
        for idx, item in enumerate(v):
            if phase == 10:
                print("img:", imgs[k][idx])
                print("label:", k)
                print("predict:", classifier.predict([item])[-1])
                img_dir = imgs[k][idx].replace("data/Train", "preds/vis")[:-8]
                img_file = imgs[k][idx].replace("data/Train", "preds/vis")[-8:]
                if classifier.predict([item])[-1] == k:
                    # ./data/Train/phase_1/009/028.jpg
                    img_dir = img_dir + "/right"
                    os.makedirs(img_dir, exist_ok=True)
                    gen_heatmap_and_save(
                        imgs[k][idx], embedding_model, img_dir + img_file
                    )
                else:
                    img_dir = img_dir + "/wrong"
                    os.makedirs(img_dir, exist_ok=True)
                    gen_heatmap_and_save(
                        imgs[k][idx], embedding_model, img_dir + img_file
                    )
    return classifier.score(X, y)
