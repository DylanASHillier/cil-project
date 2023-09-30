"""Script used for running training experiments."""
import collections

import classifiers
import dataset
import embedders
import memory
import predictions
import torch
import torch.utils.data
import typer


def run_experiment(
    data_folder_path: str = typer.Argument(...),
    device: str = typer.Option("cpu"),
    random_seed: int = typer.Option(42),
    embedding_model: str = typer.Option("hf_hub:timm/tiny_vit_21m_224.in1k"),
    memory_type: str = typer.Option("kmeans"),
    pred_path: str = typer.Option(None),
    phase_model: str = typer.Option(None),
    aggregation: str = typer.Option(None),
):
    """Runs an experiment."""
    torch.manual_seed(random_seed)
    if phase_model is None:
        phase_model = "linear"
    if aggregation is None:
        aggregation = "mean"
    embedder = embedders.get_embedder(device, embedding_model)
    embedder = embedders.EmbedderCache(embedder)
    history = collections.defaultdict(list)
    val_history = collections.defaultdict(list)
    for phase in range(1, 11):
        train_dataset = dataset.get_train_dataset(
            phase, dataset.get_imagenet_transform(), data_dir=data_folder_path
        )
        train_ds, val_ds = torch.utils.data.random_split(
            train_dataset,
            [
                int(0.8 * len(train_dataset)),
                len(train_dataset) - int(0.8 * len(train_dataset)),
            ],
        )
        val_history = memory.add_memory(
            val_history,
            val_ds,
            embedder=embedder,
            label_offset=(phase - 1) * 10,
            max_per_class=1000,
        )
        memory_func = memory.get_memory_function(memory_type)
        history = memory_func(
            history,
            train_ds,
            label_offset=(phase - 1) * 10,
            embedder=embedder,
            max_per_class=5,
            device=device,
        )
        print(f"Saving predictions for phase {phase}")
        classifier = classifiers.train_classifier(
            classifier_name="linear", history=history
        )
        predictions.save_predictions(
            dataset.LeaderboardValDataset(
                f"{data_folder_path}/Val",
                dataset.get_imagenet_transform(),
            ),
            embedder=embedder,
            classifier=classifier,
            pred_path=pred_path,
            device="mps",
            phase=phase,
        )
        print(f"Phase {phase}: {len(train_dataset)} images")


if __name__ == "__main__":
    typer.run(run_experiment)
