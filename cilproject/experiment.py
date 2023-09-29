"""Script used for running training experiments."""
import torch
import typer
import torch.utils.data
import collections
import memory
import dataset
import embedders
import classifiers
import predictions
import models
import tqdm


def run_experiment(
    data_folder_path: str = typer.Argument(...),
    device: str = typer.Option("cpu"),
    random_seed: int = typer.Option(42),
    embedding_model: str = typer.Option("resnet18"),
    memory_type: str = typer.Option("kmeans"),
    pred_path: str = typer.Option(None),
    phase_model: str = typer.Option(None),
    aggregation: str = typer.Option(None),
    batch_size: int = typer.Option(32),
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
            classifier_name="linear", history=history, embedder=embedder, device="mps"
        )
        predictions.save_predictions(
            dataset.get_leaderboard_val_dataset(dataset.get_imagenet_transform()),
            embedder=embedder,
            classifier=classifier,
            pred_path=pred_path,
            device="mps",
            phase=phase,
        )
        print(f"Phase {phase}: {len(train_dataset)} images")


if __name__ == "__main__":
    typer.run(run_experiment)