"""Script used for running training experiments."""
import collections

import cilproject.classifiers as classifiers
import cilproject.dataset as dataset
import cilproject.embedders as embedders
import cilproject.memory as memory
import cilproject.predictions as predictions
import cilproject.evaluate as evaluate
import cilproject.models as models
import cilproject.train as train
import torch
import torch.utils.data
import typer


def run_experiment(
    data_folder_path: str = typer.Argument(...),
    device: str = typer.Option("mps"),
    random_seed: int = typer.Option(42),
    embedding_model: str = typer.Option("hf_hub:timm/tiny_vit_21m_224.in1k"),
    memory_type: str = typer.Option("kmeans"),
    train_phase_with_memory: bool = typer.Option(False),
    pred_path: str = typer.Option(None),
    phase_model: str = typer.Option(None),
    common_model: str = typer.Option(None),
    aggregation: str = typer.Option(None),
    save_preds: bool = typer.Option(False),
    train_type: str = typer.Option("contrastive"),
    disable_val: bool = typer.Option(False),
    num_epochs: int = typer.Option(10),
    classifier_name: str = typer.Option("linear"),
):
    """Runs an experiment."""
    torch.manual_seed(random_seed)
    if aggregation is None:
        aggregation = "concat"
    embedder = embedders.get_embedder(device, embedding_model, use_existing_head=True)
    embedder = embedders.EmbedderCache(embedder)
    history = collections.defaultdict(list)
    val_history = collections.defaultdict(list)
    model = models.get_model(
        phase_model, aggregation, common_model, embedding_size=1000
    )
    print(model)
    model.to(device)
    for phase in range(1, 11):
        train_dataset = dataset.get_train_dataset(
            phase, dataset.get_imagenet_transform(), data_dir=data_folder_path
        )
        if not disable_val:
            train_ds, val_ds = torch.utils.data.random_split(
                train_dataset,
                [
                    int(0.8 * len(train_dataset)),
                    len(train_dataset) - int(0.8 * len(train_dataset)),
                ],
            )
        else:
            train_ds = train_dataset
        train_ds = dataset.EmbeddedDataset(
            train_ds,
            embedder,
            device=device,
            save_path=f"{data_folder_path}/train_{phase}.pt",
        )
        if not disable_val:
            val_ds = dataset.EmbeddedDataset(
                val_ds,
                embedder,
                device=device,
                save_path=f"{data_folder_path}/val_{phase}.pt",
            )
        if model.per_phase_models is not None and train_type == "contrastive":
            if train_phase_with_memory:
                model = train.train_model_contrastive(
                    model,
                    train_ds,
                    phase=phase,
                    history=history,
                    epochs=num_epochs,
                    device=device,
                    lr=1e-4,
                )
            else:
                model = train.train_model_contrastive(
                    model,
                    train_ds,
                    phase=phase,
                    epochs=num_epochs,
                    device=device,
                    lr=1e-4,
                )
        elif model.per_phase_models is not None and train_type == "supervised":
            model = train.train_model_cross_entropy(
                model,
                train_ds,
                phase=phase,
                history=history,
                epochs=10,
                device=device,
                lr=1e-4,
            )
        if not disable_val:
            val_history = memory.add_memory(
                val_history,
                val_ds,
                embedder=embedder,
                label_offset=(phase - 1) * 10,
                max_per_class=1000,
                device=device,
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
        if aggregation == "calibrated_add":
            model = train.calibrate_model(
                model,
                phase=phase,
                history=history,
                epochs=2,
                device=device,
                lr=1e-3,
            )
        model_history = models.get_model_embedded_history(model, history, device=device)
        classifier = classifiers.train_classifier(
            classifier_name=classifier_name,
            history=model_history,
        )
        if not disable_val:
            model_val_history = models.get_model_embedded_history(
                model, val_history, device=device
            )
            score = evaluate.evaluate_classifier(
                model_val_history,
                classifier,
            )
            print(f"Phase {phase} score: {score}")
        if save_preds:
            print(f"Saving predictions for phase {phase} at {pred_path}")
            predictions.save_predictions(
                dataset.LeaderboardValDataset(
                    f"{data_folder_path}/Val",
                    dataset.get_imagenet_transform(),
                ),
                embedder=embedder,
                model=model,
                classifier=classifier,
                pred_path=pred_path,
                device="mps",
                phase=phase,
            )
        print(f"Phase {phase}: {len(train_dataset)} images")


if __name__ == "__main__":
    typer.run(run_experiment)
