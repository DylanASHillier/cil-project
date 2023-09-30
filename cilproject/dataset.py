"""Utilites for loading the dataset in all it's forms"""
import torchvision.datasets
import torchvision.transforms


def get_train_dataset(phase, transform, data_dir="data"):
    """Returns the train dataset for a given phase."""
    return torchvision.datasets.ImageFolder(
        f"{data_dir}/Train/phase_{phase}", transform=transform
    )


def get_leaderboard_val_dataset(transform):
    """Returns the leaderboard val dataset."""
    return torchvision.datasets.ImageFolder(
        "../data/leaderboard_val", transform=transform
    )


def get_imagenet_transform():
    """Returns the imagenet transform."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
