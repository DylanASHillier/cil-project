"""Utilites for loading the dataset in all it's forms"""
import torchvision.datasets
import torchvision.transforms
import torch.utils.data as data


def get_train_dataset(phase, transform, data_dir="data"):
    """Returns the train dataset for a given phase."""
    return torchvision.datasets.ImageFolder(
        f"{data_dir}/Train/phase_{phase}", transform=transform
    )


class LeaderboardValDataset(data.Dataset):
    """The leaderboard val dataset."""

    def __init__(self, path, transform):
        self.dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset.imgs[index][0]

    def __len__(self):
        return len(self.dataset)


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
