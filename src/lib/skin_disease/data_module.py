import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DataModule:
    def __init__(
        self, data_dir, img_size=224, batch_size=32, val_split=0.2, mean=None, std=None
    ):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.mean = mean if mean else [0.485, 0.456, 0.406]
        self.std = std if std else [0.229, 0.224, 0.225]
        self._create_transforms()

    def _create_transforms(self):
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def setup(self):
        full_dataset = datasets.ImageFolder(root=self.data_dir)
        self.classes = full_dataset.classes
        indices = list(range(len(full_dataset)))
        split = int(np.floor(self.val_split * len(full_dataset)))

        np.random.seed(42)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]

        self.train_ds = ApplyTransform(
            Subset(full_dataset, train_idx), self.train_transform
        )
        self.val_ds = ApplyTransform(Subset(full_dataset, val_idx), self.test_transform)
        print(f'✅ Data Loaded: Train({len(self.train_ds)}), Val({len(self.val_ds)})')

    def get_loaders(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader
