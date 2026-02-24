from torch.utils.data import DataLoader, Dataset


def create_loader(train_dataset, test_dataset, batch_size=32):
    train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return (train, test)


class ApplyTransform(Dataset):
    """Subset에 각각 다른 Transform을 적용하기 위한 래퍼 클래스"""

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


def set_transform(dataset, transforms_tuple):
    train_subset, test_subset = dataset
    train_trans, test_trans = transforms_tuple

    # 원본을 건드리지 않고 개별 전처리를 입힌 새 객체 생성
    train_dataset = ApplyTransform(train_subset, train_trans)
    test_dataset = ApplyTransform(test_subset, test_trans)

    return train_dataset, test_dataset
