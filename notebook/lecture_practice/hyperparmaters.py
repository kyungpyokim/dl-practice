# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dl_practice (3.11.14)
#     language: python
#     name: python3
# ---

# %%
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms


# %%
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# %%
train_datasets = datasets.ImageFolder(root='./dataset/train', transform=transform_train)
test_datasets = datasets.ImageFolder(root='./dataset/test', transform=transform_test)

# %%
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 3)
model

# %%
from torch.utils.data import random_split

train_data_size = len(train_datasets)

# 8:2  train:8   test:2
train_size = int(train_data_size * 0.8)
val_size = train_data_size - train_size

print(f'train data size : {train_size}    val data size : {val_size}')

train_dataset, val_dataset = random_split(train_datasets, [train_size, val_size])

# batchsize 최적화를 위해 objective 함수 안에서 처리
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 3)
model = model.to(device)


# %%
def objective(trial):

    batch_size = trial.suggest_categorical('batch_size', [4, 6, 8])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epochs = 20
    val_loss = 0

    for epoch in range(epochs):
        model.train()

        for img, labels in train_dataloader:
            optimizer.zero_grad()
            preds = model(img.to(device))
            loss = criterion(preds, labels.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for img, labels in val_dataloader:
                img = img.to(device)
                labels = labels.to(device)
                preds = model(img)
                val_loss += criterion(preds, labels)

        total_loss = val_loss / len(val_dataloader)

        trial.report(total_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return total_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
