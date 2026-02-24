import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import tqdm
from torch import optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from lib.path import model_file_path, output_path


class TrainModel:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    __device = 'cuda'

    @property
    def device(self):
        if self.__device != 'cuda':
            self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return self.__device

    def train(self):
        log_dir = output_path() / 'tensorboard'
        writer = SummaryWriter(log_dir=log_dir)

        self.optim = Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        epochs = 20

        count = 0

        # 체크포인트 불러오기
        start_ep = self.__load_checkpoint()

        for ep in range(start_ep + 1, epochs):
            train_tqdm = tqdm.tqdm(self.train_loader)
            for data, label in train_tqdm:
                data = data.to(self.device)
                label = label.to(self.device)
                self.optim.zero_grad()
                pred = self.model(data)
                loss = criterion(pred, label)
                writer.add_scalar('Loss/train', loss, count)
                count += 1
                loss.backward()
                self.optim.step()

                train_tqdm.set_description(f'epoch : {ep + 1} loss : {loss.item():.2f}')
            # 체크포인트 저장
            self.__checkpoint(ep, loss)

    def __checkpoint(self, ep, loss):
        torch.save(
            {
                'epoch': ep,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': loss,
            },
            model_file_path('checkpoint.pth'),
        )

    def __load_checkpoint(self):
        import os

        path = model_file_path('checkpoint.pth')

        # 1. 파일 존재 여부 먼저 확인 (경로 오류 방지)
        if not os.path.exists(path):
            return 0

        try:
            # 2. weights_only=True로 로드 속도 및 안정성 향상 (PyTorch 최신버전 권장)
            checkpoint = torch.load(path, map_location=self.device)

            if checkpoint:
                # 3. 모델 가중치 로드
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # 4. 옵티마이저 로드 (이 과정에서 메모리 점유가 늘어남)
                if 'optimizer_state_dict' in checkpoint:
                    self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

                # 5. 메모리 정리 (중요!)
                del checkpoint
                torch.cuda.empty_cache()  # GPU 사용 시

                return checkpoint.get('epoch', 0)

        except RuntimeError as e:
            print(f'체크포인트 로드 중 오류 발생: {e}')
            # 여기서 커널이 죽는다면 대부분 메모리 부족입니다.

        return 0


def save_checkpoint(ep, model, optim, loss):
    torch.save(
        {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
        },
        model_file_path(f'{model.arc_name}_checkpoint.pth'),
    )


def load_checkpoint(model, optim, device):
    path = model_file_path('checkpoint.pth')

    # 1. 파일 존재 여부 먼저 확인 (경로 오류 방지)
    if not os.path.exists(path):
        return 0
    try:
        # 2. weights_only=True로 로드 속도 및 안정성 향상 (PyTorch 최신버전 권장)
        checkpoint = torch.load(path, map_location=device)
        if checkpoint:
            # 3. 모델 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])

            # 4. 옵티마이저 로드 (이 과정에서 메모리 점유가 늘어남)
            if 'optimizer_state_dict' in checkpoint:
                optim.load_state_dict(checkpoint['optimizer_state_dict'])

            # 5. 메모리 정리 (중요!)
            del checkpoint
            torch.cuda.empty_cache()  # GPU 사용 시

            return checkpoint.get('epoch', 0)

    except RuntimeError as e:
        print(f'체크포인트 로드 중 오류 발생: {e}')
        # 여기서 커널이 죽는다면 대부분 메모리 부족입니다.

    return 0


def train_model(model, train_loader, lr=1e-4, epochs=20):
    log_dir = (
        output_path()
        / 'tensorboard'
        / model.arc_name
        / datetime.now().strftime('%Y%m%d')
    )
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    # start_ep = load_checkpoint(model, optim, device)
    # start_ep = start_ep > 0 if start_ep else -1
    start_ep = -1
    count = 0

    for ep in range(start_ep + 1, epochs):
        train_tqdm = tqdm.tqdm(train_loader)
        for image, labels in train_tqdm:
            optimizer.zero_grad()
            preds = model(image.to(device))
            loss = criterion(preds, labels.to(device))
            writer.add_scalar('Loss/train', loss.item(), count)
            count += 1
            loss.backward()
            optimizer.step()

            train_tqdm.set_description(f'epoch : {ep} loss : {loss.item():.2f}')

        save_checkpoint(ep, model, optimizer, loss)

    writer.close()
    print('Completed')


def eval_model(model, test_loader, test_dataset, show_image=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()

    with torch.no_grad():
        corrects = 0

        for i, (image, labels) in enumerate(test_loader):
            preds = model(image.to(device))
            pred = torch.max(preds, 1)[1]

            corrects += torch.sum(pred == labels.to(device).data)

            image_grid = torchvision.utils.make_grid(image)
            print(labels)
            if i == 0 and show_image is not None:
                image_grid = torchvision.utils.make_grid(image[:8])  # 앞 8개만
                show_image(image_grid.cpu(), title=f'Pred: {pred[:8].tolist()}')

        acc = corrects.double() / len(test_dataset)
        print(f'정확도 : {acc:.4f}')
