from datetime import datetime

import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter

from lib.path import output_path


class Trainer:
    def __init__(self, model, train_loader, val_loader, model_name, lr=1e-4):
        self.device = self.__get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()

        # 로그 경로 설정
        log_path = output_path() / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=str(log_path))
        self.best_acc = 0.0
        self.save_dir = log_path

    def __get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def run(self, epochs=30):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch}')

            for i, (imgs, labels) in enumerate(pbar):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(imgs), labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                self.writer.add_scalar(
                    'Loss/train', loss.item(), epoch * len(self.train_loader) + i
                )
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            acc = self.evaluate()
            self.writer.add_scalar('Accuracy/val', acc, epoch)

            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')

            print(f'✨ Epoch {epoch}: Val Acc {acc:.2f}% | Best {self.best_acc:.2f}%')
        self.writer.close()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            preds = self.model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return 100 * correct / total
