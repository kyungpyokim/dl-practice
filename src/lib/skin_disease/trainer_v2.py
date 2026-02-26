import copy

import torch


class TrainerV2:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        model_name,
        lr=1e-4,
        device='cuda',
        patience=5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.device = device

        self.criterion = torch.nn.CrossEntropyLoss()
        # 학습 가능한 파라미터만 optimizer에 전달 (Freeze 대응)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        # Early Stopping 관련 변수
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.early_stop = False

    def run(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # 검증 단계
            val_acc, val_loss = self.evaluate()
            print(
                f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(self.train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%'
            )

            # 🌟 Early Stopping 로직
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.counter = 0
                # 최고 성능일 때 모델 저장
                torch.save(self.best_model_wts, f'best_{self.model_name}.pth')
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    print('🛑 Early stopping triggered. 학습을 종료합니다.')
                    self.early_stop = True
                    break

        # 학습 종료 후 최적의 가중치로 복구
        self.model.load_state_dict(self.best_model_wts)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return (100 * correct / total), (val_loss / len(self.val_loader))
