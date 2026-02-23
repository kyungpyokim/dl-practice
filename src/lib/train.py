import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from lib.path import model_file_path, output_path


class TrainModel:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        log_dir = output_path() / 'tensorboard'
        writer = SummaryWriter(log_dir=log_dir)

        optim = Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        epochs = 20

        count = 0

        # 체크포인트 불러오기
        start_ep = self.load_checkpoint()

        for ep in range(start_ep + 1, epochs):
            train_tqdm = tqdm.tqdm(self.train_loader)
            for data, label in train_tqdm:
                data = data.to(self.device)
                label = label.to(self.device)
                optim.zero_grad()
                pred = self.model(data)
                loss = criterion(pred, label)
                writer.add_scalar('Loss/train', loss, count)
                count += 1
                loss.backward()
                optim.step()

                train_tqdm.set_description(f'epoch : {ep + 1} loss : {loss.item():.2f}')
            # 체크포인트 저장
            self.checkpoint(ep, loss)

    def checkpoint(self, ep, loss):
        torch.save(
            {
                'epoch': ep,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': loss,
            },
            model_file_path('checkpoint.pth'),
        )

    def load_checkpoint(self):
        checkpoint = torch.load(
            model_file_path('checkpoint.pth'), map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch']
