import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def test_model(model, test_loader, classes, device):
    model.eval()
    all_preds = []
    all_labels = []

    # 1. 예측 수행
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 2. 성능 지표 출력 (Classification Report)
    print('\n📊 상세 성능 리포트:')
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 3. 혼동 행렬 (Confusion Matrix) 시각화
    # 어떤 병을 어떤 병으로 헷갈려 하는지 한눈에 보여줍니다.
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
