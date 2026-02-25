import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms


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


def predict_skin_disease(image_path, model, classes, device, img_size=224):
    # 1. 모델을 평가 모드로 전환
    model.eval()

    # 2. 이미지 로드 및 전처리
    # 학습 때 사용했던 Compose와 동일한 순서여야 합니다 (단, 증강은 제외)
    preprocess = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert('RGB')
    input_tensor = (
        preprocess(image).unsqueeze(0).to(device)
    )  # 배치 차원 추가 (1, 3, H, W)

    # 3. 추론 수행
    with torch.no_grad():
        outputs = model(input_tensor)

        # Softmax를 통해 각 클래스별 확률(0~1) 계산
        probabilities = F.softmax(outputs, dim=1)

        # 가장 높은 확률의 인덱스와 값 추출
        conf, pred_idx = torch.max(probabilities, dim=1)

    result_class = classes[pred_idx.item()]
    confidence = conf.item() * 100

    print(f'🔍 분석 결과: {result_class} ({confidence:.2f}%)')
    return result_class, confidence


# --- 사용 예시 ---
# class_names = ['herpes', 'panu', 'rosacea', ...] (직접 입력하거나 dm.classes 사용)
# result, score = predict_skin_disease('test_image.jpg', model, class_names, trainer.device)
