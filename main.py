# flake8: noqa: F401

import io
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # global model
    # model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    app.state.device = device
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    app.state.model = model
    model.fc = nn.Linear(512, 3)
    # model.load_state_dict(torch.load('./models/best_model.pth', map_location=device))

    checkpoint = torch.load('./models/best_model.pth', map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print('====================== 모델 세팅 완료 ==============================')
    yield
    # Clean up the ML models and release the resources
    print('서버 종료')


app = FastAPI(lifespan=lifespan)


@app.get('/')
def root():
    return {'result': 'Hi!!!'}


@app.post('/infer')
async def infer(req: Request, file: Annotated[UploadFile, File(...)]):
    allowed_ext = ['jpg', 'jpeg', 'png', 'webp']
    ext = file.filename.split('.')[-1].lower()

    if ext not in allowed_ext:
        return {'error': '이미지 파일만 업로드하세요!!!'}

    new_file_name = f'{uuid.uuid4()}.{ext}'
    file_path = os.path.join('upload_img', new_file_name)

    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 추론 코드

    # img = await file.read()
    # img_data = Image.open(io.BytesIO(img)).convert('RGB')
    img_data = await process_image(file)

    # 전처리
    device = req.app.state.device
    transforms_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transforms_test(img_data).unsqueeze(0).to(device)

    model = req.app.state.model
    with torch.no_grad():
        pred = model(input_tensor)
        index = torch.argmax(pred, dim=1).item()

    model_class = ['마동석', '카리나', '장원영']

    return {'result': model_class[index], 'index': index, 'filename': new_file_name}


async def process_image(file):
    try:
        # 1. 데이터 읽기
        await file.seek(0)
        img_bytes = await file.read()

        # 2. 바이트 데이터 확인
        if not img_bytes:
            # 문자열을 리턴하는 대신, 에러를 발생시켜서 멈추게 합니다.
            raise ValueError('파일 데이터가 비어 있어 처리를 중단합니다.')

        # 3. 이미지 오픈 시도
        img_data = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        print(
            f'DEBUG: type of img_data is {type(img_data)}'
        )  # 여기서 <class 'str'>이 나오는지 확인

        # 성공 시 로직 처리
        return img_data

    except UnidentifiedImageError:
        return '이미지 형식을 인식할 수 없습니다. 유효한 이미지 파일인지 확인해주세요.'
    except Exception as e:
        return f'오류 발생: {str(e)}'
