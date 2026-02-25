# flake8: noqa: F401

import json
import os
import shutil
import uuid

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()


@app.get('/')
def root():
    return {'result': 'Hi!!!'}


@app.post('/infer')
def infer(file: UploadFile | None = None):
    if file is None:
        raise HTTPException(status_code=400, detail='파일을 업로드하세요!!!')

    allowed_ext = ['jpg', 'jpeg', 'png', 'webp']
    ext = file.filename.split('.')[-1].lower()

    if ext not in allowed_ext:
        return {'error': '이미지 파일만 업로드하세요!!!'}

    new_file_name = f'{uuid.uuid4()}.{ext}'
    file_path = os.path.join('upload_img', new_file_name)

    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 추론 코드

    return {'result': '카리나', 'index': 2, 'filename': new_file_name}
