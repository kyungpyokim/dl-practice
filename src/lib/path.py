from pathlib import Path

__all__ = ['model_file_path', 'data_file_path']

base_path = Path.cwd().parent.parent

# 현재 파일(노트북)의 위치를 기준으로 상위 부모 디렉토리로 이동
# .parent를 두 번 사용하여 notebook/lecture -> notebook -> 루트로 이동

# 디렉토리가 없으면 생성 (exist_ok=True는 이미 있으면 무시함)
# model_path.mkdir(parents=True, exist_ok=True)

# 저장 예시 (예: torch, sklearn, tensorflow 등)
# file_path = model_path / 'my_model.pth'
# print(f'모델 저장 경로: {file_path}')


def model_file_path(file_name):
    path = base_path / 'models'
    return path / file_name


def data_file_path(file_name=None):
    path = base_path / 'data'

    return path if file_name is None else path / file_name


def output_path():
    return base_path / 'output'
