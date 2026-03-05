import base64

from pyprojroot import here

root = here()


def save_key(key):
    rb = base64.urlsafe_b64decode(key)

    with open('secret.key', 'wb') as f:
        f.write(rb)

    print('✅ secret.key 파일이 생성되었습니다. 구글 드라이브에 업로드하세요.')


def load_key(file_name):
    path = root / file_name
    with open(str(path), 'rb') as f:
        data = f.read()

    # 순수 바이너리를 다시 Fernet이 이해하는 Base64 형태로 인코딩
    return base64.urlsafe_b64encode(data).decode('utf-8')


if __name__ == '__main__':
    # save_key('nF00I3dYwshLL9WnalIUUyMXJ7CIkRhJUa1-Hfg_2jg=')
    print(load_key('secret.key'))
