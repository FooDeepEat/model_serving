from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import base64

app = FastAPI()

# 모델 로드
model = tf.keras.models.load_model("mobilenet_v92t98.h5")

# 라벨 불러오기
# 라벨을 숫자 형식이 아닌 텍스트 형식으로 불러옵니다 : 01~과 같은 값이 있기 때문
labels = pd.read_csv('labeldata16.csv', dtype=str, encoding='cp949')
# 클래스 이름이 label이라는 데이터 프레임의 소분류 컬럼의 값으로 지정되어있습니다(예: 쌀밥) = 해당 값을 리스트로 반환합니다
class_names = labels.소분류.values
# 데이터 프레임 라벨을 리스트 형태로 변환
class_names = class_names.tolist()
# print(class_names)


class ImageHandler:

    def __init__(self, food_img):
        self.food_img = food_img


    # 사용자로부터 받은 이미지의 각 색상 채널을 0~255의 값이 아닌
    # 0~1의 사이로 바꿔줍니다
    def rescale(self):
        food_img = self.food_img / 255.0
        return food_img

    # 사용자로부터 입력받은 사진을 왼쪽 20%, 오른쪽 20%를 자르고,
    # 크기를 모델에 맞는 224,224 사이즈로 만들어 줍니다
    def crop_and_resize(self):
        # 이미지 크기
        height, width = self.food_img.shape[:2]
        # crop 영역 설정
        left = int(width * 0.2)
        right = int(width * 0.8)
        # crop 수행
        img_cropped = self.food_img[:, left:right, :]
        # resize 수행
        img_resized = cv2.resize(img_cropped, (224, 224))
        food_img = img_resized
        return food_img

    # cvtColor : cv2의 경우 이미지 색상 채널을 BGR 순서로 불러올 수 있어
    # BGR 순서로 된 경우 RGB 순서로 바꿔줍니다
    ## 체크 - astype float 32 순서가 중요
    def cvtcolor(self):
        # 입력된 이미지의 데이터 타입이 OpenCV에서 지원하지 않는 CV_64F (64-bit float) 형식
        # cvtColor 함수를 사용하기 전에 이미지 데이터를 float32 형식으로 변환
        food_img = self.food_img.astype(np.float32)
        food_img = cv2.cvtColor(food_img, cv2.COLOR_BGR2RGB)
        return food_img


@app.post('/cnn_model')
async def cnn_model(uploaded_files: List[UploadFile] = File(...)) -> JSONResponse:
    # 결과와 이미지를 딕셔너리에 담아 리스트에 추가
    image_with_results = []
    # 업로드된 파일의 객체를 바이너리로 이미지 데이터를 읽는다.
    for uploaded_file in uploaded_files:
        content = await uploaded_file.read()

    #     # 바이너리 데이터를 이미지로 디코딩를 한다.
        decode_img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    #     # 이미지 전처리 과정
        crop_resize_img = ImageHandler(decode_img).crop_and_resize()
        cvtcolor_img = ImageHandler(crop_resize_img).cvtcolor()
        rescale_img = ImageHandler(cvtcolor_img).rescale()

    #     # 모델 실행 및 결과 예측
        pred = model.predict(np.array([rescale_img]))
        class_index = np.argmax(pred)
        class_name = class_names[class_index]
    #
    #     # 바이너리 이미지를 인코딩
        encode_content = base64.b64encode(content).decode('utf-8')
    #
        image_with_result = {
                'image': encode_content,
                'food_name': class_name
        }
        image_with_results.append(image_with_result)
        for idx, image_with_result in enumerate(image_with_results, start=1):
            print(idx, image_with_result['food_name'])
    #
    return JSONResponse(content=image_with_results)
