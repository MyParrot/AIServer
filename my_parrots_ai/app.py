
import os

# 현재 파일 위치 기준으로 temp 폴더 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)


from flask import Flask, request
from flask_cors import CORS
import base64
import re
import cv2
import numpy as np
from live_yolo import process_frame  # YOLO 감지 함수

app = Flask(__name__)
CORS(app)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return {'status': 'error', 'message': '이미지가 없습니다.'}, 400

    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img_bytes = base64.b64decode(image_data)
    np_img = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    print("프레임 수신 완료:", img.shape)
    process_frame(img)

    return {'status': 'received'}

if __name__ == '__main__':
    app.run(debug=True)
