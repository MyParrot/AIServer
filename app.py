from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import os, base64, re, cv2, numpy as np
from live_yolo import process_frame  # YOLO 감지 함수
import requests
import shutil

# temp 폴더 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)  # Swagger 초기화

@app.route('/')
def index():
    return "Flask 서버 실행 중입니다. POST 요청은 /upload_frame 에 보내세요."

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """
    YOLO 프레임 전송 API
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            image:
              type: string
              description: base64로 인코딩된 이미지 데이터 (data:image/jpeg;base64,...)
    responses:
      200:
        description: 처리 결과
        schema:
          type: object
          properties:
            status:
              type: string
            message:
              type: string
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return {'status': 'error', 'message': '이미지가 없습니다.'}, 400

    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img_bytes = base64.b64decode(image_data)
    np_img = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    print("프레임 수신 완료:", img.shape)
    result = process_frame(img)
    if result is None:
        return jsonify({"status": "skipped"}), 200
    summary, image_path = result

    print('****', summary)
    print('****',image_path)

    folder_path = os.path.dirname(image_path)

    # 3. S3 업로드

    try:
        with open(image_path, 'rb') as file:
            files = {'file': (os.path.basename(image_path), file, 'image/jpeg')}
            data = {'userName': 'my_parrots_ai_gogo'}
            res = requests.post("http://11.11.1.1", files=files, data=data)

        if res.status_code == 200:
            s3_url = res.json()['data']
            return jsonify({
                "status": "success" if summary else "no_summary",
                "summary": summary,
                "s3_url": s3_url
            })
        else:
            return jsonify({
                "status": "upload_failed",
                "summary": summary,
                "error": res.text
            }), 500

    except Exception as e:
        return jsonify({
            "status": "success" if summary else "no_summary",
            "summary": summary,
            "s3_url": None
        })

    finally:
        try:
            folder_path = os.path.abspath(os.path.dirname(image_path))
            temp_root = os.path.abspath("temp")

            if os.path.exists(folder_path) and os.path.commonpath([folder_path, temp_root]) == temp_root:
                shutil.rmtree(folder_path)

            os.makedirs(temp_root, exist_ok=True)

        except Exception as cleanup_error:
            print(f"폴더 삭제 중 오류 발생: {cleanup_error}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
