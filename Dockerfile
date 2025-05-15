FROM python:3.10-slim

# 필수 시스템 라이브러리 설치 (OpenCV, torch, matplotlib 등용)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt 복사 후 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 전체 복사 (app.py, live_yolo.py, Gemini_str.py 등)
COPY . .

# YOLO 모델 파일 복사 (yolov5s.pt이 반드시 있어야 함)
COPY yolov5s.pt ./yolov5s.pt

# Flask 앱 실행 (app.py에 __main__ 있음)
CMD ["python", "app.py"]
