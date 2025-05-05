
# yolo_snap.py
import os
import cv2
import time
from ultralytics import YOLO
import numpy as np
from Gemini_str import multimodalLLM  # LLM 처리 함수

model = YOLO("yolov5s.pt")
model.verbose = False

DANGEROUS_CLASSES = ['car', 'motorcycle', 'bicycle', 'bench', 'person']
TEMP_DIR = 'temp'

# YOLO 상태 관리
frames = []
detected_classes = set()
recording = False
record_start_time = None
frame_limit = 20
record_duration = 5  # seconds

processing_llm = False


def is_black_image(img, threshold=10):
    return np.mean(img) < threshold  # 전체 픽셀 평균이 매우 낮으면 검정


def get_next_folder_index(base_path=TEMP_DIR):
    existing = [int(name) for name in os.listdir(base_path) if name.isdigit()]
    return str(max(existing) + 1 if existing else 1)

def process_frame(img):
    global frames, detected_classes, recording, record_start_time, processing_llm

    if is_black_image(img):  # 검정 이미지 감지되면 무시
        return

    if processing_llm: #LLM 분석상태면 프래임 안받음
        return

    results = model(img)[0]
    current_classes = [model.names[int(cls)] for cls in results.boxes.cls]

    if not recording and any(obj in DANGEROUS_CLASSES for obj in current_classes):  # 위험 객체 확인시 시작
        recording = True
        record_start_time = time.time()
        frames = []
        detected_classes = set(current_classes)

    if recording:
        frames.append(img)
        detected_classes.update(current_classes)

        if (time.time() - record_start_time > record_duration) or len(frames) >= frame_limit: #조건 (20)장 에 맞는지 확인
            if len(frames) > frame_limit:
                step = len(frames) // frame_limit
                selected_frames = frames[::step][:frame_limit]
            else:
                selected_frames = frames

            folder_index = get_next_folder_index()
            save_dir = os.path.join(TEMP_DIR, folder_index)
            os.makedirs(save_dir, exist_ok=True)

            for i, f in enumerate(selected_frames):
                path = os.path.join(save_dir, f'frame_{i:02d}.jpg')
                cv2.imwrite(path, f)

            print(f"사진 {len(selected_frames)}장 저장 완료: {save_dir}, 객체: {list(detected_classes)}")

            processing_llm = True

                # ... 저장 및 LLM 호출
            try:
                summary = multimodalLLM(save_dir, detected_classes)
                print("yolo LLM 요약 결과:", summary)
                processing_llm = False
                return summary
            except Exception as e:
                print("LLM 호출 중 오류 발생:", str(e))

                # LLM 끝났으니 다시 활성화


            frames = []
            recording = False
            detected_classes = set()

            return summary
