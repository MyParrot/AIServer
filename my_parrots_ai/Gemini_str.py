
import os
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
google_api_key = os.getenv("GOOGLE_API_KEY")

# Gemini API 키 설정 (본인의 API 키로 변경)
genai.configure(api_key=google_api_key)


def multimodalLLM(image_folder_path, detected_classes):
    """
    이미지 폴더 경로와 감지된 객체 정보를 받아 Gemini Pro Vision으로 분석.

    Args:
        image_folder_path (str): 이미지 파일들이 있는 폴더 경로 ex)temp/1
        detected_classes (list[str]): YOLO로 감지된 객체들

    Returns:
        str: 분석 결과 텍스트
    """

    # 이미지 파일 경로 정렬
    image_paths = sorted([
        os.path.join(image_folder_path, fname)
        for fname in os.listdir(image_folder_path)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_paths:
        return "폴더 안에 이미지가 없습니다."

    # Gemini 모델 준비
    model = genai.GenerativeModel('gemini-1.5-flash')
    images = [PIL.Image.open(path) for path in image_paths]

    # 기본 프롬프트 불러오기
    with open('prompt', "r", encoding='utf-8') as f:
        prompt_text = f.read()

    # 감지된 객체를 프롬프트에 포함
    if detected_classes:
        detected_str = ', '.join(set(detected_classes))
        object_intro = f"\n\n[참고 정보]\n이 이미지들에서 감지된 객체는 다음과 같습니다: {detected_str}\n"
    else:
        object_intro = "\n\n[참고 정보]\n감지된 위험 객체는 없습니다.\n"

    # 전체 프롬프트 구성
    prompt_parts = [prompt_text + object_intro] + images

    # Gemini에 요청
    response = model.generate_content(prompt_parts)
    return response.text
