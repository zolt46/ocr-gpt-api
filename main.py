# ✅ main.py - Google Vision API + GPT 연동

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import openai
import base64
import os
import tempfile
import json 

# ====== 환경변수 설정 ======
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")

# ====== FastAPI 초기화 ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Google Vision OCR 함수 ======
def call_google_vision(image_path):
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    body = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    response = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}",
        json=body
    )

    result = response.json()
    #print("📦 Vision API 응답:", json.dumps(result, indent=2, ensure_ascii=False))
    try:
        # 1순위: fullTextAnnotation
        if 'fullTextAnnotation' in result['responses'][0]:
            return result['responses'][0]['fullTextAnnotation']['text']
        # 2순위: textAnnotations
        elif 'textAnnotations' in result['responses'][0]:
            return result['responses'][0]['textAnnotations'][0]['description']
        else:
            return "OCR 실패 또는 텍스트 인식 불가"
    except:
        return "OCR 실패 또는 텍스트 인식 불가"

# ====== GPT 추출 프롬프트 ======
def extract_inbody_with_gpt(ocr_text):
    prompt = f"""
다음은 인바디 검사 결과지에서 OCR로 추출한 텍스트입니다: \n\n{ocr_text}\n\n

이 중에서 다음 수치만 정확히 찾아 JSON 형식으로 반환하세요:
- 체중(kg): weight
- 골격근량(kg): skeletalMuscle
- 체지방량(kg): bodyFat

❗ 단, 아래와 같은 올바른 JSON 형식으로 응답하세요. 숫자는 0이 아닌 이상 00.0이 아닌 **유효한 부동소수점 숫자**로 작성하세요.

예시:
{{
  "weight": 65.4,
  "skeletalMuscle": 28.2,
  "bodyFat": 12.7
}}
    """.strip()

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return completion.choices[0].message.content

# ====== 업로드 엔드포인트 ======
@app.post("/extract_inbody")
async def extract(file: UploadFile = File(...)):
    # 이미지 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 1. Google OCR 호출
    ocr_text = call_google_vision(tmp_path)
    #print("OCR 텍스트:\n", ocr_text)

    # 2. GPT에게 수치 추출 요청
    try:
        gpt_response = extract_inbody_with_gpt(ocr_text)
        return {"success": True, "data": gpt_response}
    except Exception as e:
        return {"success": False, "error": str(e), "ocr_text": ocr_text}
