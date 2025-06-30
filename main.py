from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import openai
import base64
import os
import tempfile
import json

# 🔑 환경 변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")

# ✅ FastAPI 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Google Vision OCR 함수
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
    try:
        if 'fullTextAnnotation' in result['responses'][0]:
            return result['responses'][0]['fullTextAnnotation']['text']
        elif 'textAnnotations' in result['responses'][0]:
            return result['responses'][0]['textAnnotations'][0]['description']
        else:
            return "OCR 실패 또는 텍스트 인식 불가"
    except:
        return "OCR 실패 또는 텍스트 인식 불가"

# ✅ GPT 인바디 수치 추출
def extract_inbody_with_gpt(ocr_text):
    prompt = f"""
다음은 인바디 검사 결과지에서 OCR로 추출한 텍스트입니다: \n\n{ocr_text}\n\n

이 중에서 다음 수치만 정확히 찾아 JSON 형식으로 반환하세요:
- 체중(kg): weight
- 골격근량(kg): skeletalMuscle
- 체지방량(kg): bodyFat

❗ 아래와 같은 JSON 형식으로 응답하세요:
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

# ✅ 샐러드 레시피 생성용 프롬프트 구성
def build_salad_recipe_prompt(user_info: dict) -> str:
    ALLOWED_INGREDIENTS = [
        "salmon", "avocado", "chicken breast", "lettuce", "spinach",
        "quinoa", "tomato", "cucumber", "olive oil", "lemon",
        "bell pepper", "pumpkin seeds", "egg", "tofu", "carrot"
    ]

    gender = user_info.get("gender", "선택 안함")
    inbody = user_info.get("inbody", {})
    weight = inbody.get("weight", 0.0)
    skeletal = inbody.get("skeletalMuscle", 0.0)
    fat = inbody.get("bodyFat", 0.0)
    noFood = user_info.get("noFood", [])
    goal = ", ".join(user_info.get("purpose", [])) or "건강한 식단"

    prompt = f"""
당신은 전문 영양사이자 샐러드 셰프입니다. 사용자의 건강 정보와 식습관 데이터를 기반으로 샐러드 레시피를 3가지 제안해주세요.
반드시 아래 재료 목록만 사용하며, 한국어로만 응답하세요.

[사용자 정보]
- 성별: {gender}
- 체중: {weight}kg
- 골격근량: {skeletal}kg
- 체지방량: {fat}kg
- 제외 재료: {', '.join(noFood) if noFood else "없음"}
- 목적: {goal}

[허용된 재료 목록]
{', '.join(ALLOWED_INGREDIENTS)}

응답 형식:
1. 샐러드 이름
2. 재료 목록
3. 영양 정보 (칼로리, 단백질, 탄수화물, 지방)
    """.strip()
    return prompt


# ✅ /extract_inbody - OCR + GPT 수치 추출
@app.post("/extract_inbody")
async def extract(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    ocr_text = call_google_vision(tmp_path)

    try:
        gpt_response = extract_inbody_with_gpt(ocr_text)
        return {"success": True, "data": gpt_response}
    except Exception as e:
        return {"success": False, "error": str(e), "ocr_text": ocr_text}

# ✅ /generate_recipe - 종합 정보로 GPT 레시피 생성
@app.post("/generate_recipe")
async def generate_recipe(request: Request):
    user_data = await request.json()
    prompt = build_salad_recipe_prompt(user_data)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return {
            "success": True,
            "recipe": response["choices"][0]["message"]["content"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

