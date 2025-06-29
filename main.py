from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import tempfile
import openai
import os
import json
from fastapi.responses import JSONResponse

# 🔐 OpenAI 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ✅ CORS 허용 (브라우저에서 요청 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OCR 수행 함수
def perform_ocr(image_path: str) -> str:
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang="eng+kor")
        return text
    except Exception as e:
        return f"OCR 실패: {str(e)}"

# ✅ API: 이미지 업로드 → OCR → GPT → 인바디 수치 추출
@app.post("/extract_inbody")
async def extract_inbody(file: UploadFile = File(...)):
    # 1. 이미지 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 2. OCR 실행
    ocr_text = perform_ocr(tmp_path)

    # 3. GPT에게 체중, 골격근량, 체지방량 추출 요청
    prompt = f"""
다음은 인바디 검사 결과지의 OCR 추출 텍스트입니다:\n\n{ocr_text}\n\n
이 내용 중에서 아래 3가지 항목의 수치를 추출해주세요:
- 체중(kg): weight
- 골격근량(kg): skeletalMuscle
- 체지방량(kg): bodyFat

반드시 아래와 같은 JSON 형식으로만 응답해주세요:
{{
  "weight": 00.0,
  "skeletalMuscle": 00.0,
  "bodyFat": 00.0
}}
    """.strip()

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result_str = response["choices"][0]["message"]["content"]

        # 문자열을 실제 JSON 객체로 파싱
        parsed_result = json.loads(result_str)

        # 그대로 JSON으로 응답
        return JSONResponse(content=parsed_result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "ocr_text": ocr_text})
