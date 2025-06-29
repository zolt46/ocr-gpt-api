# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import easyocr
import openai
import os
import tempfile

app = FastAPI()

# CORS 설정 (필요 시 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCR 초기화 (한글 + 영어)
reader = easyocr.Reader(['ko', 'en'])

# GPT API 키 환경변수로 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/extract_inbody")
async def extract_inbody(file: UploadFile = File(...)):
    try:
        # 업로드된 파일을 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(await file.read())
            image_path = tmp.name

        # OCR 처리
        result = reader.readtext(image_path, detail=0)
        ocr_text = "\n".join(result)

        # GPT 프롬프트 구성
        prompt = f"""
다음 텍스트는 인바디 검사지에서 OCR로 추출된 결과입니다.
여기서 체중(kg), 체지방량(kg), 골격근량(kg) 수치만 추출해 아래 JSON 형식으로 응답해주세요.

텍스트:
{ocr_text}

출력 형식:
{{
  "weight": ...,
  "bodyFat": ...,
  "skeletalMuscle": ...
}}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 헬스 트레이너이며, 인바디 데이터를 구조화하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        reply = response.choices[0].message.content
        return {"raw_ocr": ocr_text, "extracted": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
