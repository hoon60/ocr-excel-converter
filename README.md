# OCR-Excel 변환기

채권 수요예측표, 재무제표 등 이미지/PDF를 Excel로 변환하는 데스크톱 도구입니다.

## 주요 기능

- **다중 OCR 엔진**: PaddleOCR (한국어 92-96%) + EasyOCR + RapidOCR
- **Vision AI 폴백**: Gemini 2.5 Flash, Groq Llama 4 Scout
- **수요예측표 전용**: 스프레드 열 자동 인식, 열 합계 교차검증, OpenCV 격자 보정
- **재무제표 지원**: 재무상태표, 손익계산서 구조 인식
- **ML 학습**: 처리할수록 패턴 DB에 학습 → 정확도 자동 향상
- **GUI**: 드래그앤드롭으로 파일 변환

## 설치

### 방법 1: exe 실행 (Windows)

[Releases](../../releases) 페이지에서 최신 버전을 다운로드하세요.

### 방법 2: 소스에서 실행

```bash
pip install -r requirements.txt
python main.py
```

## API 키 설정

Vision AI를 사용하려면 `config.json`을 생성하세요:

```json
{
    "groq_api_key": "gsk_...",
    "gemini_api_key": "AIza...",
    "use_ai_by_default": true,
    "vision_engine": "auto"
}
```

- **Groq**: https://console.groq.com 에서 무료 API 키 발급
- **Gemini**: https://aistudio.google.com 에서 API 키 발급

## 지원 파일 형식

- 이미지: PNG, JPG, JPEG, BMP, TIFF, WebP
- PDF: 텍스트 PDF, 스캔 PDF

## 수요예측표 변환 검증

변환 결과의 정확성을 자동 검증합니다:
- 각 행의 스프레드 열 합산 = 총참여금액
- 각 열의 데이터 합산 = 합계 행 값
- 불일치 셀은 노란색으로 표시

## exe 빌드

```bash
pip install pyinstaller
pyinstaller build.spec --distpath dist_release
```

## 라이선스

MIT
