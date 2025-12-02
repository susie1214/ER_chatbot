# 🚑 ER_NOW_119

> 🏥 **실시간 응급 진료 챗봇 & 병원 안내 시스템**  
> 환자의 증상 입력 → 응급 대응법, 복용약, 진료과 안내, 인근 병원 정보까지 제공하는 AI 기반 서비스  

---

## 📌 프로젝트 개요
ER_NOW_119는 응급 상황에서 **빠르고 정확한 대처 방법**을 안내하는 챗봇입니다.  
증상에 따른 응급 처치법과 복용약, 진료과를 추천하며, 가까운 운영 병원의 위치와 전화번호까지 제공합니다.

---

## 🩺 주요 기능
- 📝 **증상 입력** → AI 기반 응급 대처 방법 안내  
- 💊 **복용약 추천** (예: 해열제, 진통제, 응급 약물)  
- 🧑‍⚕️ **진료과 안내** (예: 신경과, 외과, 피부과 등)  
- 📍 **지도 기반 병원 찾기** (현재 위치 기반, 운영 중인 병원만 표시)  
- ☎️ **전화 연결 지원**  

---

## ⚙️ 기술 스택
- **Backend** : Python, FastAPI / Flask  
- **Frontend** : HTML, CSS, JS (React 확장 가능)  
- **Database** : SQLite / PostgreSQL  
- **API** : 공공 데이터 API (병원/약품 정보) + 크롤링  
- **AI** : 증상 분석 및 응급 대처 추천 (LLM, RAG 적용)  

---

## 📂 프로젝트 구조
```bash
ER_NOW_119/
├─ app.py                     # Flask 서버 진입점
├─ app_langchain.py           # LangChain 연동 (선택적)
├─ static/
│   ├─ index.html             # 메인 UI (채팅, PHR, 병원 찾기)
│   └─styles.css          
├─ tools/
│   ├─ phr_store.py           # 개인 건강 기록 API
│   └─ rag_store.py           # RAG(검색·임베딩) API
├─ providers/
│   ├─ openai_client.py       # OpenAI API 클라이언트
│   ├─ gemini_client.py       # Gemini API 클라이언트
│   └─ vllm_client.py         # VLLM 연동
├─ scripts/                   # 독립 실행 스크립트 (데이터 전처리·실험)
│   ├─ crawling_kdca.py
│   ├─ ocr_merge_build.py
│   └─ pdf_embede.py
├─ mem/
│   ├─ sessions.jsonl         # 세션 히스토리 (비식별)
│   ├─ phr_data.json          # PHR 로컬 저장소
│   └─ corpus/
│       └─ scenarios.json     # 시나리오 카드 데이터
├─ models/
│   └─ bpe_tokenizer/…        # 토크나이저 파일
├─ ops/
│   └─ logger.py              # 로깅 헬퍼
├─ providers/                 # 외부 API 래퍼
├─ qdrant_db/                 # 로컬 벡터 DB (옵션)
└─  requirements.txt           # Python 의존성
```
## 🖼️ 화면 예시
<img width="1895" height="919" alt="image" src="https://github.com/user-attachments/assets/f971ce38-a09a-4b99-af7c-c32c18519e2b" />
<img width="1323" height="647" alt="image" src="https://github.com/user-attachments/assets/e06a0aff-0b87-4ec5-be54-365f0adfbfb4" />
<img width="1324" height="646" alt="image" src="https://github.com/user-attachments/assets/9409648b-37d9-4f72-b036-ee6f4d39dbcf" />


