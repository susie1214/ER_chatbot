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
├── app.py # FastAPI 진입점
├── graph_app.py # LangGraph RAG 파이프라인 (대표 실행 파일)
├── requirements.txt # 의존성 패키지
├── README.md # 프로젝트 설명
├── .gitignore # Git 무시 규칙
├── .env.example # 환경 변수 예시 (API 키 미포함)
│
├── providers/ # LLM, Embedding, API Provider 모듈
├── tools/ # 크롤링, OCR, 데이터 처리 유틸
├── static/ # 정적 파일 (UI 등)
├── data/ # (Git 무시) 질병/증상 데이터셋
└── qdrant_db/ # (Git 무시) 벡터 DB 저장소
```
## 🖼️ 화면 예시
<img width="379" height="138" alt="image" src="https://github.com/user-attachments/assets/8d38ff51-d134-47ef-9c5a-bd4b69387934" />
<img width="1323" height="647" alt="image" src="https://github.com/user-attachments/assets/e06a0aff-0b87-4ec5-be54-365f0adfbfb4" />
<img width="1324" height="646" alt="image" src="https://github.com/user-attachments/assets/9409648b-37d9-4f72-b036-ee6f4d39dbcf" />


