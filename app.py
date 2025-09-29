# app.py  — ER_NOW_119 Flask entrypoint (stable)
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# LangGraph 실행기 & RAG 예열
from graph_flow import run_agent
from tools.rag_store import warmup as rag_warmup
from tools.hira_api import hira_search
from tools.egen_api import egen_beds

# ---------------- Env & Paths ----------------
def _load_env():
    """.env를 현재 작업폴더→app.py 폴더→상위폴더에서 순차 탐색 로드"""
    for p in [Path.cwd() / ".env",
              Path(__file__).resolve().parent / ".env",
              Path(__file__).resolve().parent.parent / ".env"]:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)
            return
_load_env()


api_key = os.getenv("OPENAI_API_KEY")

APP_PORT = int(os.getenv("APP_PORT", "8000"))
BASE_DIR = Path(__file__).resolve().parent
MEM_DIR = BASE_DIR / "mem"
MEM_DIR.mkdir(parents=True, exist_ok=True)
SESS_FILE = MEM_DIR / "sessions.jsonl"        # 세션 히스토리 비식별 저장
SCEN_PATH = MEM_DIR / "corpus" / "scenarios.json"  # 상황 카드 JSON (있으면 제공)

# ---------------- Flask ----------------
app = Flask(__name__, static_folder="static", static_url_path="/")
app.config["JSON_AS_ASCII"] = False  # 한글 깨짐 방지
CORS(app)

# 로그 설정(간단)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# 인메모리 세션 히스토리(프론트 재표시용)
SESS: Dict[str, List[Dict[str, str]]] = {}

def _normalize_history(hist: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in hist or []:
        if isinstance(t, dict):
            out.append({"role": str(t.get("role", "")), "content": str(t.get("content", ""))})
        else:
            out.append({"role": str(getattr(t, "role", "")), "content": str(getattr(t, "content", ""))})
    return out

def save_history(session_id: str, history: Any) -> None:
    try:
        safe_hist = _normalize_history(history)
        with SESS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"session_id": session_id, "history": safe_hist}, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.warning(f"save_history error: {e}")

# --------- 서버 시작 시 RAG(임베딩/리랭커) 예열 ---------
rag_warmup()

# ---------------- Routes ----------------
@app.get("/")
def index():
    idx = Path(app.static_folder or ".") / "index.html"
    if idx.exists():
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"ok": True, "msg": "ER_NOW_119 server running"})

@app.get("/favicon.ico")
def favicon():
    path = Path(app.static_folder or ".") / "favicon.ico"
    if path.exists():
        return send_from_directory(app.static_folder, "favicon.ico")
    return ("", 204)

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/chat")
def api_chat():
    """
    입력 JSON:
      - q: 사용자 질문 (필수)
      - session_id: 세션 식별자 (선택, 기본 'sid')
      - lang: 'ko' 기본
      - history: 직전 대화 배열(선택)
      - assist_prompt: 선택 카드에서 전달되는 보조 프롬프트(선택)
    """
    try:
        body = request.get_json(force=True) or {}
        sid = body.get("session_id", "sid")
        q = (body.get("q") or body.get("query") or "").strip()
        lang = body.get("lang", "ko")
        assist = (body.get("assist_prompt") or "").strip()
        history = body.get("history") or SESS.get(sid, [])

        if not q:
            return jsonify({"error": "empty_query"}), 400

        user_query = q
        if assist:
            user_query = f"[선택된 상황 카드 요약]\n{assist}\n\n[사용자 질문]\n{q}"

        out = run_agent(session_id=sid, query=user_query, lang=lang, history=history)

        # 프런트 재표시용 메모리 히스토리 갱신
        SESS.setdefault(sid, []).append({"role": "user", "content": q})
        SESS[sid].append({"role": "assistant", "content": out.get("answer", "")})

        # 파일로도 비식별 저장
        save_history(sid, out.get("history", []))

        print(f"[DEBUG] out : {out}")

        return_obj = jsonify({
            "answer": out.get("answer", ""),
            # "alerts": out.get("alerts", []),
            # "ddx": out.get("ddx", []),
            # "retrieved": out.get("retrieved", []),
            # "beds": out.get("beds", []),
            # "hira": out.get("hira", []),
            # "metrics": out.get("metrics", {}),
            # "history": out.get("history", []),
        })

        print(f"[DEBUG] return object : {return_obj}")

        # 필요한 필드만 반환(프런트에서 바로 사용)
        return return_obj
    except Exception as e:
        logging.exception("chat error")
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---- 병원/과 단독 검색(프런트에서 별도 호출 시 유용) ----
@app.get("/api/search/hospitals")
def search_hospitals():
    try:
        dept_kw = request.args.get("dept")          # 예: 심장내과, 안과, 신경과...
        sigungu_code = request.args.get("sigungu")  # 행정코드(선택)
        specialized_only = request.args.get("specialized", "true").lower() != "false"
        result = hira_search(sigungu_code=sigungu_code, dept_kw=dept_kw, specialized_only=specialized_only)
        return jsonify({"hospitals": result})
    except Exception as e:
        logging.exception("hospitals search error")
        return jsonify({"hospitals": [], "error": str(e)}), 200

@app.get("/api/search/beds")
def search_beds():
    try:
        sido = request.args.get("sido")  # 예: 서울특별시
        sgg  = request.args.get("sgg")   # 예: 강남구
        result = egen_beds(sido=sido, sgg=sgg)
        return jsonify({"beds": result})
    except Exception as e:
        logging.exception("beds search error")
        return jsonify({"beds": [], "error": str(e)}), 200

# ---- 상황별 대처 카드: 목록/상세 ----
@app.get("/api/scenarios")
def api_scenarios():
    if not SCEN_PATH.exists():
        return jsonify({"cards": []})
    try:
        data = json.loads(SCEN_PATH.read_text(encoding="utf-8"))
        return jsonify({"cards": data.get("cards", [])})
    except Exception as e:
        return jsonify({"cards": [], "error": str(e)}), 200

@app.get("/api/scenario/<sid>")
def api_scenario(sid: str):
    if not SCEN_PATH.exists():
        return jsonify({"ok": False, "error": "no_scenarios"})
    try:
        data = json.loads(SCEN_PATH.read_text(encoding="utf-8"))
        card = next((c for c in data.get("cards", []) if c.get("id") == sid), None)
        if not card:
            return jsonify({"ok": False, "error": "not_found"})
        prompt = (
            f"출처:{card.get('source','')} / 라이선스:{card.get('license','')}.\n"
            f"안전 원칙: 진단 단정 금지, 119 신고 우선, 위험행위 금지, 근거 표기.\n"
            f"상황: {card.get('title')}\n"
            "핵심 단계:\n- " + "\n- ".join(card.get('steps', [])[:8]) + "\n\n"
            "사용자 상황을 반영해, 1)즉시 행동 2)주의/금기 3)119/응급실 이동 기준을 6줄 이내로 안내하세요."
        )
        return jsonify({"ok": True, "card": card, "prompt": prompt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200

# ---------------- Main ----------------
if __name__ == "__main__":
    # 로컬 개발용 (외부 노출 금지)
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
