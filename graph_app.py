# graph_app.py
from __future__ import annotations
import os
import traceback
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---- LangGraph 최소 구성 ----
try:
    # langgraph 0.2+ 기준
    from langgraph.graph import StateGraph, START, END
except Exception as e:
    raise SystemExit(
        "LangGraph가 설치되어 있지 않습니다. 먼저:\n"
        "  pip install 'langgraph>=0.2.0' fastapi uvicorn pydantic openai\n"
        f"원인: {type(e).__name__}: {e}"
    )

# ---- 선택적 OpenAI (키 없으면 더미 LLM로 자동 대체) ----
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        from openai import OpenAI

        oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    except Exception as e:
        # 키가 있어도 클라이언트 초기화 실패시 더미로 폴백
        print(f"[WARN] OpenAI 초기화 실패 -> 더미 LLM로 대체: {type(e).__name__}: {e}")
        USE_OPENAI = False


# =========================
# 1) 상태 정의
# =========================
class ERState(BaseModel):
    query: str
    context: List[str] = Field(default_factory=list)
    answer: str = ""
    error: Optional[str] = None


# =========================
# 2) 노드 정의
# =========================
def retrieve(state: ERState) -> ERState:
    """간단한 더미 retriever: 필요시 로컬 파일/DB로 교체"""
    try:
        q = state.query.strip()
        # 여기에 실제 검색(예: Qdrant/BM25)을 연결하면 됨.
        # 키/외부 의존성 없이도 데모되도록 더미 문맥 제공
        ctx = [
            "응급상황에서는 기도 유지와 출혈 통제가 우선입니다.",
            "흉통과 호흡곤란은 즉시 119에 연락하고 아스피린 복용 여부를 확인합니다.",
            "화상은 흐르는 미지근한 물로 10~20분 냉각합니다.",
        ]
        # 아주 단순한 필터링
        if "화상" in q:
            ctx = [c for c in ctx if "화상" in c] or ctx
        return state.copy(update={"context": ctx})
    except Exception as e:
        tb = traceback.format_exc()
        return state.copy(update={"error": f"[retrieve]{type(e).__name__}: {e}\n{tb}"})


def generate(state: ERState) -> ERState:
    """LLM 호출(가능하면 OpenAI, 없으면 더미 LLM)"""
    try:
        if state.error:
            # 이전 노드 오류가 있으면 그대로 끝내지 말고 메세지 포함하여 답변 작성
            prefix = "⚠️ 내부 오류가 있었지만 가능한 범위에서 답변합니다.\n\n"
        else:
            prefix = ""

        # 컨텍스트 조립
        ctx_text = (
            "\n".join(f"- {c}" for c in state.context)
            if state.context
            else "- (no context)"
        )
        prompt = (
            "You are an emergency assistant. Answer in Korean, concise and safe.\n\n"
            f"질문: {state.query}\n\n참고 문맥:\n{ctx_text}\n\n"
            "주의: 위험한 처치는 피하고 119 연락 기준을 분명히 해주세요."
        )

        if USE_OPENAI:
            # OpenAI가 가능하면 실제 LLM 호출
            resp = oai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
        else:
            # 더미 LLM: 키 없이도 돌아가게 간단 응답 생성
            safety_tail = "\n\n※ 생명 위협 증상이 있으면 즉시 119에 연락하세요."
            if "화상" in state.query:
                answer = (
                    "화상은 흐르는 미지근한 물로 10~20분간 냉각하세요. 물집은 터뜨리지 마세요."
                    + safety_tail
                )
            elif "흉통" in state.query or "가슴" in state.query:
                answer = (
                    "가슴 통증이 지속되면 즉시 119에 연락하세요. 진통제 임의 복용은 피하세요."
                    + safety_tail
                )
            else:
                answer = (
                    "증상이 심하거나 악화되면 119 또는 응급실로 가세요. 기본적인 안전조치를 먼저 수행하세요."
                    + safety_tail
                )

        return state.copy(update={"answer": prefix + answer})
    except Exception as e:
        tb = traceback.format_exc()
        return state.copy(update={"error": f"[generate]{type(e).__name__}: {e}\n{tb}"})


def route_on_error(state: ERState) -> str:
    """오류가 있으면 종료, 없으면 다음 단계"""
    return END if state.error else END  # 이후 확장 여지(예: 후처리 노드)


# =========================
# 3) 그래프 빌드
# =========================
def build_graph():
    graph = StateGraph(ERState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", route_on_error, {END: END})

    return graph.compile()


# =========================
# 4) FastAPI
# =========================
app = FastAPI(title="ER_NOW_119 (LangGraph minimal demo)")

graph = build_graph()


class AskReq(BaseModel):
    query: str


class AskRes(BaseModel):
    answer: str
    context: List[str]
    error: Optional[str]


@app.post("/ask", response_model=AskRes)
def ask(req: AskReq):
    try:
        init = ERState(query=req.query)
        final: ERState = graph.invoke(init)
        if final.error:
            # 에러도 함께 반환 (프론트에서 표시)
            return AskRes(
                answer=final.answer or "", context=final.context, error=final.error
            )
        return AskRes(answer=final.answer, context=final.context, error=None)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, detail=f"[server]{type(e).__name__}: {e}\n{tb}"
        )


@app.get("/ping")
def ping():
    return {
        "ok": True,
        "use_openai": USE_OPENAI,
        "model": (
            os.getenv("OPENAI_MODEL", "gpt-4o-mini") if USE_OPENAI else "dummy-llm"
        ),
        "langgraph": "ok",
    }


# 로컬 실행: uvicorn graph_app:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
