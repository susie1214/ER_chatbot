# tools/rag_store.py
# RAG 인덱스/검색: AMC(asan_emergency.json) + KDCA(er_kdca_merged.json)
# - 임베딩: intfloat/multilingual-e5-base (쿼리/문서 prefix 사용)
# - 벡터: FAISS 저장(data/embeddings/rag_index/)
# - BM25: data/lexical/bm25.pkl
# - 하이브리드: FAISS k + BM25 k 점수정규화 → 가중합(alpha)
# - 검색 API: rag_search(query, k=5, alpha=0.7, category="응급질환")
from __future__ import annotations

import json, re, pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
AMC_JSON = DATA_DIR / "asan_emergency.json"  # 새로 크롤링한 응급질환
KDCA_JSON = DATA_DIR / "er_kdca_merged.json"  # KDCA 병합본
VEC_DIR = DATA_DIR / "embeddings" / "rag_index"  # FAISS 저장 위치
LEX_DIR = DATA_DIR / "lexical"  # BM25 저장 위치
LEX_DIR.mkdir(parents=True, exist_ok=True)
BM25_PKL = LEX_DIR / "bm25.pkl"

# 임베딩: e5 계열(다국어/한국어 강함)
EMB_MODEL = "intfloat/multilingual-e5-base"

# 전역 핸들
_emb: Optional[HuggingFaceEmbeddings] = None
_vs: Optional[FAISS] = None
_bm25: Optional[BM25Okapi] = None
_docs: List[Document] = []
_raw_texts: List[str] = []

# ──────────────────────────────────────────────────────────────
# 텍스트/토큰 유틸
_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def _simple_tokens(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


# ──────────────────────────────────────────────────────────────
# 로더: AMC
def _load_amc(path: Path) -> List[Document]:
    """asan_emergency.json → Document 목록(passages)"""
    if not path.exists():
        return []
    js = json.loads(path.read_text(encoding="utf-8"))
    items = js.get("items", [])
    docs: List[Document] = []
    for it in items:
        title = _clean(it.get("title", ""))
        url = it.get("url", "")
        cid = it.get("id", "")
        src = it.get("source", js.get("source", "AMC"))
        cat = it.get("category", js.get("category", "응급질환"))
        secs: Dict[str, str] = it.get("sections", {})
        for sec, body in secs.items():
            text = _clean(body)
            if not text:
                continue
            # e5 권장 prefix: passage:
    # 본문: 값들 중 문자열만 모음(너무 짧은 건 제외)
    parts: List[str] = []
    for k, v in d.items():
        if isinstance(v, str) and len(v.strip()) > 3:
            parts.append(f"{k}:\n{_clean(v)}")
    body = "\n".join(parts)
    return title, body


def _load_kdca(path: Path) -> List[Document]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    # list 혹은 {"items":[...]} 모두 허용
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    docs: List[Document] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title, body = _flatten_strings(it)
        if not body:
            continue
        content = f"passage: [KDCA] {title}\n{body}"
        meta = {
            "id": f"KDCA::{it.get('id','')}",
            "title": title or "KDCA",
            "section": "본문",
            "url": "",
            "source": "KDCA",
            "category": "공공지침",
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs


# ──────────────────────────────────────────────────────────────
# 인덱스 생성/로드
def warmup(rebuild: bool = False) -> None:
    """
    서버 시작 시 호출.
    - 인덱스 존재: 로드
    - 없거나 rebuild=True: AMC+KDCA 문서를 로딩해 새 인덱스 생성
    """
    global _emb, _vs, _bm25, _docs, _raw_texts
    if _emb is None:
        _emb = HuggingFaceEmbeddings(
            model_name=EMB_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    if VEC_DIR.exists() and not rebuild:
        _vs = FAISS.load_local(str(VEC_DIR), _emb, allow_dangerous_deserialization=True)
        _docs = list(_vs.docstore._dict.values())
        _raw_texts = [d.page_content for d in _docs]
        # BM25 로드(없으면 생성)
        if BM25_PKL.exists():
            pack = pickle.loads(BM25_PKL.read_bytes())
            _bm25 = pack["bm25"]
            _raw_texts = pack["texts"]
        else:
            _bm25 = BM25Okapi([_simple_tokens(t) for t in _raw_texts])
            BM25_PKL.write_bytes(pickle.dumps({"bm25": _bm25, "texts": _raw_texts}))
        return

    # 새로 구성
    amc_docs = _load_amc(AMC_JSON)
    kdca_docs = _load_kdca(KDCA_JSON)
    _docs = amc_docs + kdca_docs
    _raw_texts = [d.page_content for d in _docs]

    _vs = FAISS.from_documents(_docs, _emb)
    VEC_DIR.parent.mkdir(parents=True, exist_ok=True)
    _vs.save_local(str(VEC_DIR))

    _bm25 = BM25Okapi([_simple_tokens(t) for t in _raw_texts])
    BM25_PKL.write_bytes(pickle.dumps({"bm25": _bm25, "texts": _raw_texts}))


# 내부: FAISS 상위 k (점수는 “높을수록 좋음”으로 변환)
def _score_faiss(query: str, k: int) -> List[Tuple[int, float]]:
    assert _vs is not None
    q = f"query: {query}"
    hits = _vs.similarity_search_with_score(q, k=k)
    # docstore 순서로 원래 인덱스 복구
    id2i = {
        doc.metadata.get("id", "") + doc.metadata.get("section", ""): i
        for i, doc in enumerate(_docs)
    }
    out: List[Tuple[int, float]] = []
    for doc, dist in hits:
        key = doc.metadata.get("id", "") + doc.metadata.get("section", "")
        idx = id2i.get(key, None)
        if idx is not None:
            out.append((idx, -float(dist)))  # 거리를 음수로 → 값이 클수록 유사
    return out


# 내부: BM25 상위 k
def _score_bm25(query: str, k: int) -> List[Tuple[int, float]]:
    assert _bm25 is not None
    scores = _bm25.get_scores(_simple_tokens(query))
    idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in idx]


def _norm(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    vals = np.array([s for _, s in pairs], dtype=float)
    mx, mn = float(vals.max()), float(vals.min())
    if mx == mn:
        return {i: 1.0 for i, _ in pairs}
    return {
        pairs[i][0]: float((vals[i] - mn) / (mx - mn + 1e-9)) for i in range(len(pairs))
    }


def rag_search(
    query: str,
    k: int = 5,
    alpha: float = 0.7,
    category: Optional[str] = None,
    **kwargs,  # 호환: final_k 등 옛 파라미터 무시하지 않도록
) -> List[Dict[str, Any]]:
    """
    하이브리드 검색(FAISS+BM25, 점수 정규화 후 가중합).
    - category 가 주어지면 metadata.category 로 필터링.
    - 이전 코드 호환: final_k가 들어오면 k로 치환.
    반환: [{title, section, snippet, url, source, meta, score}, ...]
    """
    if "final_k" in kwargs:
        k = int(kwargs.get("final_k") or k)

    if _vs is None or _bm25 is None:
        warmup()

    fa = _score_faiss(query, k=max(8, k * 2))
    bm = _score_bm25(query, k=max(8, k * 2))

    fa_n = _norm(fa)
    bm_n = _norm(bm)

    fused: List[Tuple[int, float]] = []
    keys = set(fa_n) | set(bm_n)
    for i in keys:
        fused.append((i, alpha * fa_n.get(i, 0.0) + (1 - alpha) * bm_n.get(i, 0.0)))
    fused.sort(key=lambda x: x[1], reverse=True)

    out: List[Dict[str, Any]] = []
    for idx, sc in fused:
        d = _docs[idx]
        if category and d.metadata.get("category") != category:
            continue
        full_text = d.page_content
        # 'passage: [제목] 본문' 형태에서 'passage: ' 제거
        clean_text = full_text.replace("passage: ", "").strip()
        
        out.append(
            {
                "title": d.metadata.get("title"),
                "section": d.metadata.get("section"),
                "snippet": clean_text[:420],
                "text": clean_text,  # ✅ 호환용 필드 추가 (passage 제거됨)
                "url": d.metadata.get("url"),
                "source": d.metadata.get("source"),
                "score": round(float(sc), 4),
                "meta": d.metadata,
            }
        )
        if len(out) >= k:
            break
    return out
