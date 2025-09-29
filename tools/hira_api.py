"""
HIRA 병원정보 API(요약)
- 실제 상세(전문의 수/전문병원 지정분야)는 추가 엔드포인트 필요
- 여기서는 기본 목록을 호출하고, 키워드 필터를 간단히 적용
"""
import os, requests

HIRA_KEY = (os.getenv("HIRA_API_KEY") or "").strip()
BASE = "https://apis.data.go.kr/B551182/hospInfoServicev2"

def hira_search(sigungu_code=None, dept_kw=None, specialized_only=True, page_no=1, num_rows=10):
    url = f"{BASE}/getHospBasisList"
    params = {"serviceKey": HIRA_KEY, "_type": "json", "pageNo": page_no, "numOfRows": num_rows}
    if sigungu_code: params["sgguCd"] = sigungu_code
    if not HIRA_KEY:
        return []
    try:
        r = requests.get(url, params=params, timeout=8)
        items = r.json().get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
        # 간단 키워드 필터(실서비스: 상세 API로 보강)
        if dept_kw:
            items = [x for x in items if dept_kw in (x.get("clCdNm","") + x.get("yadmNm",""))]
        return items
    except Exception:
        return []
