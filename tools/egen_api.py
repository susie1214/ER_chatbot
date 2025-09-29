"""
E-Gen Open API 래퍼(간단)
- Unauthorized 방지: 서비스키 공백 제거, _type=json, 한글 파라미터는 requests의 params에 그대로(자동 인코딩)
"""
import os, requests

EGEN_KEY = (os.getenv("EGEN_API_KEY") or "").strip()
BASE = "http://apis.data.go.kr/B552657/ErmctInfoInqireService"

def egen_beds(sido=None, sgg=None):
    url = f"{BASE}/getEmrrmRltmUsefulSckbdInfoInqire"
    params = {"serviceKey": EGEN_KEY, "_type": "json"}
    if sido: params["STAGE1"] = sido
    if sgg:  params["STAGE2"] = sgg
    if not EGEN_KEY:
        return []
    try:
        r = requests.get(url, params=params, timeout=8)
        j = r.json()
        return j.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
    except Exception:
        return []
