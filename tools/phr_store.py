# tools/phr_store.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# 데이터 저장 경로
BASE_DIR = Path(__file__).resolve().parents[1]
PHR_FILE = BASE_DIR / "mem" / "phr_data.json"

# 기본 데이터 구조
DEFAULT_PHR = {
    "profile": {
        "name": "",
        "birthdate": "",
        "gender": "",
        "blood_type": ""
    },
    "history": [],   # 수술/진료 기록
    "metrics": [],   # 건강 수치 (혈압, 혈당 등)
    "insurance": []  # 보험 정보
}

def _load_data() -> Dict[str, Any]:
    if not PHR_FILE.exists():
        return DEFAULT_PHR.copy()
    try:
        return json.loads(PHR_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"PHR load error: {e}")
        return DEFAULT_PHR.copy()

def _save_data(data: Dict[str, Any]) -> None:
    try:
        PHR_FILE.parent.mkdir(parents=True, exist_ok=True)
        PHR_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.error(f"PHR save error: {e}")

def get_phr() -> Dict[str, Any]:
    """전체 PHR 데이터 조회"""
    return _load_data()

def update_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """프로필 업데이트"""
    data = _load_data()
    data["profile"].update(profile_data)
    _save_data(data)
    return data

def add_history(item: Dict[str, Any]) -> Dict[str, Any]:
    """진료/수술 기록 추가"""
    data = _load_data()
    data.setdefault("history", []).append(item)
    _save_data(data)
    return data

def add_metric(item: Dict[str, Any]) -> Dict[str, Any]:
    """건강 수치 기록 추가 (예: 혈당, 혈압)"""
    data = _load_data()
    data.setdefault("metrics", []).append(item)
    _save_data(data)
    return data

def add_insurance(item: Dict[str, Any]) -> Dict[str, Any]:
    """보험 정보 추가"""
    data = _load_data()
    data.setdefault("insurance", []).append(item)
    _save_data(data)
    return data

def delete_item(category: str, index: int) -> Dict[str, Any]:
    """특정 카테고리의 항목 삭제"""
    data = _load_data()
    if category in data and isinstance(data[category], list):
        if 0 <= index < len(data[category]):
            data[category].pop(index)
            _save_data(data)
    return data
