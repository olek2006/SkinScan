import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"


def _load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_history(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_record(
    user_id: str,
    lesion_id: str,
    area_mm2: float,
    diameter_mm: float,
    A: float,
    B: float,
    C: float,
    risk: float
):

    history = _load_history()

    record = {
        "user_id": user_id,
        "lesion_id": lesion_id,
        "date": datetime.now().isoformat(),
        "area_mm2": round(area_mm2, 2),
        "diameter_mm": round(diameter_mm, 2),
        "A": round(A, 3),
        "B": round(B, 3),
        "C": round(C, 2),
        "risk": round(risk, 2)
    }

    history.append(record)
    _save_history(history)


def get_history(user_id: str | None = None, lesion_id: str | None = None):

    data = _load_history()

    if user_id is None and lesion_id is None:
        return data

    filtered = []
    for rec in data:
        if user_id is not None and rec.get("user_id") != user_id:
            continue
        if lesion_id is not None and rec.get("lesion_id") != lesion_id:
            continue
        filtered.append(rec)

    return filtered


def get_lesion_history(user_id: str, lesion_id: str):
    return get_history(user_id=user_id, lesion_id=lesion_id)


def get_last_record(user_id: str, lesion_id: str):
    hist = get_lesion_history(user_id, lesion_id)
    return hist[-1] if hist else None

