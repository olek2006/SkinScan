from datetime import datetime
from history import get_history


def _months_between(d1: str, d2: str) -> float:

    t1 = datetime.fromisoformat(d1)
    t2 = datetime.fromisoformat(d2)
    return abs((t2 - t1).days) / 30.44


def evolution_score(user_id: str, lesion_id: str):

    history = get_history(user_id, lesion_id)

    if len(history) < 2:
        return {
            "E": 0,
            "status": "Недостатньо даних",
            "rate_mm2_per_month": 0.0
        }

    first = history[0]
    last = history[-1]

    months = _months_between(first["date"], last["date"])
    if months == 0:
        return {
            "E": 0,
            "status": "Недостатньо часу",
            "rate_mm2_per_month": 0.0
        }

    delta_area = last["area_mm2"] - first["area_mm2"]
    rate = delta_area / months


    if rate < 2:
        E = 0
        status = "Стабільно"
    elif rate < 10:
        E = 1
        status = "Повільне зростання"
    else:
        E = 2
        status = "Швидке зростання ⚠️"

    return {
        "E": E,
        "status": status,
        "rate_mm2_per_month": round(rate, 2),
        "months_between": round(months, 2)
    }

