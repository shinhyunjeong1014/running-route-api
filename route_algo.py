# route_algo.py
import math
import os
import json
import logging
from typing import List, Dict, Tuple
import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 거리 계산
# -----------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2 +
        math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -----------------------------
# Valhalla API 호출
# -----------------------------
VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")

def valhalla_route(p1: Tuple[float, float], p2: Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Valhalla로 '도보 전용' 경로 요청.
    """
    body = {
        "locations": [
            {"lat": p1[0], "lon": p1[1]},
            {"lat": p2[0], "lon": p2[1]},
        ],
        "costing": "pedestrian",
        "directions_options": {"units": "meters"}
    }

    try:
        r = requests.post(VALHALLA_URL, json=body, timeout=10)
        data = r.json()
        shape = data["trip"]["legs"][0]["shape"]

        # shape polyline decode
        coords = []
        lat = 0
        lng = 0
        idx = 0

        while idx < len(shape):
            b = 1
            shift = 0
            result = 0
            while b >= 0x20:
                b = ord(shape[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            b = 1
            shift = 0
            result = 0
            while b >= 0x20:
                b = ord(shape[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            coords.append((lat / 1e5, lng / 1e5))

        return coords

    except Exception as e:
        logger.error(f"[Valhalla] error: {e}")
        return []


# -----------------------------
# 루프 점수 계산
# -----------------------------
def compute_loop_score(length_m, target_m, uturn_cnt, roundness):
    """
    러닝 루프 점수: 낮을수록 좋다.
    """
    err = abs(length_m - target_m)
    return (
        err +
        uturn_cnt * 300 +
        (1 - roundness) * 200
    )


# -----------------------------
# Area-Loop (옵션 A)
# -----------------------------
def generate_area_loop(lat: float, lng: float, km: float):
    target_m = km * 1000
    MAX_ERR = 99
    MAX_OUTER = 12
    MAX_INNER = 10

    base_r = max(280, min(420, target_m / 7))
    center = (lat, lng)

    best_score = 99999999
    best_route = []
    best_meta = {}

    for outer in range(MAX_OUTER):
        R = base_r + outer * 22

        for inner in range(MAX_INNER):
            via_cnt = 3 + (inner % 3)

            angle_step = 360 / via_cnt
            rad_points = []
            for i in range(via_cnt):
                ang = math.radians(i * angle_step)
                rad_lat = lat + (R / 111111) * math.cos(ang)
                rad_lng = lng + (R / (111111 * math.cos(math.radians(lat)))) * math.sin(ang)
                rad_points.append((rad_lat, rad_lng))

            full = [(lat, lng)]
            uturn_cnt = 0

            for i in range(via_cnt):
                seg = valhalla_route(full[-1], rad_points[i])
                if not seg:
                    seg2 = valhalla_route(full[-1], rad_points[i])
                    if not seg2:
                        uturn_cnt += 1
                        continue
                    seg = seg2

                full.extend(seg[1:])

            # 마지막: 출발지로 복귀
            back_seg = valhalla_route(full[-1], (lat, lng))
            if back_seg:
                full.extend(back_seg[1:])
            else:
                uturn_cnt += 1

            if len(full) < 4:
                continue

            length_m = 0
            for i in range(1, len(full)):
                length_m += haversine_m(full[i-1][0], full[i-1][1], full[i][0], full[i][1])

            err = abs(length_m - target_m)
            if err <= MAX_ERR:
                roundness = min(0.99, length_m / (2*math.pi*R))
                score = compute_loop_score(length_m, target_m, uturn_cnt, roundness)

                meta = {
                    "outer": outer,
                    "inner": inner,
                    "base_r": R,
                    "via": via_cnt,
                    "len": length_m,
                    "err": err,
                    "uturn": uturn_cnt,
                    "round": roundness,
                    "score": score,
                }

                logger.info(f"[AreaLoop] MATCH outer={outer}, inner={inner}, {meta}")

                if score < best_score:
                    best_score = score
                    best_route = full[:]
                    best_meta = meta

            else:
                logger.info(f"[AreaLoop] outer={outer}, inner={inner}, len={length_m:.1f}, err={err:.1f}")

    if not best_route:
        logger.warning("[AreaLoop] No loop in ±99m — returning best effort")

    return best_route, best_meta
