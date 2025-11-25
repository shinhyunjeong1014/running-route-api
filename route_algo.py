import math
import os
import logging
from typing import List, Dict, Tuple

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------------------
# 거리 계산
# -----------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -----------------------------------------
# Valhalla API
# -----------------------------------------
VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "10"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "1"))  # retry 줄임


def _decode_polyline(shape: str):
    coords = []
    lat = 0
    lng = 0
    idx = 0

    while idx < len(shape):
        b, shift, result = 1, 0, 0
        while b >= 0x20:
            b = ord(shape[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        b, shift, result = 1, 0, 0
        while b >= 0x20:
            b = ord(shape[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))
    return coords


def valhalla_route(p1, p2):
    body = {
        "locations": [
            {"lat": p1[0], "lon": p1[1]},
            {"lat": p2[0], "lon": p2[1]},
        ],
        "costing": "pedestrian",
        "directions_options": {"units": "meters"},
    }

    last_error = None
    for attempt in range(VALHALLA_MAX_RETRY + 1):
        try:
            r = requests.post(VALHALLA_URL, json=body, timeout=VALHALLA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            shape = data["trip"]["legs"][0]["shape"]
            coords = _decode_polyline(shape)
            if len(coords) < 2:
                raise ValueError("decoded polyline too short")
            return coords
        except Exception as e:
            last_error = e
            logger.warning(f"[Valhalla] attempt {attempt+1} failed: {e}")

    return []


# -----------------------------------------
# 루프 점수
# -----------------------------------------
def compute_loop_score(length_m, target_m, fail_cnt, roundness):
    err = abs(length_m - target_m)

    if err <= 150:
        dist_penalty = err
    else:
        dist_penalty = 150 + (err - 150) * 2.0

    round_penalty = (1 - roundness) * (120 if roundness >= 0.7 else 220)

    return dist_penalty + fail_cnt * 350 + round_penalty


def _estimate_roundness(length_m: float, R: float) -> float:
    if R <= 0:
        return 0.0
    ideal = 2 * math.pi * R
    if ideal <= 0:
        return 0.0

    r = length_m / ideal
    return max(0.2, min(0.99, r))


# -----------------------------------------
# 최적화된 Area Loop (빠른 버전)
# -----------------------------------------
def generate_area_loop(lat: float, lng: float, km: float):
    target_m = km * 1000.0
    MAX_ERR = 99.0

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # **가장 핵심적인 속도 개선 (탐색 범위 축소)**
    MAX_OUTER = 4
    MAX_INNER = 3
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    base_r = max(280.0, min(420.0, target_m / 7.0))

    best_any_score = float("inf")
    best_any_route = []
    best_any_meta = {}

    for outer in range(MAX_OUTER):
        R = base_r + outer * 22.0

        for inner in range(MAX_INNER):
            # via_cnt 고정 → API 호출 수 대폭 감소
            via_cnt = 3
            angle_step = 360.0 / via_cnt

            rad_points = []
            for i in range(via_cnt):
                ang = math.radians(i * angle_step)
                rad_lat = lat + (R / 111111.0) * math.cos(ang)
                rad_lng = lng + (R / (111111.0 * math.cos(math.radians(lat)))) * math.sin(ang)
                rad_points.append((rad_lat, rad_lng))

            full = [(lat, lng)]
            fail_cnt = 0

            for i in range(via_cnt):
                seg = valhalla_route(full[-1], rad_points[i])
                if not seg:
                    fail_cnt += 1
                    continue
                full.extend(seg[1:])

            back_seg = valhalla_route(full[-1], (lat, lng))
            if back_seg:
                full.extend(back_seg[1:])
            else:
                fail_cnt += 1

            if len(full) < 4:
                continue

            length_m = 0.0
            for i in range(1, len(full)):
                length_m += haversine_m(full[i - 1][0], full[i - 1][1],
                                        full[i][0], full[i][1])

            err = abs(length_m - target_m)

            if not (0.6 * target_m <= length_m <= 1.6 * target_m):
                continue

            roundness = _estimate_roundness(length_m, R)
            score = compute_loop_score(length_m, target_m, fail_cnt, roundness)

            # 성공 → 즉시 종료
            if err <= MAX_ERR:
                return full, {
                    "outer": outer, "inner": inner, "via": via_cnt,
                    "len": length_m, "err": err,
                    "round": roundness, "fail": fail_cnt,
                    "success": True, "used_fallback": False,
                }

            # Fallback 후보 갱신
            if score < best_any_score:
                best_any_score = score
                best_any_route = full[:]
                best_any_meta = {
                    "outer": outer, "inner": inner, "via": via_cnt,
                    "len": length_m, "err": err,
                    "round": roundness, "fail": fail_cnt,
                    "score": score,
                }

        # 빠르게 종료 (조건 강화)
        if best_any_score < 200:
            break

    if best_any_route:
        best_any_meta["success"] = False
        best_any_meta["used_fallback"] = True
        return best_any_route, best_any_meta

    return [(lat, lng)], {
        "len": 0.0, "err": target_m,
        "success": False, "used_fallback": False,
        "message": "Valhalla 경로 생성 실패",
    }
