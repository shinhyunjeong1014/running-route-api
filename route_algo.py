import math
import os
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
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -----------------------------
# Valhalla API 호출 (도보 전용)
# -----------------------------
VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")


def valhalla_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    Valhalla로 '도보 전용' 경로 요청.
    """
    body = {
        "locations": [
            {"lat": p1[0], "lon": p1[1]},
            {"lat": p2[0], "lon": p2[1]},
        ],
        "costing": "pedestrian",
        "directions_options": {"units": "meters"},
    }

    try:
        r = requests.post(VALHALLA_URL, json=body, timeout=10)
        r.raise_for_status()
        data = r.json()
        shape = data["trip"]["legs"][0]["shape"]

        # polyline decode
        coords: List[Tuple[float, float]] = []
        lat = 0
        lng = 0
        idx = 0

        while idx < len(shape):
            b = 1
            shift = 0
            result = 0
            # latitude
            while b >= 0x20:
                b = ord(shape[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            # longitude
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
# 루프 점수 계산 (낮을수록 좋음)
# -----------------------------
def compute_loop_score(length_m, target_m, uturn_cnt, roundness):
    err = abs(length_m - target_m)
    return (
        err +            # 거리 오차
        uturn_cnt * 300  # U턴 패널티
        + (1 - roundness) * 200  # 원형도(반복 없이 둥근지) 패널티
    )


# -----------------------------
# Area-Loop (옵션 A)
#  - 도보 전용 Valhalla 경로 사용
#  - 목표 거리 ±99m 이내 우선 탐색
#  - 실패 시 점수 기준 최선 루프 폴백
# -----------------------------
def generate_area_loop(lat: float, lng: float, km: float):
    target_m = km * 1000
    MAX_ERR = 99
    MAX_OUTER = 12
    MAX_INNER = 10

    # 기본 반경 설정 (대략적인 1바퀴 둘레 기준)
    base_r = max(280, min(420, target_m / 7))
    center = (lat, lng)

    best_score = float("inf")
    best_route: List[Tuple[float, float]] = []
    best_meta: Dict = {}

    # MAX_ERR 밖이더라도 가장 점수가 좋은 후보 (폴백용)
    best_any_score = float("inf")
    best_any_route: List[Tuple[float, float]] = []
    best_any_meta: Dict = {}

    for outer in range(MAX_OUTER):
        R = base_r + outer * 22  # 점점 큰 원으로
        for inner in range(MAX_INNER):
            via_cnt = 3 + (inner % 3)  # 3~5개 방사형 포인트

            angle_step = 360 / via_cnt
            rad_points: List[Tuple[float, float]] = []
            for i in range(via_cnt):
                ang = math.radians(i * angle_step)

                rad_lat = lat + (R / 111111) * math.cos(ang)
                rad_lng = lng + (R / (111111 * math.cos(math.radians(lat)))) * math.sin(ang)
                rad_points.append((rad_lat, rad_lng))

            full: List[Tuple[float, float]] = [(lat, lng)]
            uturn_cnt = 0

            # 중심 → 방사형 포인트들 순서대로
            for i in range(via_cnt):
                seg = valhalla_route(full[-1], rad_points[i])
                if not seg:
                    # 한 번 더 재시도
                    seg2 = valhalla_route(full[-1], rad_points[i])
                    if not seg2:
                        uturn_cnt += 1
                        continue
                    seg = seg2

                full.extend(seg[1:])

            # 마지막: 다시 출발점으로 복귀
            back_seg = valhalla_route(full[-1], (lat, lng))
            if back_seg:
                full.extend(back_seg[1:])
            else:
                uturn_cnt += 1

            if len(full) < 4:
                continue

            # 전체 길이 계산
            length_m = 0.0
            for i in range(1, len(full)):
                length_m += haversine_m(
                    full[i - 1][0], full[i - 1][1],
                    full[i][0], full[i][1],
                )

            err = abs(length_m - target_m)
            # 원형도: 실제 둘레 / 이론적 원둘레 (1에 가까울수록 좋음)
            roundness = min(0.99, length_m / (2 * math.pi * R))
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

            # 1) 목표거리 ±99m 이내 후보 중에서 점수 최소
            if err <= MAX_ERR and score < best_score:
                best_score = score
                best_route = full[:]
                best_meta = meta
                logger.info(f"[AreaLoop] MATCH outer={outer}, inner={inner}, {meta}")
            else:
                # 단순 디버그용 로그
                logger.info(
                    "[AreaLoop] outer=%d, inner=%d, len=%.1fm, err=%.1fm, br=%.2f, "
                    "uturn=%d, round=%.2f, score=%.1f",
                    outer, inner, length_m, err, roundness, uturn_cnt, roundness, score
                )

            # 2) 에러 상관없이 전체 중 최선 후보 (폴백용)
            if score < best_any_score:
                best_any_score = score
                best_any_route = full[:]
                best_any_meta = meta

    # 우선: MAX_ERR 이내 최선 루프
    if best_route:
        return best_route, best_meta

    # 없으면: 전체 중 최선 루프 폴백
    if best_any_route:
        logger.warning(
            "[AreaLoop] MAX_ERR 내 루프를 찾지 못해 최선 루프 반환: "
            "len=%.1fm, target=%.1fm, err=%.1fm, score=%.1f",
            best_any_meta.get("len", 0.0),
            target_m,
            best_any_meta.get("err", 0.0),
            best_any_meta.get("score", 0.0),
        )
        return best_any_route, best_any_meta

    # Valhalla가 전부 실패한 극단적인 경우
    logger.error("[AreaLoop] Valhalla 실패로 유효한 루프를 만들지 못했습니다.")
    return [(lat, lng)], {"len": 0.0, "err": target_m}
