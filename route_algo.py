import math
import os
import time
import logging
from typing import List, Dict, Tuple, Optional

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 기본 설정
# -----------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
# 개별 요청 타임아웃 (기존 10초 → 3초로 단축)
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "3"))
# 재시도 횟수 (보통 로컬 Valhalla 는 빠르므로 1~2 회면 충분)
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "2"))

# 러닝 속도(분당 km) – 요약 정보에만 사용
RUNNING_SPEED_KMH = 8.0  # 8km/h 기준


# -----------------------------
# 거리 / 기하 유틸
# -----------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 대략적인 거리(m)."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polyline_length_m(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        total += haversine_m(lat1, lon1, lat2, lon2)
    return total


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """p1 → p2 방위각 (deg, 0=북, 시계방향)."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def project_point(
    lat: float,
    lon: float,
    distance_m: float,
    bearing_deg_: float,
) -> Tuple[float, float]:
    """시작점에서 특정 거리/방위각만큼 이동한 위경도 (단순 구면 좌표)."""
    R = 6371000.0
    br = math.radians(bearing_deg_)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(distance_m / R)
        + math.cos(phi1) * math.sin(distance_m / R) * math.cos(br)
    )
    lam2 = lam1 + math.atan2(
        math.sin(br) * math.sin(distance_m / R) * math.cos(phi1),
        math.cos(distance_m / R) - math.sin(phi1) * math.sin(phi2),
    )

    return (math.degrees(phi2), (math.degrees(lam2) + 540.0) % 360.0 - 180.0)


# -----------------------------
# Valhalla polyline 디코딩
# -----------------------------

def _decode_polyline(shape: str) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    lat = 0
    lng = 0
    idx = 0

    while idx < len(shape):
        b = 0x20
        shift = 0
        result = 0
        while b >= 0x20:
            b = ord(shape[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        b = 0x20
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


# -----------------------------
# Valhalla API 호출 (도보 전용)
# -----------------------------

def valhalla_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Valhalla로 도보 경로를 요청하고 polyline 좌표 리스트를 반환.

    실패 시 지정된 횟수만큼 재시도 후 빈 리스트 반환.
    """
    lat1, lon1 = p1
    lat2, lon2 = p2

    last_error: Optional[Exception] = None

    for attempt in range(VALHALLA_MAX_RETRY):
        try:
            payload = {
                "locations": [
                    {"lat": lat1, "lon": lon1, "type": "break"},
                    {"lat": lat2, "lon": lon2, "type": "break"},
                ],
                "costing": "pedestrian",
            }
            resp = requests.post(
                VALHALLA_URL,
                json=payload,
                timeout=VALHALLA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            shape = data["trip"]["legs"][0]["shape"]
            coords = _decode_polyline(shape)
            if len(coords) < 2:
                raise ValueError("decoded polyline too short")
            return coords
        except Exception as e:
            last_error = e
            logger.warning(
                "[Valhalla] attempt %d failed for %s -> %s: %s",
                attempt + 1,
                p1,
                p2,
                e,
            )

    logger.error("[Valhalla] all attempts failed for %s -> %s: %s", p1, p2, last_error)
    return []


# -----------------------------
# 루프 품질 평가
# -----------------------------

def _loop_roundness(points: List[Tuple[float, float]]) -> float:
    """루프의 '원형도'를 0~1 사이로 대략 계산.

    - 전체 궤적의 centroid 를 중심으로
    - 각 점의 중심까지 거리의 평균과 분산을 기반으로 함
    """
    if len(points) < 4:
        return 0.0

    xs = [p[1] for p in points]
    ys = [p[0] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    dists = [haversine_m(cy, cx, lat, lon) for lat, lon in points]
    if not dists:
        return 0.0

    mean_r = sum(dists) / len(dists)
    if mean_r <= 0:
        return 0.0

    var = sum((d - mean_r) ** 2 for d in dists) / len(dists)
    # 분산이 작을수록(동그랗게 분포) roundness ↑
    # 경험적으로 0~1 사이에 오도록 스케일링
    score = 1.0 / (1.0 + var / (mean_r * mean_r + 1e-6))
    return max(0.0, min(1.0, score))


def _score_loop(
    points: List[Tuple[float, float]],
    target_m: float,
) -> Tuple[float, Dict]:
    length_m = polyline_length_m(points)
    if length_m <= 0.0:
        return float("inf"), {
            "len": 0.0,
            "err": target_m,
            "roundness": 0.0,
        }

    err = abs(length_m - target_m)
    roundness = _loop_roundness(points)

    # 오차 1m당 1점, roundness(0~1) 보정(0.0~0.3 * target_m)
    score = err + (1.0 - roundness) * 0.3 * target_m

    meta = {
        "len": length_m,
        "err": err,
        "roundness": roundness,
        "score": score,
    }
    return score, meta


# -----------------------------
# Area Loop 생성 (경량/안정 버전)
# -----------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
):
    """목표 거리(km) 근처의 '짧은 러닝 루프'를 생성한다.

    기존 버전에서는 outer/inner 중첩 루프 + 다수의 Valhalla 호출 때문에
    최악의 경우 수십 초~수 분까지 걸릴 수 있었다.

    여기서는:
      - 후보 방향을 소수(4~8개)만 사용
      - 각 후보는 '왕복 경로' 2회 호출로 제한
      - 전체 Valhalla 호출 수를 최대 16회 이내로 유지

    반환:
      (polyline(List[(lat,lng)]), meta(dict))
    """
    target_m = max(300.0, km * 1000.0)  # 최소 300m 정도는 확보
    km_requested = km

    # 원둘레 = 2πR  →  R ≈ L / (2π)
    ideal_R = target_m / (2.0 * math.pi)
    # 도심부/공원 등을 고려해 너무 작거나 크지 않게 클램프
    R = max(180.0, min(600.0, ideal_R))

    # 후보 방위각 (8방위)
    bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    best_route: List[Tuple[float, float]] = []
    best_meta: Dict = {}
    best_score = float("inf")

    valhalla_calls = 0
    MAX_TOTAL_CALLS = 16

    start = (lat, lng)

    for br in bearings:
        if valhalla_calls >= MAX_TOTAL_CALLS:
            break

        via = project_point(lat, lng, R, br)

        # 1) 출발 → via
        out_seg = valhalla_route(start, via)
        valhalla_calls += 1

        if not out_seg or len(out_seg) < 2:
            continue

        if valhalla_calls >= MAX_TOTAL_CALLS:
            break

        # 2) via → 출발
        back_seg = valhalla_route(out_seg[-1], start)
        valhalla_calls += 1

        if not back_seg or len(back_seg) < 2:
            continue

        # 왕복 루프 polyline 구성 (접점 중복 제거)
        loop_pts: List[Tuple[float, float]] = out_seg + back_seg[1:]

        score, local_meta = _score_loop(loop_pts, target_m)
        if score < best_score:
            best_score = score
            best_route = loop_pts
            best_meta = local_meta

    # -----------------------------
    # 결과 정리
    # -----------------------------
    if best_route:
        length_m = best_meta.get("len", polyline_length_m(best_route))
        err = best_meta.get("err", abs(length_m - target_m))

        best_meta.update(
            {
                "success": True,
                "used_fallback": False,
                "km_requested": km_requested,
                "target_m": target_m,
                "valhalla_calls": valhalla_calls,
            }
        )
        return best_route, best_meta

    # -----------------------------
    # 완전 실패 시: 가장 단순한 out-and-back 시도
    # -----------------------------
    # R 을 조금 줄여서 한 번만 시도
    simple_via = project_point(lat, lng, R * 0.6, 0.0)
    out_seg = valhalla_route(start, simple_via)
    if out_seg and len(out_seg) >= 2:
        back_seg = list(reversed(out_seg))
        loop_pts = out_seg + back_seg[1:]
        _, meta = _score_loop(loop_pts, target_m)
        meta.update(
            {
                "success": False,
                "used_fallback": True,
                "km_requested": km_requested,
                "target_m": target_m,
                "valhalla_calls": valhalla_calls,
                "message": "안전한 루프를 찾지 못해 단순 왕복 경로를 사용했습니다.",
            }
        )
        return loop_pts, meta

    # Valhalla 자체가 완전히 실패한 경우
    return [start], {
        "len": 0.0,
        "err": target_m,
        "success": False,
        "used_fallback": False,
        "km_requested": km_requested,
        "target_m": target_m,
        "valhalla_calls": valhalla_calls,
        "message": "Valhalla 경로 생성 실패",
    }
