"""
route_algo.py

- Valhalla 기반 도보(러닝) 전용 루프 경로 생성기
- Area-Loop (옵션 A1, 전국 공통 안정형 둥근 루프) 알고리즘 사용
- 목표 거리: km 단위 (예: 2.0km). 실제 경로 길이는 목표 ±99m 이내를 우선적으로 선택.
"""

from __future__ import annotations

import os
import math
import random
import logging
from typing import List, Tuple, Dict, Optional

import requests


logger = logging.getLogger(__name__)

# Valhalla 엔드포인트 (환경변수로 덮어쓰기 가능)
VALHALLA_URL = os.getenv("VALHALLA_URL", "http://localhost:8002/route")

# 거리 관련 상수
TARGET_ERR_M = 99.0        # 목표 거리와의 허용 오차 (러닝앱에서 0.1km 단위 표시용)
MAX_MULTIPLIER_ERR = 2.0   # target * 이 값보다 길이가 크면 후보에서 제외

# Area-Loop 탐색 설정
OUTER_LOOPS = 10           # 반경 스케일링 반복 횟수
VIA_COUNTS = (3, 4, 5)     # 삼각형 / 사각형 / 오각형 루프
RADIUS_JITTER = 0.25       # 반경에 주는 랜덤 편차 비율 (±25%)


# ------------------------------------------------------------
# 기본 지오메트리 / 유틸 함수
# ------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도간 거리 (미터)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _decode_polyline6(polyline: str) -> List[Dict[str, float]]:
    """
    Valhalla/OSRM 스타일 6-precision polyline 디코더.
    결과: [{lat: float, lng: float}, ...]
    """
    coords: List[Dict[str, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(polyline)

    while index < length:
        result = 1
        shift = 0
        while True:
            if index >= length:
                break
            b = ord(polyline[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        result = 1
        shift = 0
        while True:
            if index >= length:
                break
            b = ord(polyline[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append(
            {"lat": lat / 1e6, "lng": lng / 1e6}
        )

    return coords


def _route_segment(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
) -> Tuple[List[Dict[str, float]], float]:
    """
    Valhalla에 도보 경로를 요청하여
    - polyline (list[ {lat,lng} ])
    - length_m (float)
    를 반환한다.
    실패 시 ([], 0.0)
    """
    body = {
        "locations": [
            {"lat": start_lat, "lon": start_lng},
            {"lat": end_lat, "lon": end_lng},
        ],
        "costing": "pedestrian",
        "costing_options": {
            "pedestrian": {
                "walking_speed": 5.0,   # km/h
                "use_sidewalk": 1.0,
            }
        },
        "directions_options": {"units": "kilometers"},
    }

    try:
        resp = requests.post(VALHALLA_URL, json=body, timeout=8)
        resp.raise_for_status()
        js = resp.json()
    except Exception as e:
        logger.warning("[AreaLoop] Valhalla 요청 실패: %s", e)
        return [], 0.0

    try:
        leg = js["trip"]["legs"][0]
        shape = leg["shape"]
        length_km = float(leg["summary"]["length"])
    except Exception as e:
        logger.warning("[AreaLoop] Valhalla 응답 파싱 실패: %s", e)
        return [], 0.0

    coords = _decode_polyline6(shape)
    length_m = length_km * 1000.0
    return coords, length_m


def dest_point(
    lat: float,
    lng: float,
    bearing_rad: float,
    distance_m: float,
) -> Tuple[float, float]:
    """시작점에서 bearing, 거리로 이동한 지점의 위경도."""
    R = 6371000.0
    d = distance_m / R
    lat1 = math.radians(lat)
    lon1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def roundness(center_lat: float, center_lng: float, polyline: List[Dict[str, float]]) -> float:
    """
    루프의 '둥근 정도'(0~1).
    - 중심점에서 각 좌표까지의 거리 분산이 작을수록 1에 가까워짐.
    """
    if len(polyline) < 4:
        return 0.0

    dists = [
        haversine_m(center_lat, center_lng, p["lat"], p["lng"])
        for p in polyline
    ]
    mean = sum(dists) / len(dists)
    if mean <= 1e-6:
        return 0.0

    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    sigma = math.sqrt(var)
    sigma_ratio = sigma / mean  # 작을수록 좋음

    # sigma_ratio 0 -> 1.0, 1 이상이면 0 근처로
    r = max(0.0, 1.0 - min(1.0, sigma_ratio))
    return r


# ------------------------------------------------------------
# Area-Loop A1: 전국 공통 안정형 둥근 루프
# ------------------------------------------------------------

def _candidate_vias(
    start_lat: float,
    start_lng: float,
    base_radius_m: float,
    via_count: int,
) -> List[Tuple[float, float]]:
    """
    중심에서 base_radius 주변에 via_count 개의 점을 균등하게 배치하고
    약간의 랜덤 편차를 준다.
    """
    vias: List[Tuple[float, float]] = []
    # 시작 각도도 랜덤으로 회전
    base_theta = random.random() * 2 * math.pi

    for i in range(via_count):
        theta = base_theta + 2 * math.pi * (i / via_count)
        # 반경에 ±RADIUS_JITTER 범위로 랜덤 편차
        radius = base_radius_m * (1.0 + RADIUS_JITTER * (random.random() * 2 - 1))
        vlat, vlng = dest_point(start_lat, start_lng, theta, radius)
        vias.append((vlat, vlng))

    return vias


def _build_loop_from_vias(
    start_lat: float,
    start_lng: float,
    vias: List[Tuple[float, float]],
) -> Tuple[List[Dict[str, float]], float]:
    """
    start → via1 → via2 → ... → viaN → start 로 도보 경로를 붙여서 하나의 polyline 생성.
    """
    poly: List[Dict[str, float]] = []
    total_len = 0.0

    prev_lat, prev_lng = start_lat, start_lng
    for (vlat, vlng) in vias + [(start_lat, start_lng)]:
        seg_coords, seg_len = _route_segment(prev_lat, prev_lng, vlat, vlng)
        if not seg_coords or seg_len <= 1.0:
            return [], 0.0

        if not poly:
            poly.extend(seg_coords)
        else:
            # 중복되는 시작점 하나 제거
            poly.extend(seg_coords[1:])

        total_len += seg_len
        prev_lat, prev_lng = vlat, vlng

    return poly, total_len


def generate_loop_route(
    start_lat: float,
    start_lng: float,
    km: float,
) -> Tuple[List[Dict[str, float]], float, str]:
    """
    Area-Loop A1 알고리즘으로 러닝 루프를 생성한다.

    반환:
        polyline: list[{lat,lng}]
        total_length_m: 실제 경로 길이 (미터)
        meta: 알고리즘/파라미터 정보 문자열
    """
    target_m = km * 1000.0
    center_lat, center_lng = start_lat, start_lng

    # "지하철 한 정거장 정도" 감각에 맞는 초기 반경 추정
    # 대략 2πR ≈ target 이므로 R ≈ target / (2π) 를 기본으로,
    # 너무 작거나 크지 않게 클램프
    base_r = max(180.0, min(target_m / (2 * math.pi), 900.0))

    logger.info("[AreaLoop] target=%.1fm, base_r=%.1fm", target_m, base_r)

    best_ok: Optional[Tuple[List[Dict[str, float]], float, float, int, float]] = None
    best_any: Optional[Tuple[List[Dict[str, float]], float, float, int, float]] = None
    # (poly, total_len, err, via_count, round_score)

    for outer in range(OUTER_LOOPS):
        # 바깥 루프마다 반경을 약간씩 키우거나 줄이면서 탐색
        radius = base_r * (0.8 + 0.05 * outer)

        for via_count in VIA_COUNTS:
            vias = _candidate_vias(center_lat, center_lng, radius, via_count)
            poly, total_len = _build_loop_from_vias(center_lat, center_lng, vias)
            if not poly or total_len <= 0.0:
                logger.info(
                    "[AreaLoop] outer=%d, via=%d: Valhalla 세그먼트 실패",
                    outer,
                    via_count,
                )
                continue

            err = abs(total_len - target_m)

            # 너무 과도하게 길거나 짧은 건 버리기
            if total_len > target_m * MAX_MULTIPLIER_ERR:
                logger.info(
                    "[AreaLoop] outer=%d, via=%d: len=%.1fm, err=%.1fm (과도한 길이로 스킵)",
                    outer,
                    via_count,
                    total_len,
                    err,
                )
                continue

            round_score = roundness(center_lat, center_lng, poly)

            logger.info(
                "[AreaLoop] outer=%d, via=%d, r=%.0fm, len=%.1fm, err=%.1fm, round=%.2f",
                outer,
                via_count,
                radius,
                total_len,
                err,
                round_score,
            )

            cand = (poly, total_len, err, via_count, round_score)
            # err 기준으로 우선 후보 갱신
            if err <= TARGET_ERR_M:
                # 허용 오차 이내 루프
                if (
                    best_ok is None
                    or err < best_ok[2]
                    or (math.isclose(err, best_ok[2]) and round_score > best_ok[4])
                ):
                    best_ok = cand
            else:
                # 아직 허용 오차 이내 루프가 없다면, 전체 중 최선도 기억
                if best_any is None or err < best_any[2]:
                    best_any = cand

    if best_ok is not None:
        poly, total_len, err, via_count, round_score = best_ok
        meta = f"AreaLoop-A1 via={via_count} r≈{int(base_r)}m ok_err={int(err)}m round={round_score:.2f}"
        logger.info(
            "[AreaLoop] 선택된 루프(OK): len=%.1fm, target=%.1fm, err=%.1fm, via=%d, round=%.2f",
            total_len,
            target_m,
            err,
            via_count,
            round_score,
        )
        return poly, total_len, meta

    if best_any is not None:
        poly, total_len, err, via_count, round_score = best_any
        meta = f"AreaLoop-A1 via={via_count} r≈{int(base_r)}m best_err={int(err)}m round={round_score:.2f}"
        logger.warning(
            "[AreaLoop] MAX_ERR 내 루프를 찾지 못해 최선 루프 반환: len=%.1fm, target=%.1fm, err=%.1fm, via=%d, round=%.2f",
            total_len,
            target_m,
            err,
            via_count,
            round_score,
        )
        return poly, total_len, meta

    # Valhalla 호출이 모두 실패한 경우: 시작점만 반환
    logger.error("[AreaLoop] 어떤 루프도 생성하지 못했습니다. 빈 루프 반환.")
    return [{"lat": start_lat, "lng": start_lng}], 0.0, "AreaLoop-A1 FAILED"
