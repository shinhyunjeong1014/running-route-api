# route_algo.py — Valhalla 기반 삼각 러닝 루프 (옵션 C)

import math
import random
import logging
import os
from typing import List, Dict, Tuple

import requests

# ------------------------------
# 기본 설정
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 목표 거리 허용 오차 (m) – 예: 2km 요청 시 1.75~2.25km 정도 허용
TARGET_RANGE_M = 250.0

# Valhalla 라우팅 엔드포인트
# 따로 설정 안 하면 localhost:8002 사용
VALHALLA_URL = os.getenv("VALHALLA_URL", "http://localhost:8002/route")


# ------------------------------
# 유틸 함수들 (turn_algo에서도 사용)
# ------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 거리를 미터(m) 단위로 계산 (Haversine)."""
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _cumulative_distances(polyline: List[Dict[str, float]]) -> List[float]:
    """Polyline 각 포인트까지의 누적 거리 리스트를 계산."""
    dists = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        dists.append(dists[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return dists


# ------------------------------
# 지구 위에서 특정 각도/거리만큼 이동
# ------------------------------
def _move_point(lat: float, lng: float, bearing_deg: float, distance_m: float) -> Dict[str, float]:
    """
    시작점(lat, lng)에서 방위각(bearing_deg) 방향으로 distance_m 만큼 이동한 점을 구한다.
    (단순 구면삼각법; 러닝 코스 생성에는 충분한 정밀도)
    """
    R = 6371000.0
    bearing = math.radians(bearing_deg)

    lat1 = math.radians(lat)
    lon1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2),
    )

    return {
        "lat": math.degrees(lat2),
        "lng": (math.degrees(lon2) + 540) % 360 - 180,  # [-180, 180] 범위 보정
    }


# ------------------------------
# polyline6 디코더 (Valhalla shape)
# ------------------------------
def _decode_polyline6(encoded: str) -> List[Dict[str, float]]:
    """Valhalla에서 사용하는 polyline6 문자열을 (lat, lng) 리스트로 변환."""
    coords: List[Dict[str, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)

    while index < length:
        # latitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        # longitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append({"lat": lat / 1e6, "lng": lng / 1e6})

    return coords


# ------------------------------
# Valhalla 한 구간 라우팅
# ------------------------------
def _route_segment(start: Dict[str, float], end: Dict[str, float]) -> Tuple[List[Dict[str, float]], float]:
    """
    Valhalla에 start -> end 경로를 요청하고,
    (polyline 좌표 리스트, 경로 길이[m])를 반환.
    """
    payload = {
        "locations": [
            {"lat": start["lat"], "lon": start["lng"]},
            {"lat": end["lat"], "lon": end["lng"]},
        ],
        "costing": "pedestrian",
        "directions_options": {
            "units": "kilometers",
            "narrative": False,
        },
    }

    try:
        resp = requests.post(VALHALLA_URL, json=payload, timeout=5)
    except Exception as e:
        raise RuntimeError(f"Valhalla 접속 오류: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Valhalla 오류: status={resp.status_code}, body={resp.text[:200]}")

    data = resp.json()
    trip = data.get("trip")
    if not trip or "legs" not in trip:
        raise RuntimeError(f"Valhalla 응답 형식 오류: {data}")

    coords: List[Dict[str, float]] = []
    total_len_m = 0.0

    for leg in trip["legs"]:
        shape = leg.get("shape")
        if not shape:
            continue
        leg_coords = _decode_polyline6(shape)
        if not leg_coords:
            continue

        if not coords:
            coords.extend(leg_coords)
        else:
            coords.extend(leg_coords[1:])  # 앞 점 중복 제거

        summary = leg.get("summary", {})
        leg_len_km = float(summary.get("length", 0.0))
        total_len_m += leg_len_km * 1000.0

    if len(coords) < 2 or total_len_m <= 0:
        raise RuntimeError("Valhalla 경로가 비어 있거나 길이 0")

    return coords, total_len_m


# ------------------------------
# 네트워크 친화적인 삼각 러닝 루프 생성 (옵션 C)
# ------------------------------
def _build_triangle_loop_network_aware(
    lat: float,
    lng: float,
    km: float,
    max_outer_attempts: int = 8,
    inner_attempts: int = 5,
) -> Tuple[List[Dict[str, float]], float, str]:
    """
    Valhalla를 사용해 '시작점-포인트A-포인트B-시작점' 형태의 삼각 러닝 루프를 생성한다.
    - 거리: target_m ± TARGET_RANGE_M
    - C 방식: 네트워크 상황에 따라 반지름/방위각을 적응적으로 조정
    """
    start = {"lat": lat, "lng": lng}
    target_m = km * 1000.0

    # 초기 반지름 추정 (도시/주거지 기준 러닝 코스 느낌)
    base_radius = max(300.0, min(target_m * 0.6, target_m / (3.0 * 1.3)))

    best_route = None  # (coords, length_m, algo_tag)
    best_err = float("inf")

    for outer in range(max_outer_attempts):
        # 방위각 기본값 + 약간의 랜덤 편차
        base_bearing = random.uniform(0.0, 360.0)

        # 네트워크 밀도/실패 여부에 따라 radius를 늘렸다 줄였다
        radius = base_radius * (0.8 + 0.4 * random.random())

        for inner in range(inner_attempts):
            # 정삼각형을 기본으로 하되, 각도를 약간 랜덤하게 움직여 "왜곡된 삼각형" 허용
            jitter = random.uniform(-20.0, 20.0)
            bearing_a = base_bearing + jitter
            bearing_b = base_bearing + 120.0 + jitter

            A = _move_point(lat, lng, bearing_a, radius)
            B = _move_point(lat, lng, bearing_b, radius)

            try:
                seg1, len1 = _route_segment(start, A)
                seg2, len2 = _route_segment(A, B)
                seg3, len3 = _route_segment(B, start)
            except Exception as e:
                logger.info(f"[TriLoop] Valhalla 세그먼트 실패 outer={outer}, inner={inner}: {e}")
                # 이 bearing/radius 조합은 망했으니, inner 루프를 종료하고 다음 outer로
                break

            loop_coords: List[Dict[str, float]] = []
            loop_coords.extend(seg1)
            loop_coords.extend(seg2[1:])
            loop_coords.extend(seg3[1:])

            total_len = len1 + len2 + len3
            err = abs(total_len - target_m)

            logger.info(
                f"[TriLoop] outer={outer}, inner={inner}, "
                f"radius={radius:.0f}m, length={total_len:.1f}m, "
                f"target={target_m:.1f}m, err={err:.1f}m"
            )

            # 가장 좋은 후보 저장
            if err < best_err:
                best_err = err
                best_route = (loop_coords, total_len, f"VH_Triangle_C r={radius:.0f}m")

            # 허용 오차 이내면 바로 반환
            if err <= TARGET_RANGE_M:
                return loop_coords, total_len, f"VH_Triangle_C_OK r={radius:.0f}m"

            # 길이에 따라 radius를 조금 보정 (네트워크 따라 적응)
            if total_len < target_m - TARGET_RANGE_M:
                radius *= 1.15
            elif total_len > target_m + TARGET_RANGE_M:
                radius *= 0.85
            else:
                # TARGET_RANGE_M보다 크지만, 더 줄이기 애매하면 inner 종료
                break

    if best_route:
        # 최선의 후보라도 반환 (UX 관점에서 "대충이라도 루프" 제공)
        logger.warning(
            f"[TriLoop] 목표 거리에 정확히 못 맞췄지만 최선의 루프 반환: "
            f"len={best_route[1]:.1f}m, target={target_m:.1f}m, err={abs(best_route[1]-target_m):.1f}m"
        )
        return best_route

    raise RuntimeError("Valhalla 기반 삼각 루프를 생성하지 못했습니다.")


# ------------------------------
# 외부 인터페이스
# ------------------------------
def generate_route(lat: float, lng: float, km: float):
    """
    FastAPI(app.py)에서 호출하는 외부 인터페이스.
    - 반환값: (polyline_coords, length_m, algorithm_used)
    """
    coords, length_m, algo_tag = _build_triangle_loop_network_aware(lat, lng, km)
    return coords, length_m, algo_tag


# 과거 코드 호환용 (혹시 generate_loop_route를 임포트하는 버전이 있을 경우)
def generate_loop_route(lat: float, lng: float, km: float):
    return generate_route(lat, lng, km)
