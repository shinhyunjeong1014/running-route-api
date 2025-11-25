# route_algo.py
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

# 목표 거리 허용 오차 (예: 2km 요청 시 1.8~2.2km)
TARGET_RANGE_M = 200.0

# Valhalla 라우팅 엔드포인트
VALHALLA_URL = os.getenv("VALHALLA_URL", "http://localhost:8002/route")


# ------------------------------
# 유틸 함수들 (turn_algo에서도 사용)
# ------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 거리를 미터 단위로 계산 (Haversine)."""
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

    resp = requests.post(VALHALLA_URL, json=payload, timeout=4)
    if resp.status_code != 200:
        raise RuntimeError(f"Valhalla 오류: status={resp.status_code}, body={resp.text[:200]}")

    data = resp.json()
    if "trip" not in data or "legs" not in data["trip"]:
        raise RuntimeError(f"Valhalla 응답 형식 오류: {data}")

    coords: List[Dict[str, float]] = []
    total_len_m = 0.0

    for leg in data["trip"]["legs"]:
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

        # summary.length 는 km 단위
        summary = leg.get("summary", {})
        leg_len_km = float(summary.get("length", 0.0))
        total_len_m += leg_len_km * 1000.0

    if len(coords) < 2 or total_len_m <= 0:
        raise RuntimeError("Valhalla 경로가 비어 있거나 길이 0")

    return coords, total_len_m


# ------------------------------
# 삼각 러닝 루프 생성 (Valhalla 기반)
# ------------------------------
def _build_triangle_loop(
    lat: float,
    lng: float,
    km: float,
    max_attempts: int = 6,
    inner_attempts: int = 5,
) -> Tuple[List[Dict[str, float]], float, str]:
    """
    Valhalla를 사용해 '시작점-포인트A-포인트B-시작점' 형태의 삼각 루프를 생성한다.
    - 거리: target_m ± TARGET_RANGE_M 내에 들어오도록 radius를 조정.
    """
    start = {"lat": lat, "lng": lng}
    target_m = km * 1000.0

    # 네트워크 굴곡을 고려한 초기 반지름 추정값 (거리의 약 1/3 / 1.3)
    base_radius = max(250.0, min(target_m * 0.6, target_m / (3.0 * 1.3)))

    best_route = None  # (coords, length_m, tag)
    best_err = float("inf")

    for outer in range(max_attempts):
        base_bearing = random.uniform(0.0, 360.0)
        radius = base_radius

        for inner in range(inner_attempts):
            # 정삼각형 형태의 두 꼭짓점(A, B)을 생성 (시작점이 세 번째 꼭짓점 역할)
            A = _move_point(lat, lng, base_bearing, radius)
            B = _move_point(lat, lng, base_bearing + 120.0, radius)

            try:
                seg1, len1 = _route_segment(start, A)
                seg2, len2 = _route_segment(A, B)
                seg3, len3 = _route_segment(B, start)
            except Exception as e:
                logging.warning(f"Valhalla 세그먼트 라우팅 실패 (outer={outer}, inner={inner}): {e}")
                break  # 바깥 bearing을 바꿔 다시 시도

            # 세그먼트 연결
            loop_coords: List[Dict[str, float]] = []
            loop_coords.extend(seg1)
            loop_coords.extend(seg2[1:])
            loop_coords.extend(seg3[1:])

            total_len = len1 + len2 + len3
            err = abs(total_len - target_m)

            logging.info(
                f"삼각 루프 시도 outer={outer}, inner={inner}, radius={radius:.0f}m, "
                f"length={total_len:.1f}m, target={target_m:.1f}m, err={err:.1f}m"
            )

            # 가장 좋은 후보 저장
            if err < best_err:
                best_err = err
                best_route = (loop_coords, total_len, f"VH_Triangle r={radius:.0f}m")

            # 허용오차 이내면 바로 반환
            if err <= TARGET_RANGE_M:
                return loop_coords, total_len, f"VH_Triangle_OK r={radius:.0f}m"

            # 길이가 부족하면 반지름 확대, 너무 길면 축소
            if total_len < target_m - TARGET_RANGE_M:
                radius *= 1.15
            elif total_len > target_m + TARGET_RANGE_M:
                radius *= 0.85
            else:
                # TARGET_RANGE_M보다는 크지만, 조정 여지가 적은 경우
                break

    if best_route:
        # 최선의 후보라도 반환 (러닝 앱 UX 관점에서 "대충이라도 루프" 제공)
        return best_route

    raise RuntimeError("Valhalla 기반 삼각 루프 생성 실패")


# ------------------------------
# 외부 인터페이스
# ------------------------------
def generate_route(lat: float, lng: float, km: float):
    """
    FastAPI(app.py)에서 호출하는 외부 인터페이스.
    - 반환값: (polyline_coords, length_m, algorithm_used)
    """
    coords, length_m, tag = _build_triangle_loop(lat, lng, km)
    return coords, length_m, tag
