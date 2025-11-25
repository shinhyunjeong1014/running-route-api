import math
import random
from typing import List, Dict, Tuple

import requests


VALHALLA_URL = "http://localhost:8002/route"


# ============================
# 유틸 함수들
# ============================

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def cumulative_distance(polyline: List[Dict[str, float]]) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for p, q in zip(polyline[:-1], polyline[1:]):
        total += haversine_m(p["lat"], p["lng"], q["lat"], q["lng"])
    return total


def decode_polyline6(encoded: str) -> List[Tuple[float, float]]:
    """
    Valhalla shape 디코딩 (precision=6)
    """
    if not encoded:
        return []

    coordinates = []
    index = 0
    lat = 0
    lng = 0
    factor = 10 ** 6

    length = len(encoded)

    while index < length:
        # latitude
        result = 0
        shift = 0
        while True:
            if index >= length:
                break
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        delta_lat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += delta_lat

        # longitude
        result = 0
        shift = 0
        while True:
            if index >= length:
                break
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        delta_lng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += delta_lng

        coordinates.append((lat / factor, lng / factor))

    return coordinates


def dest_point(lat: float, lng: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """
    시작 좌표에서 특정 방향/거리만큼 떨어진 위경도 계산 (단순 구면 모델)
    """
    R = 6371000.0
    brng = math.radians(bearing_deg)

    lat1 = math.radians(lat)
    lon1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


# ============================
# Valhalla 호출
# ============================

def call_valhalla_route(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    costing: str = "pedestrian",
    timeout: float = 5.0,
) -> List[Dict[str, float]]:
    """
    Valhalla /route API를 호출해 도보 경로 polyline을 반환.
    실패 시 예외 발생.
    """
    payload = {
        "locations": [
            {"lat": start_lat, "lon": start_lng},
            {"lat": end_lat, "lon": end_lng},
        ],
        "costing": costing,
        "directions_options": {"units": "meters"},
    }

    resp = requests.post(VALHALLA_URL, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Valhalla HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    trip = data.get("trip")
    if not trip:
        raise RuntimeError("Valhalla 응답에 trip 데이터가 없습니다.")

    legs = trip.get("legs", [])
    if not legs:
        raise RuntimeError("Valhalla 응답에 legs가 없습니다.")

    shape = legs[0].get("shape")
    if not shape:
        raise RuntimeError("Valhalla 응답에 shape가 없습니다.")

    coords = decode_polyline6(shape)
    polyline = [{"lat": lat, "lng": lng} for lat, lng in coords]
    return polyline


# ============================
# 루프 품질 평가
# ============================

def loop_score(polyline: List[Dict[str, float]]) -> float:
    """
    루프 품질을 대략 평가하는 점수.
    - 길이가 너무 짧거나
    - 거의 직선 왕복에 가까우면 낮은 점수.
    """
    if not polyline or len(polyline) < 4:
        return 0.0

    length = cumulative_distance(polyline)
    if length < 300:  # 300m 미만은 너무 짧음
        return 0.0

    # 루프의 "퍼짐" 정도 평가 (bounding box)
    lats = [p["lat"] for p in polyline]
    lngs = [p["lng"] for p in polyline]
    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)

    diag = haversine_m(min_lat, min_lng, max_lat, max_lng,)

    # 길이가 긴데도 bbox 대각선이 너무 짧으면 직선에 가까운 경향
    straightness = diag / max(length, 1.0)  # 0 ~ 1 사이쯤
    # straightness가 너무 크거나 너무 작으면 점수 낮게
    # (적당한 루프는 length 대비 bbox diag가 중간 정도)
    penalty = abs(straightness - 0.35)  # 0.35 정도를 이상적인 값으로 둠

    # 중복 점 비율 (왕복 여부 대략 판단)
    unique_points = {(round(p["lat"], 6), round(p["lng"], 6)) for p in polyline}
    unique_ratio = len(unique_points) / len(polyline)

    base = length  # 기본은 길이가 긴 루프가 유리
    base *= max(unique_ratio, 0.5)  # 중복이 많으면 페널티
    base *= max(0.1, 1.0 - penalty)  # 직선형이면 페널티

    return base


# ============================
# 루프 생성 (Valhalla 기반)
# ============================

def generate_loop_route(
    start_lat: float,
    start_lng: float,
    km: float,
    bearings: List[float] = None,
) -> Tuple[List[Dict[str, float]], float, Dict]:
    """
    Valhalla를 이용해 start를 기준으로 루프를 생성한다.
    - start -> mid -> start 형태로 여러 후보를 만들고
    - 목표 거리(km)에 가까우면서 loop_score가 높은 것을 선택
    """
    if bearings is None:
        # 다양한 방향으로 샘플 (0~360도)
        bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    target_m = km * 1000.0
    radius_m = target_m * 0.5  # 중간 지점까지 대략 거리

    candidates = []

    for b in bearings:
        mid_lat, mid_lng = dest_point(start_lat, start_lng, b, radius_m)

        try:
            # start -> mid
            out_poly = call_valhalla_route(start_lat, start_lng, mid_lat, mid_lng)
            # mid -> start
            back_poly = call_valhalla_route(mid_lat, mid_lng, start_lat, start_lng)

            if len(out_poly) < 2 or len(back_poly) < 2:
                continue

            # 루프 결합 (중간 지점 중복 제거)
            loop_poly = out_poly + back_poly[1:]

            length_m = cumulative_distance(loop_poly)
            score = loop_score(loop_poly)

            # 너무 짧거나 너무 긴 루프는 제외 (느슨한 필터)
            if length_m < 0.5 * target_m:
                continue

            candidates.append((loop_poly, length_m, score, b))

        except Exception:
            # 해당 방향으로는 경로를 만들 수 없는 경우 무시
            continue

    if not candidates:
        raise RuntimeError("Valhalla 기반 루프 후보를 생성하지 못했습니다.")

    # 1차로 score 기준 정렬 (루프 품질)
    # 2차로 목표 거리와의 차이 기준 정렬
    candidates.sort(
        key=lambda item: (-item[2], abs(item[1] - target_m))
    )

    best_poly, best_len, best_score, best_bearing = candidates[0]

    meta = {
        "engine": "valhalla_loop_v1",
        "valhalla_url": VALHALLA_URL,
        "target_m": target_m,
        "length_m": round(best_len, 1),
        "bearing_selected": best_bearing,
        "candidate_count": len(candidates),
    }

    return best_poly, best_len, meta
