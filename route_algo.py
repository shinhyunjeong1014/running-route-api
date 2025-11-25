import math
import random
import logging
from typing import List, Dict, Tuple

import requests

logger = logging.getLogger(__name__)

# 로컬에서 띄워둔 Valhalla 서버 주소
VALHALLA_URL = "http://localhost:8002/route"


# ======================
# 기본 거리/좌표 유틸
# ======================
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """단순 해버사인 거리(m)."""
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
    """turn_algo에서 import해서 쓰는 누적 거리 유틸."""
    dists = [0.0]
    if not polyline or len(polyline) < 2:
        return dists
    for p, q in zip(polyline[:-1], polyline[1:]):
        dists.append(dists[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return dists


def cumulative_distance(polyline: List[Dict[str, float]]) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    return _cumulative_distances(polyline)[-1]


def decode_polyline6(encoded: str) -> List[Tuple[float, float]]:
    """
    Valhalla polyline 디코더 (precision=6).
    """
    if not encoded:
        return []

    coords: List[Tuple[float, float]] = []
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
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20 or index >= length:
                break
        delta_lat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += delta_lat

        # longitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20 or index >= length:
                break
        delta_lng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += delta_lng

        coords.append((lat / factor, lng / factor))

    return coords


def dest_point(lat: float, lng: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """시작점에서 특정 방향/거리만큼 떨어진 위경도 계산."""
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


# ======================
# Valhalla 호출
# ======================
def call_valhalla_route(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    costing: str = "pedestrian",
    timeout: float = 5.0,
) -> List[Dict[str, float]]:
    """
    Valhalla /route API를 호출해서 polyline을 받아온다.
    실패하면 예외를 던진다.
    """
    payload = {
        "locations": [
            {"lat": start_lat, "lon": start_lng},
            {"lat": end_lat, "lon": end_lng},
        ],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }

    resp = requests.post(VALHALLA_URL, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Valhalla HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    trip = data.get("trip")
    if not trip:
        raise RuntimeError("Valhalla 응답에 trip이 없습니다.")

    legs = trip.get("legs") or []
    if not legs:
        raise RuntimeError("Valhalla 응답에 legs가 없습니다.")

    shape = legs[0].get("shape")
    if not shape:
        raise RuntimeError("Valhalla 응답에 shape가 없습니다.")

    coords = decode_polyline6(shape)
    return [{"lat": la, "lng": lo} for la, lo in coords]


# ======================
# 루프 품질 평가
# ======================
def loop_score(polyline: List[Dict[str, float]]) -> float:
    """
    루프 품질 점수.
    - 길이
    - bbox 퍼짐
    - 중복 포인트 비율 등으로 대략 평가
    """
    if not polyline or len(polyline) < 4:
        return 0.0

    length = cumulative_distance(polyline)
    if length < 300:  # 300m 미만은 러닝 코스로 보기 어렵다.
        return 0.0

    lats = [p["lat"] for p in polyline]
    lngs = [p["lng"] for p in polyline]
    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)

    diag = haversine_m(min_lat, min_lng, max_lat, max_lng)
    straightness = diag / max(length, 1.0)  # 0~1 사이 값

    # 0.3~0.5 정도가 적당한 루프라고 가정하고 페널티 부여
    penalty = abs(straightness - 0.4)

    unique_points = {(round(p["lat"], 6), round(p["lng"], 6)) for p in polyline}
    unique_ratio = len(unique_points) / len(polyline)

    base = length
    base *= max(unique_ratio, 0.4)       # 중복이 너무 많으면 페널티
    base *= max(0.1, 1.0 - penalty)      # 너무 일자형/너무 꼬인 경로면 페널티

    return base


# ======================
# 간단 fallback 루프
# ======================
def simple_out_and_back_loop(start_lat: float, start_lng: float, km: float):
    """
    모든 후보 생성이 실패했을 때를 위한 아주 단순한 fallback.
    여러 방향으로 start -> mid -> start 왕복 경로 중 하나라도 되면 채택.
    """
    target_m = km * 1000.0
    radius = max(300.0, min(target_m * 0.45, 1500.0))
    bearings = [0, 60, 120, 180, 240, 300]

    for base_b in bearings:
        for jitter in [-15, 0, 15]:
            b = (base_b + jitter) % 360
            mid_lat, mid_lng = dest_point(start_lat, start_lng, b, radius)
            try:
                out_poly = call_valhalla_route(start_lat, start_lng, mid_lat, mid_lng)
                back_poly = call_valhalla_route(mid_lat, mid_lng, start_lat, start_lng)
                if len(out_poly) < 2 or len(back_poly) < 2:
                    continue
                loop_poly = out_poly + back_poly[1:]
                length = cumulative_distance(loop_poly)
                if length > 200:  # 최소 200m
                    return loop_poly, length, {
                        "engine": "valhalla_loop_fallback_simple",
                        "bearing": b,
                        "radius_m": round(radius, 1),
                        "length_m": round(length, 1),
                        "target_m": round(target_m, 1),
                        "fallback": True,
                        "schema_version": "1.0.0",
                    }
            except Exception as e:
                logger.warning(f"simple fallback 실패(bearing={b}): {e}")
                continue

    raise RuntimeError("Valhalla 기반 간단 fallback 루프도 생성하지 못했습니다.")


# ======================
# 메인: 루프 생성
# ======================
def generate_loop_route(
    start_lat: float,
    start_lng: float,
    km: float,
    bearings: List[float] = None,
):
    """
    Valhalla 기반 루프 생성.
    - 여러 방향/반경으로 mid를 찍어서 start -> mid -> start 루프 생성
    - 품질/거리 기준으로 best 선택
    - 모든 후보 실패 시 simple_out_and_back_loop로 한 번 더 시도
    """
    if bearings is None:
        bearings = [0, 60, 120, 180, 240, 300]

    target_m = km * 1000.0
    base_radius = target_m * 0.45
    base_radius = max(400.0, min(base_radius, 2000.0))  # 400m~2km

    candidates = []
    first_success = None
    total_attempts = 0
    success_attempts = 0

    for b in bearings:
        for scale in [0.7, 1.0, 1.3]:
            radius_m = base_radius * scale
            jitter = random.uniform(-20, 20)
            bearing = (b + jitter) % 360
            total_attempts += 1

            mid_lat, mid_lng = dest_point(start_lat, start_lng, bearing, radius_m)

            try:
                out_poly = call_valhalla_route(start_lat, start_lng, mid_lat, mid_lng)
                back_poly = call_valhalla_route(mid_lat, mid_lng, start_lat, start_lng)
                if len(out_poly) < 2 or len(back_poly) < 2:
                    continue

                loop_poly = out_poly + back_poly[1:]
                length_m = cumulative_distance(loop_poly)
                success_attempts += 1

                if first_success is None:
                    first_success = (loop_poly, length_m, bearing, radius_m)

                # 길이 필터 (매우 느슨하게)
                if length_m < 0.3 * target_m or length_m > 2.5 * target_m:
                    continue

                score = loop_score(loop_poly)
                if score <= 0:
                    continue

                candidates.append(
                    {
                        "polyline": loop_poly,
                        "length_m": length_m,
                        "score": score,
                        "bearing": bearing,
                        "radius_m": radius_m,
                    }
                )
            except Exception as e:
                logger.warning(f"Valhalla 경로 생성 실패(bearing={bearing}, r={radius_m}): {e}")
                continue

    if candidates:
        candidates.sort(key=lambda c: (-c["score"], abs(c["length_m"] - target_m)))
        best = candidates[0]
        poly = best["polyline"]
        length_m = best["length_m"]
        meta = {
            "engine": "valhalla_loop_v2",
            "valhalla_url": VALHALLA_URL,
            "target_m": round(target_m, 1),
            "length_m": round(length_m, 1),
            "bearing_selected": round(best["bearing"], 1),
            "radius_m": round(best["radius_m"], 1),
            "candidate_count": len(candidates),
            "attempts_total": total_attempts,
            "attempts_success": success_attempts,
            "fallback": False,
            "schema_version": "1.0.0",
        }
        return poly, length_m, meta

    # 후보는 없지만, 최소한 한 번이라도 out/back 루프는 성공한 경우 → 이것이라도 사용
    if first_success is not None:
        loop_poly, length_m, bearing, radius_m = first_success
        meta = {
            "engine": "valhalla_loop_first_success",
            "valhalla_url": VALHALLA_URL,
            "target_m": round(target_m, 1),
            "length_m": round(length_m, 1),
            "bearing_selected": round(bearing, 1),
            "radius_m": round(radius_m, 1),
            "candidate_count": 0,
            "attempts_total": total_attempts,
            "attempts_success": success_attempts,
            "fallback": True,
            "schema_version": "1.0.0",
        }
        return loop_poly, length_m, meta

    # 여기까지 왔으면 Valhalla 호출 자체가 거의 다 실패한 상황 → simple fallback 한 번 더 시도
    logger.warning(
        f"Valhalla 기반 루프 후보를 생성하지 못했습니다. "
        f"(attempts_total={total_attempts}, attempts_success={success_attempts})"
    )
    return simple_out_and_back_loop(start_lat, start_lng, km)
