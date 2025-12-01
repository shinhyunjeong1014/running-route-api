import math
import time
import json
import logging
import requests
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ====================================================
# 0. Kakao API Key (요청대로 직접 포함)
# ====================================================
KAKAO_REST_KEY = "dc3686309f8af498d7c62bed0321ee64"


# ====================================================
# 1. 거리 / 기하 유틸리티
# ====================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이 거리(m) 계산"""
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2)**2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2)**2
    )
    return 2 * R * math.asin(math.sqrt(a))


def polyline_length_m(poly):
    """polyline 길이(m)"""
    if not poly or len(poly) < 2:
        return 0.0
    total = 0.0
    for (lat1, lng1), (lat2, lng2) in zip(poly[:-1], poly[1:]):
        total += haversine_m(lat1, lng1, lat2, lng2)
    return total


def project_point(lat, lng, dist_m, bearing_deg):
    """
    시작점에서 dist_m 만큼 bearing 방향으로 이동한 좌표 계산
    """
    R = 6371000.0
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(dist_m / R)
        + math.cos(lat1) * math.sin(dist_m / R) * math.cos(br)
    )
    lng2 = lng1 + math.atan2(
        math.sin(br) * math.sin(dist_m / R) * math.cos(lat1),
        math.cos(dist_m / R) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lng2)


def _loop_roundness(poly):
    """
    루프 모양 점수: 0~1 (1에 가까울수록 더 둥근 루프)
    """
    if not poly:
        return 0
    # 중심 (평균)
    cx = sum(p[1] for p in poly) / len(poly)
    cy = sum(p[0] for p in poly) / len(poly)

    dists = []
    for lat, lng in poly:
        d = haversine_m(cy, cx, lat, lng)
        dists.append(d)

    if not dists:
        return 0
    mean_d = sum(dists) / len(dists)
    var = sum((d - mean_d)**2 for d in dists) / len(dists)
    score = 1 / (1 + var / (mean_d**2 + 1e-9))
    return score


# ====================================================
# 2. Valhalla 라우팅
# ====================================================

VALHALLA_URL = "http://localhost:8002/route"   # 실제 서버 주소로 바꾸면 됨

def valhalla_route(start: Tuple[float, float]], dest: Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Valhalla 도보 경로 요청
    """
    lat1, lng1 = start
    lat2, lng2 = dest

    body = {
        "locations": [
            {"lat": lat1, "lon": lng1},
            {"lat": lat2, "lon": lng2},
        ],
        "costing": "pedestrian",
        "directions_options": {"units": "kilometers"},
    }

    resp = requests.post(VALHALLA_URL, json=body, timeout=10)
    resp.raise_for_status()
    js = resp.json()

    if "routes" not in js or not js["routes"]:
        return []

    shape = js["routes"][0]["shape"]
    # shape decoding
    return decode_valhalla_polyline(shape)


def decode_valhalla_polyline(encoded):
    """Valhalla polyline decode"""
    # 여기서는 간단히 구현된 decode 함수 넣어둘게
    # (실제 사용 중이라면 기존 decode 함수 그대로 사용)
    coords = []
    index = lat = lng = 0
    shift = result = 0

    while index < len(encoded):
        for unit in (lat, lng):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if not (b & 0x20):
                    break
            d = ~(result >> 1) if result & 1 else (result >> 1)
            if unit is lat:
                lat += d
            else:
                lng += d
        coords.append((lat / 1e6, lng / 1e6))
    return coords


# ====================================================
# 3. Kakao 도보 경로 라우팅
# ====================================================

def kakao_walk_route(start: Tuple[float, float]], dest: Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Kakao Mobility Directions API: 보행자(Walk) 경로 요청
    """
    lat1, lng1 = start
    lat2, lng2 = dest

    url = "https://apis-navi.kakaomobility.com/v1/directions"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "origin": {"x": lng1, "y": lat1},
        "destination": {"x": lng2, "y": lat2},
        "priority": "RECOMMEND",
        "car_model": "walk"
    }

    resp = requests.post(url, json=body, headers=headers, timeout=10)
    resp.raise_for_status()
    js = resp.json()

    if "routes" not in js or not js["routes"]:
        return []

    # Kakao 좌표 배열 수집
    # routes[0].sections[].roads[].vertexes: [x1,y1, x2,y2, ...]
    points = []
    for sec in js["routes"][0]["sections"]:
        for road in sec["roads"]:
            v = road["vertexes"]
            # v는 x,y,x,y,... 구조
            for i in range(0, len(v), 2):
                x = v[i]
                y = v[i + 1]
                points.append((y, x))  # (lat, lng)

    return points


# ====================================================
# 4. Out-and-back 루프 생성
# ====================================================

def _build_out_and_back_loop(route: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """[p0...pn] → [p0...pn...p0] 형태 왕복 루프"""
    if not route or len(route) < 2:
        return []
    return route + list(reversed(route[1:-1]))


# ====================================================
# 5. C안: Valhalla 기반 + Kakao 안전망
# ====================================================

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
    *,
    max_valhalla_calls: int = 30,
    km_tolerance: float = 0.099,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    C안 구현:
      1) Valhalla 기반 루프 생성
      2) 만족 못하면 Kakao 기반 루프 생성
      3) 둘 다 실패하면 최소 사각형 루프
    """
    start_t = time.time()
    target_m = km * 1000
    tolerance_m = km_tolerance * 1000

    start = (lat, lng)

    valhalla_calls = 0
    kakao_calls = 0
    routes_checked = 0
    routes_validated = 0

    # 후보 저장
    candidates = []

    # ------------------------------------------------
    # 1. Valhalla 후보들
    # ------------------------------------------------
    bearings = [0, 45, 90, 135, 180, 225, 270, 315]
    radius_cnt = [0.35, 0.4, 0.45, 0.5]

    for f in radius_cnt:
        if valhalla_calls >= max_valhalla_calls:
            break
        R = target_m * f
        for b in bearings:
            if valhalla_calls >= max_valhalla_calls:
                break

            dest_lat, dest_lng = project_point(lat, lng, R, b)
            try:
                route = valhalla_route(start, (dest_lat, dest_lng))
                valhalla_calls += 1
            except:
                continue

            routes_checked += 1
            if not route:
                continue

            loop = _build_out_and_back_loop(route)
            if not loop:
                continue

            length = polyline_length_m(loop)
            if length < target_m * 0.4:
                continue

            err = abs(length - target_m)
            roundness = _loop_roundness(loop)
            routes_validated += 1

            candidates.append({
                "poly": loop,
                "len": length,
                "err": err,
                "round": roundness,
                "fallback": False,
            })

    # ------------------------------------------------
    # 2. Kakao 후보들 (Valhalla가 실패하거나 오차 큰 경우)
    # ------------------------------------------------
    use_kakao = False
    if not candidates or min(c["err"] for c in candidates) > tolerance_m:
        use_kakao = True

    if use_kakao:
        kb = [0, 90, 180, 270]
        kr = [0.35, 0.4, 0.45, 0.5]

        for f in kr:
            R = target_m * f
            for b in kb:
                dest_lat, dest_lng = project_point(lat, lng, R, b)
                try:
                    route = kakao_walk_route(start, (dest_lat, dest_lng))
                    kakao_calls += 1
                except:
                    continue

                routes_checked += 1
                if not route:
                    continue

                loop = _build_out_and_back_loop(route)
                if not loop:
                    continue

                length = polyline_length_m(loop)
                if length < target_m * 0.4:
                    continue

                err = abs(length - target_m)
                roundness = _loop_roundness(loop)
                routes_validated += 1

                candidates.append({
                    "poly": loop,
                    "len": length,
                    "err": err,
                    "round": roundness,
                    "fallback": True,
                })

    # ------------------------------------------------
    # 3. 최악의 상황 → 사각형 루프
    # ------------------------------------------------
    if not candidates:
        d = max(50, target_m * 0.1)
        sq = []
        for b in [0, 90, 180, 270, 0]:
            sq.append(project_point(lat, lng, d, b))
        length = polyline_length_m(sq)
        err = abs(length - target_m)
        roundness = _loop_roundness(sq)

        meta = {
            "len": length,
            "err": err,
            "roundness": roundness,
            "success": False,
            "length_ok": False,
            "used_fallback": True,
            "valhalla_calls": valhalla_calls,
            "kakao_calls": kakao_calls,
            "routes_checked": routes_checked,
            "routes_validated": routes_validated,
            "km_requested": km,
            "target_m": target_m,
            "time_s": time.time() - start_t,
            "message": "Valhalla/Kakao 모두 실패하여 최소 사각형 루프 생성"
        }
        return sq, meta

    # ------------------------------------------------
    # 4. 최적 후보 선택
    # ------------------------------------------------
    # (err, -roundness) 기준 정렬
    candidates.sort(key=lambda c: (c["err"], -c["round"]))

    best = candidates[0]
    err = best["err"]
    success = err <= tolerance_m

    if success:
        msg = f"요청 오차(±{int(tolerance_m)}m) 만족"
    else:
        msg = f"요청 오차 벗어남 → err={int(err)}m"

    meta = {
        "len": best["len"],
        "err": best["err"],
        "roundness": best["round"],
        "success": success,
        "length_ok": success,
        "used_fallback": best["fallback"],
        "valhalla_calls": valhalla_calls,
        "kakao_calls": kakao_calls,
        "routes_checked": routes_checked,
        "routes_validated": routes_validated,
        "km_requested": km,
        "target_m": target_m,
        "time_s": time.time() - start_t,
        "message": msg
    }

    return best["poly"], meta
