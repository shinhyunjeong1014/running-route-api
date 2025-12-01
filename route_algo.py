import math
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Any

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# ----------------------------------------------------
# 기본 설정
# ----------------------------------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "2"))

# 카카오 도보 경로
KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY", "")
# 사용자가 하드코딩한 키가 있다면 주입
if not KAKAO_API_KEY:
    KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64"

KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"

# 런닝 관련
RUNNING_SPEED_KMH = 8.0

# 전역 제한
GLOBAL_TIMEOUT_S = 10.0          # 전체 알고리즘 제한 시간
MAX_TOTAL_CALLS = 30             # Valhalla 최대 호출 횟수
MAX_LENGTH_ERROR_M = 99.0        # 허용 오차: 요청거리 ~ 요청거리 + 99m
MAX_BEST_ROUTES_TO_TEST = 5      # 후보 중 정밀 튜닝 시도 개수


# ----------------------------------------------------
# 거리 / 기하 유틸
# ----------------------------------------------------

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
    """폴리라인의 총 길이(m)."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        total += haversine_m(lat1, lon1, lat2, lon2)
    return total


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """p1 -> p2 대략적인 방위각(0~360 deg)."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def project_point(
    lat: float, lon: float, distance_m: float, bearing_deg_: float
) -> Tuple[float, float]:
    """위경도 + 거리/방위로 새 좌표를 구함."""
    R = 6371000.0
    br = math.radians(bearing_deg_)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(distance_m / R)
        + math.cos(phi1)
        * math.sin(distance_m / R)
        * math.cos(br)
    )
    lam2 = lam1 + math.atan2(
        math.sin(br) * math.sin(distance_m / R) * math.cos(phi1),
        math.cos(distance_m / R) - math.sin(phi1) * math.sin(phi2),
    )
    return (
        math.degrees(phi2),
        (math.degrees(lam2) + 540.0) % 360.0 - 180.0,
    )


# ----------------------------------------------------
# Valhalla / Kakao API
# ----------------------------------------------------

def _get_bounding_box_polygon(
    points: List[Tuple[float, float]], buffer_deg: float = 0.00001
) -> Optional[List[Tuple[float, float]]]:
    """경로 중복 회피용 Bounding Box Polygon."""
    if not points:
        return None

    min_lat = min(p[0] for p in points)
    max_lat = max(p[0] for p in points)
    min_lon = min(p[1] for p in points)
    max_lon = max(p[1] for p in points)

    # 너무 짧은 구간은 의미 없음
    if haversine_m(min_lat, min_lon, max_lat, max_lon) < 20:
        return None

    buf = buffer_deg
    return [
        (min_lat - buf, min_lon - buf),
        (max_lat + buf, min_lon - buf),
        (max_lat + buf, max_lon + buf),
        (min_lat - buf, max_lon + buf),
        (min_lat - buf, min_lon - buf),
    ]


def valhalla_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    avoid_polygons: Optional[List[List[Tuple[float, float]]]] = None,
) -> List[Tuple[float, float]]:
    """Valhalla 보행자 경로 1회 요청."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    last_error: Optional[Exception] = None

    costing_options = {
        "pedestrian": {
            "avoid_steps": 1.0,
            "service_penalty": 1000,
            "use_hills": 0.0,
            "use_ferry": 0.0,
            "track_type_penalty": 0,
            "private_road_penalty": 10000,
            "bicycle_network_preference": 0.5,
            "sidewalk_preference": 1.0,
            "alley_preference": -1.0,
            "max_road_class": 0.5,
        }
    }

    for attempt in range(VALHALLA_MAX_RETRY):
        try:
            payload: Dict[str, Any] = {
                "locations": [
                    {"lat": lat1, "lon": lon1, "type": "break"},
                    {"lat": lat2, "lon": lon2, "type": "break"},
                ],
                "costing": "pedestrian",
                "costing_options": costing_options,
            }
            if avoid_polygons:
                valhalla_polys = []
                for poly in avoid_polygons:
                    valhalla_polys.append([[lon, lat] for lat, lon in poly])
                payload["avoid_polygons"] = valhalla_polys

            resp = requests.post(
                VALHALLA_URL, json=payload, timeout=VALHALLA_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            shape = data["trip"]["legs"][0]["shape"]
            return _decode_polyline(shape)
        except Exception as e:
            last_error = e
            logger.warning(
                "[Valhalla] attempt %d failed for %s -> %s: %s",
                attempt + 1,
                p1,
                p2,
                e,
            )

    logger.error(
        "[Valhalla] all attempts failed for %s -> %s: %s",
        p1,
        p2,
        last_error,
    )
    return []


def kakao_walk_route(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> Optional[List[Tuple[float, float]]]:
    """카카오 길찾기 API (도보)를 호출하여 경로 폴리라인을 반환."""
    if not KAKAO_API_KEY:
        logger.error("[Kakao API] KAKAO_API_KEY not configured.")
        return None

    lon1, lat1 = p1[::-1]
    lon2, lat2 = p2[::-1]

    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {
        "origin": f"{lon1},{lat1}",
        "destination": f"{lon2},{lat2}",
        "waypoints": "",
        "priority": "RECOMMEND",
        "car_model": "walk",
    }

    try:
        resp = requests.get(
            KAKAO_ROUTE_URL,
            params=params,
            headers=headers,
            timeout=VALHALLA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("routes") and data["routes"][0]["result_code"] == 0:
            coords: List[Tuple[float, float]] = []
            for route in data["routes"]:
                for section in route["sections"]:
                    for road in section.get("roads", []):
                        vertices = road.get("vertexes", [])
                        for i in range(0, len(vertices), 2):
                            if i + 1 < len(vertices):
                                lon = vertices[i]
                                lat = vertices[i + 1]
                                coords.append((lat, lon))

            if coords and len(coords) >= 2:
                if coords[0] != p1:
                    coords.insert(0, p1)
                if coords[-1] != p2:
                    coords.append(p2)
                return coords

    except Exception as e:
        logger.error("[Kakao API] Request failed (Parsing or Network): %s", e)
        return None
    return None


def _decode_polyline(shape: str) -> List[Tuple[float, float]]:
    """Valhalla polyline 디코더."""
    coords: List[Tuple[float, float]] = []
    lat = 0
    lng = 0
    idx = 0
    precision = 1e6

    try:
        while idx < len(shape):
            shift = 0
            result = 0
            while True:
                b = ord(shape[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            shift = 0
            result = 0
            while True:
                b = ord(shape[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            current_lat = lat / precision
            current_lng = lng / precision
            if not (-90.0 <= current_lat <= 90.0 and -180.0 <= current_lng <= 180.0):
                logger.error(
                    "[Valhalla Decode] Sanity check failed: (%s, %s)",
                    current_lat,
                    current_lng,
                )
                return []
            coords.append((current_lat, current_lng))
    except IndexError:
        logger.error("[Valhalla Decode] Unexpected end of polyline string.")
    return coords


# ----------------------------------------------------
# 루프 품질 / 거리 제어 유틸
# ----------------------------------------------------

def _loop_roundness(points: List[Tuple[float, float]]) -> float:
    """루프가 얼마나 원형에 가까운지(0~1)."""
    if len(points) < 4:
        return 0.0
    xs = [p[1] for p in points]
    ys = [p[0] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    dists = [haversine_m(cy, cx, lat, lon) for lat, lon in points]
    mean_r = sum(dists) / len(dists)
    if mean_r <= 0:
        return 0.0
    var = sum((d - mean_r) ** 2 for d in dists) / len(dists)
    score = 1.0 / (1.0 + var / (mean_r * mean_r + 1e-6))
    return max(0.0, min(1.0, score))


def _calculate_overlap_penalty(
    seg_out: List[Tuple[float, float]], seg_back: List[Tuple[float, float]]
) -> float:
    """
    복귀 경로(seg_back)가 나가는 경로(seg_out)와 얼마나 많이 겹치는지에 대한 페널티.
    """
    if not seg_out or not seg_back:
        return 0.0

    overlap_count = 0
    OVERLAP_THRESHOLD_DEG = 0.0002  # 위경도 약 20m 근접성

    for lat_c, lon_c in seg_back:
        is_close = False
        for lat_a, lon_a in seg_out:
            if (
                abs(lat_c - lat_a) < OVERLAP_THRESHOLD_DEG
                and abs(lon_c - lon_a) < OVERLAP_THRESHOLD_DEG
            ):
                is_close = True
                break
        if is_close:
            overlap_count += 1

    seg_back_len = len(seg_back)
    if seg_back_len > 0 and overlap_count / seg_back_len > 0.1:
        overlap_ratio = overlap_count / seg_back_len
        # 많이 겹칠수록 거리 점수에 큰 페널티
        return overlap_ratio * 1000.0

    return 0.0


def _try_shrink_path_kakao(
    current_route: List[Tuple[float, float]],
    target_m: float,
    start_time: float,
    global_timeout: float,
) -> List[Tuple[float, float]]:
    """
    경로가 target_m 보다 너무 길 때,
    중간 40~60% 구간을 카카오 도보 경로로 치환해서 단축을 시도한다.
    성공/실패에 관계 없이 항상 '가장 짧아진 버전'을 반환.
    """
    route_to_shrink = current_route[:]

    if len(route_to_shrink) < 4:
        return route_to_shrink

    MAX_SHRINK_ATTEMPTS = 3

    for _ in range(MAX_SHRINK_ATTEMPTS):
        if time.time() - start_time >= global_timeout:
            break

        cur_len = polyline_length_m(route_to_shrink)
        error_m = cur_len - target_m

        # 이미 타겟보다 짧거나, 허용 오차 내로 들어오면 더 이상 단축 X
        if cur_len < target_m or abs(error_m) <= MAX_LENGTH_ERROR_M:
            break

        pts = route_to_shrink
        idx_a = max(1, int(len(pts) * 0.40))
        idx_b = min(len(pts) - 2, int(len(pts) * 0.60))
        if idx_a >= idx_b:
            break

        p_a = pts[idx_a]
        p_b = pts[idx_b]

        reconnect_seg = kakao_walk_route(p_a, p_b)
        if not reconnect_seg or len(reconnect_seg) < 2:
            # 더 이상 단축 불가능
            break

        seg_len_original = polyline_length_m(pts[idx_a : idx_b + 1])
        seg_len_new = polyline_length_m(reconnect_seg)
        reduction = seg_len_original - seg_len_new

        if reduction <= 0:
            # 기존 구간보다 짧아지지 않으면 시도 중단
            break

        # 실제로 교체
        new_route = pts[:idx_a] + reconnect_seg + pts[idx_b + 1 :]

        # 연속 중복 좌표 제거
        cleaned = [new_route[0]]
        for p in new_route[1:]:
            if p != cleaned[-1]:
                cleaned.append(p)

        route_to_shrink = cleaned

    return route_to_shrink


def _extend_path_kakao_spur(
    current_route: List[Tuple[float, float]],
    target_m: float,
    start_time: float,
    global_timeout: float,
) -> List[Tuple[float, float]]:
    """
    현재 루프가 target_m 보다 짧을 때,
    경로 중간 지점에 '왕복 스퍼(spur)'를 붙여 길이를 늘린다.
    - 카카오 도보 경로만 사용 (Valhalla 호출 없음)
    - 한 바퀴 도는 느낌을 유지하면서 루프에 작은 돌출부를 만든다.
    """
    route = current_route[:]
    if len(route) < 3:
        return route

    MAX_EXTEND_ATTEMPTS = 3

    for _ in range(MAX_EXTEND_ATTEMPTS):
        if time.time() - start_time >= global_timeout:
            break

        cur_len = polyline_length_m(route)
        error_m = target_m - cur_len

        # 이미 오차 범위 안이면 종료
        if abs(error_m) <= MAX_LENGTH_ERROR_M:
            break

        # 이미 목표 이상이 되면 여기서는 멈추고, 필요 시 다른 쪽에서 단축
        if error_m <= 0:
            break

        # 남은 거리가 너무 작으면 의미 없는 스퍼
        if error_m < 30.0:
            break

        # 스퍼 길이 설정 (왕복 기준 총 길이)
        extend_len = min(error_m + 50.0, 600.0)
        spur_dist = extend_len / 2.0

        # 중간 지점 선택
        mid_idx = len(route) // 2
        p_mid = route[mid_idx]

        # 대략적인 진행 방향
        if 0 < mid_idx < len(route) - 1:
            br_forward = bearing_deg(
                route[mid_idx][0],
                route[mid_idx][1],
                route[mid_idx + 1][0],
                route[mid_idx + 1][1],
            )
        else:
            br_forward = 0.0

        candidate_bearings = [
            (br_forward + 90.0) % 360.0,
            (br_forward + 270.0) % 360.0,
            0.0,
            180.0,
        ]

        best_spur = None
        best_spur_len = 0.0

        for br in candidate_bearings:
            if time.time() - start_time >= global_timeout:
                break

            spur_point = project_point(p_mid[0], p_mid[1], spur_dist, br)
            spur_out = kakao_walk_route(p_mid, spur_point)
            spur_back = kakao_walk_route(spur_point, p_mid)

            if not spur_out or not spur_back:
                continue

            spur_path = spur_out + spur_back[1:]
            spur_len = polyline_length_m(spur_path)

            if spur_len < 100.0:
                continue

            if spur_len > best_spur_len:
                best_spur_len = spur_len
                best_spur = spur_path

        if not best_spur:
            # 더 이상 연장할 수 없음
            break

        # route[mid_idx] == p_mid 이므로, 스퍼의 첫 점은 중복 제거
        route = route[: mid_idx + 1] + best_spur[1:] + route[mid_idx + 1 :]

        # 루프가 너무 길어졌으면 이후 shrink에서 조정 가능
    return route


def _select_best_by_distance(
    candidates: List[Dict[str, Any]],
    target_m: float,
) -> Optional[Dict[str, Any]]:
    """
    거리 제어 엔진의 핵심:
    1) [target, target+99] 구간에 있는 루프가 있으면 그 중에서
       거리 오차가 가장 작고, roundness가 높은 것을 선택
    2) 없으면 target 이상인 루프 중에서 가장 짧은 것 선택
    3) 그것도 없으면 target 미만 중에서 가장 긴 루프 선택
    """
    if not candidates:
        return None

    # 1단계: 허용 오차 구간 내 후보
    in_band = [
        c
        for c in candidates
        if target_m <= c["len"] <= target_m + MAX_LENGTH_ERROR_M
    ]
    if in_band:
        return min(
            in_band,
            key=lambda c: (abs(c["len"] - target_m), -c["roundness"]),
        )

    # 2단계: target 이상 중에서 가장 짧은 루프
    over = [c for c in candidates if c["len"] > target_m]
    if over:
        return min(
            over,
            key=lambda c: (c["len"], -c["roundness"]),
        )

    # 3단계: target 미만 중에서 가장 긴 루프
    under = [c for c in candidates if c["len"] < target_m]
    if under:
        return max(
            under,
            key=lambda c: (c["len"], c["roundness"]),
        )

    return None


# ----------------------------------------------------
# B안: Start → viaA → viaB → Start 루프 생성
# ----------------------------------------------------

def _generate_candidates_two_segment(
    start: Tuple[float, float],
    target_m: float,
    radii: List[float],
    bearings: List[float],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    B안 구조로 루프 후보를 생성한다.
    Out1: Start → viaA (Valhalla)
    Out2: viaA → viaB (Valhalla)
    Back: viaB → Start (Valhalla + Kakao 후보)
    """
    candidate_routes: List[Dict[str, Any]] = []
    valhalla_calls = 0
    total_routes_checked = 0

    for R in radii:
        if valhalla_calls + 3 > MAX_TOTAL_CALLS:
            break
        if time.time() - start_time >= GLOBAL_TIMEOUT_S:
            break

        for br1 in bearings:
            if valhalla_calls + 3 > MAX_TOTAL_CALLS:
                break
            if time.time() - start_time >= GLOBAL_TIMEOUT_S:
                break

            via_a = project_point(start[0], start[1], R, br1)
            seg_out1 = valhalla_route(start, via_a)
            valhalla_calls += 1
            if not seg_out1 or len(seg_out1) < 2:
                continue

            pivot = seg_out1[-1]

            for br2 in bearings:
                if valhalla_calls + 2 > MAX_TOTAL_CALLS:
                    break
                if time.time() - start_time >= GLOBAL_TIMEOUT_S:
                    break

                via_b = project_point(pivot[0], pivot[1], R, br2)
                seg_out2 = valhalla_route(pivot, via_b)
                valhalla_calls += 1
                if not seg_out2 or len(seg_out2) < 2:
                    continue

                seg_out = seg_out1 + seg_out2[1:]
                comeback_point = seg_out[-1]

                back_segments: List[List[Tuple[float, float]]] = []

                # Valhalla Back
                if valhalla_calls + 1 <= MAX_TOTAL_CALLS:
                    seg_back_v = valhalla_route(comeback_point, start)
                    valhalla_calls += 1
                    if seg_back_v and len(seg_back_v) >= 2:
                        back_segments.append(seg_back_v)

                # Kakao Back
                seg_back_k = kakao_walk_route(comeback_point, start)
                if seg_back_k and len(seg_back_k) >= 2:
                    back_segments.append(seg_back_k)

                for seg_back in back_segments:
                    overlap_penalty = _calculate_overlap_penalty(seg_out, seg_back)
                    if overlap_penalty > 300.0:
                        # 지나치게 많이 겹치는 루프는 제외
                        continue

                    total_route = seg_out + seg_back[1:]
                    if total_route and total_route[0] != start:
                        total_route.insert(0, start)
                    if total_route and total_route[-1] != start:
                        total_route.append(start)

                    # 연속 중복 좌표 제거
                    cleaned = [total_route[0]]
                    for p in total_route[1:]:
                        if p != cleaned[-1]:
                            cleaned.append(p)
                    total_route = cleaned

                    length_m = polyline_length_m(total_route)
                    if length_m <= 0.0:
                        continue

                    roundness = _loop_roundness(total_route)
                    candidate_routes.append(
                        {
                            "route": total_route,
                            "len": length_m,
                            "roundness": roundness,
                            "overlap_penalty": overlap_penalty,
                        }
                    )
                    total_routes_checked += 1

        if valhalla_calls + 3 > MAX_TOTAL_CALLS:
            break

    return candidate_routes, valhalla_calls, total_routes_checked


# ----------------------------------------------------
# 메인 엔트리: generate_area_loop
# ----------------------------------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """
    B안 전체 알고리즘 (+ 거리 제어 엔진 포함):

    1) target_m = km * 1000, 최소 300m 보정
    2) R_ideal을 중심으로 여러 반지름/방위각에 대해
       Start → viaA → viaB → Start 루프 후보 생성
    3) 후보들 중 '거리 제어 엔진'으로 최적 루프 선택
       - 우선순위: [target, target+99] 구간 → 그 외 target 이상 → 불가시 target 미만 중 최장
    4) 선택된 루프가 너무 길면 카카오 단축, 너무 짧으면 카카오 스퍼로 연장
    5) 최종 루프와 메타 정보 반환
    """
    start_time = time.time()
    start = (lat, lng)

    target_m = max(300.0, km * 1000.0)
    km_requested = km

    # 시간 초과 방어 (이론상 바로 걸릴 일은 없지만, 안전장치)
    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
        return [start], {
            "len": 0.0,
            "err": target_m,
            "roundness": 0.0,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": 0,
            "time_s": 0.0,
            "message": "경로 생성 시작 시점에 이미 시간 제한을 초과했습니다.",
            "length_ok": False,
            "routes_checked": 0,
            "routes_processed": 0,
            "routes_validated": 0,
        }

    # 이상적인 반지름
    R_ideal = target_m / (2.0 * math.pi)

    # 기본 R 세트 (소형 루프 ~ 중/대형 루프까지)
    R_MIN = max(80.0, min(R_ideal * 0.3, 250.0))
    R_SMALL = max(150.0, min(R_ideal * 0.6, 400.0))
    R_MEDIUM = max(250.0, min(R_ideal * 1.0, 800.0))
    R_LARGE = max(400.0, min(R_ideal * 1.3, 1500.0))
    R_XLARGE = max(600.0, min(R_ideal * 1.6, 2000.0))

    base_radii = sorted({R_MIN, R_SMALL, R_MEDIUM, R_LARGE, R_XLARGE})
    base_bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    # 1차: 기본 R, 전체 방위각으로 후보 생성
    candidate_routes, valhalla_calls, total_routes_checked = _generate_candidates_two_segment(
        start,
        target_m,
        base_radii,
        base_bearings,
        start_time,
    )

    # 2차: 거리 보정용 R 스케일링 (옵션)
    #  - 1차에서 band 안의 후보를 못 찾았고
    #  - 아직 시간/호출 여유가 있을 때만 수행
    primary_best = _select_best_by_distance(candidate_routes, target_m)
    if (
        primary_best is not None
        and not (target_m <= primary_best["len"] <= target_m + MAX_LENGTH_ERROR_M)
        and valhalla_calls + 6 <= MAX_TOTAL_CALLS  # 2차에서도 어느 정도 호출 여유 확보
        and time.time() - start_time < GLOBAL_TIMEOUT_S * 0.7
    ):
        # 현재 가장 가까운 루프를 기준으로 스케일링 비율 결정
        if primary_best["len"] > 0:
            scale = max(0.4, min(2.0, target_m / primary_best["len"]))
        else:
            scale = 1.0

        scaled_radii = sorted({max(80.0, r * scale) for r in base_radii})

        # 2차에서는 방위각을 4방향으로 제한해서 탐색 비용 절감
        second_bearings = [0, 90, 180, 270]

        candidate_routes_2, valhalla_calls_2, checked_2 = _generate_candidates_two_segment(
            start,
            target_m,
            scaled_radii,
            second_bearings,
            start_time,
        )

        valhalla_calls += valhalla_calls_2
        total_routes_checked += checked_2
        candidate_routes.extend(candidate_routes_2)

    # 전체 후보 중 거리 관점 최적 루프 선택
    best_candidate = _select_best_by_distance(candidate_routes, target_m)

    if not best_candidate:
        # 후보 자체가 하나도 없을 때 (Valhalla 죽었거나, 완전히 단절된 지형)
        return [start], {
            "len": 0.0,
            "err": target_m,
            "roundness": 0.0,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": valhalla_calls,
            "time_s": round(time.time() - start_time, 2),
            "message": "탐색 결과, 유효한 루프 후보를 찾을 수 없습니다.",
            "length_ok": False,
            "routes_checked": total_routes_checked,
            "routes_processed": 0,
            "routes_validated": 0,
        }

    route = best_candidate["route"]
    route_len = best_candidate["len"]

    # ------------------------------------------------
    # 거리 제어 엔진: 카카오 스퍼 / 단축으로 미세 조정
    # ------------------------------------------------
    # 1) 너무 긴 경우 → 단축 시도
    if route_len > target_m + MAX_LENGTH_ERROR_M:
        shrunk = _try_shrink_path_kakao(
            route,
            target_m,
            start_time,
            GLOBAL_TIMEOUT_S,
        )
        shrunk_len = polyline_length_m(shrunk)

        # shrink 결과가 더 좋으면 교체
        if abs(shrunk_len - target_m) < abs(route_len - target_m):
            route = shrunk
            route_len = shrunk_len

    # 2) 여전히 짧고 오차가 크면 → 스퍼로 연장
    if route_len < target_m or abs(route_len - target_m) > MAX_LENGTH_ERROR_M:
        extended = _extend_path_kakao_spur(
            route,
            target_m,
            start_time,
            GLOBAL_TIMEOUT_S,
        )
        ext_len = polyline_length_m(extended)

        # extend 결과가 더 좋으면 교체
        if abs(ext_len - target_m) < abs(route_len - target_m):
            route = extended
            route_len = ext_len

    # 단축/연장 이후에도 너무 길면 한 번 더 단축 시도(옵션)
    if route_len > target_m + MAX_LENGTH_ERROR_M:
        shrunk2 = _try_shrink_path_kakao(
            route,
            target_m,
            start_time,
            GLOBAL_TIMEOUT_S,
        )
        shrunk2_len = polyline_length_m(shrunk2)
        if abs(shrunk2_len - target_m) < abs(route_len - target_m):
            route = shrunk2
            route_len = shrunk2_len

    # ------------------------------------------------
    # 최종 메타 정보 정리
    # ------------------------------------------------
    final_roundness = _loop_roundness(route)
    final_err = route_len - target_m
    is_perfect = (route_len >= target_m) and (route_len <= target_m + MAX_LENGTH_ERROR_M)

    meta = {
        "len": route_len,
        "err": final_err,
        "roundness": final_roundness,
        "success": is_perfect,
        "used_fallback": False,
        "valhalla_calls": valhalla_calls,
        "time_s": round(time.time() - start_time, 2),
        "message": "요청 거리에 맞는 루프를 생성했습니다."
        if is_perfect
        else "허용 오차(±99m)를 만족하지는 않지만, 가장 인접한 루프를 반환합니다.",
        "length_ok": is_perfect,
        "routes_checked": total_routes_checked,
        "routes_processed": len(candidate_routes),
        "routes_validated": 1 if is_perfect else 0,
        "km_requested": km_requested,
        "target_m": target_m,
    }

    return route, meta
