import math
import os
import time
import logging
import json
from typing import List, Dict, Tuple, Optional, Any

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 기본 설정 (강화)
# -----------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "2"))

KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"

RUNNING_SPEED_KMH = 8.0
GLOBAL_TIMEOUT_S = 10.0
MAX_TOTAL_CALLS = 30
MAX_LENGTH_ERROR_M = 99.0
MAX_BEST_ROUTES_TO_TEST = 5
MAX_ROUTES_TO_PROCESS = 10

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


# -----------------------------
# Valhalla/Kakao API 호출
# -----------------------------


def _get_bounding_box_polygon(
    points: List[Tuple[float, float]], buffer_deg: float = 0.00001
) -> Optional[List[Tuple[float, float]]]:
    """경로 중복 회피용 Bounding Box Polygon을 생성합니다."""
    if not points:
        return None

    min_lat = min(p[0] for p in points)
    max_lat = max(p[0] for p in points)
    min_lon = min(p[1] for p in points)
    max_lon = max(p[1] for p in points)

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
    is_shrink_attempt: bool = False,
) -> List[Tuple[float, float]]:
    lat1, lon1 = p1
    lat2, lon2 = p2
    last_error: Optional[Exception] = None

    # [핵심] 도보 전용 경로 강제를 위한 Costing Options
    costing_options = {
        "pedestrian": {
            "avoid_steps": 1.0,
            "service_penalty": 1000,
            "use_hills": 0.0,
            "use_ferry": 0.0,
            "track_type_penalty": 0,  # 좁은 길 패널티 제거 (탐색 유연화)
            "private_road_penalty": 10000,
            # [최종 보강] 넓은 길 활용: 자전거 네트워크 활용 및 차도 회피 강화
            "bicycle_network_preference": 0.5,  # 자전거 네트워크 선호도
            "sidewalk_preference": 1.0,
            "alley_preference": -1.0,
            "max_road_class": 0.5,  # 보행자가 이용할 수 있는 최대 도로 등급 제한
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
    """카카오 길찾기 API (도보)를 호출하여 경로 폴리라인을 반환"""
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
                    f"[Valhalla Decode] Sanity check failed: ({current_lat}, {current_lng})"
                )
                return []
            coords.append((current_lat, current_lng))
    except IndexError:
        logger.error("[Valhalla Decode] Unexpected end of polyline string.")
        return []
    return coords


# -----------------------------
# 루프 품질 평가 / 단축 재연결 로직
# -----------------------------


def _loop_roundness(points: List[Tuple[float, float]]) -> float:
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


def _score_loop(
    points: List[Tuple[float, float]], target_m: float
) -> Tuple[float, Dict]:
    length_m = polyline_length_m(points)
    if length_m <= 0.0:
        return float("inf"), {
            "len": 0.0,
            "err": target_m,
            "roundness": 0.0,
            "score": float("inf"),
        }
    err = abs(length_m - target_m)
    roundness = _loop_roundness(points)
    score = err + (1.0 - roundness) * 0.3 * target_m
    length_ok = True
    return score, {
        "len": length_m,
        "err": err,
        "roundness": roundness,
        "score": score,
        "length_ok": length_ok,
    }


def _is_path_safe(points: List[Tuple[float, float]]) -> bool:
    """ 안전성 기준을 제거했으므로, 이 함수는 항상 True를 반환합니다. """
    return True


def _try_shrink_path_kakao(
    current_route: List[Tuple[float, float]],
    target_m: float,
    valhalla_calls: int,
    start_time: float,
    global_timeout: float,
) -> Tuple[Optional[List[Tuple[float, float]]], int]:
    current_len = polyline_length_m(current_route)
    error_m = current_len - target_m

    if time.time() - start_time >= global_timeout:
        return None, valhalla_calls

    pts = current_route
    idx_a = max(1, int(len(pts) * 0.40))
    idx_b = min(len(pts) - 2, int(len(pts) * 0.60))

    if idx_a < idx_b:
        p_a = pts[idx_a]
        p_b = pts[idx_b]

        # [핵심] 반복 단축 시도 시작
        MAX_SHRINK_ATTEMPTS = 3
        route_to_shrink = current_route[:]

        for attempt in range(MAX_SHRINK_ATTEMPTS):
            current_len = polyline_length_m(route_to_shrink)
            error_m = current_len - target_m

            if abs(error_m) <= MAX_LENGTH_ERROR_M:
                return route_to_shrink, valhalla_calls  # 목표 달성
            if error_m < 0:
                break

            if time.time() - start_time >= global_timeout:
                break

            reconnect_seg = kakao_walk_route(p_a, p_b)

            if reconnect_seg and len(reconnect_seg) >= 2:
                seg_len_original = polyline_length_m(pts[idx_a : idx_b + 1])
                seg_len_new = polyline_length_m(reconnect_seg)
                reduction = seg_len_original - seg_len_new

                if reduction > 0:  # 단축 효과가 있다면 경로 교체
                    new_route = pts[:idx_a] + reconnect_seg + pts[idx_b + 1 :]
                    route_to_shrink = new_route[:]  # 다음 반복을 위해 경로 업데이트
                else:
                    break

    # 최종 검증 후, 목표 달성했으면 경로 반환, 아니면 None 반환
    if (
        abs(polyline_length_m(current_route) - target_m)
        <= MAX_LENGTH_ERROR_M
    ):
        return current_route, valhalla_calls
    else:
        return None, valhalla_calls


def _calculate_overlap_penalty(
    seg_out: List[Tuple[float, float]], seg_back: List[Tuple[float, float]]
) -> float:
    """
    복귀 경로(seg_back)가 나가는 경로(seg_out)와 공간적으로 겹치는 정도를 측정하여 페널티를 부과합니다.
    """
    if not seg_out or not seg_back:
        return 0.0

    overlap_count = 0
    OVERLAP_THRESHOLD_DEG = 0.0002  # 약 20m 근접성

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
        # 10% 이상 겹치면 페널티
        overlap_ratio = overlap_count / seg_back_len
        return overlap_ratio * 1000.0  # 1000m 상당의 페널티 부과 (최대화)

    return 0.0


# -----------------------------
# 루프 길이 "연장" 스퍼 추가 (NEW)
# -----------------------------


def _try_extend_path_kakao(
    current_route: List[Tuple[float, float]],
    target_m: float,
    start_time: float,
    global_timeout: float,
) -> Optional[List[Tuple[float, float]]]:
    """
    경로 길이가 target_m보다 짧을 때,
    중간 지점에서 카카오 도보 경로를 왕복 스퍼로 붙여 길이를 늘린다.
    - 되돌아가는 메인 Back 경로는 여전히 '최단 복귀' 전략을 유지
    - 여기서는 루프 중간에만 작은 왕복 꼬리를 붙여 전체 거리만 보정
    """
    if len(current_route) < 4:
        return None

    route = current_route[:]

    MAX_EXTEND_ATTEMPTS = 3

    for _ in range(MAX_EXTEND_ATTEMPTS):
        if time.time() - start_time >= global_timeout:
            break

        cur_len = polyline_length_m(route)
        error_m = target_m - cur_len

        # 이미 충분히 가까우면 종료
        if abs(error_m) <= MAX_LENGTH_ERROR_M:
            return route
        # 이미 target보다 길어졌다면 여기서는 멈춤 (단축 로직은 다른 함수에서 처리)
        if error_m <= 0:
            break

        # 스퍼로 추가할 총 길이 (최대 1.5km)
        extend_len = min(error_m, 1500.0)
        spur_dist = max(150.0, extend_len / 2.0)  # 왕복이므로 절반씩

        # 경로 중간 지점 선택
        idx_mid = len(route) // 2
        p_mid = route[idx_mid]

        # 중간 지점의 진행 방향(대략적인 bearing) 추정
        if 0 < idx_mid < len(route) - 1:
            br_forward = bearing_deg(
                route[idx_mid][0],
                route[idx_mid][1],
                route[idx_mid + 1][0],
                route[idx_mid + 1][1],
            )
        else:
            br_forward = 0.0

        # 진행 방향에 수직(좌/우)인 두 방향으로 스퍼 후보 생성
        candidate_bearings = [
            (br_forward + 90.0) % 360.0,
            (br_forward + 270.0) % 360.0,
        ]

        best_spur: Optional[List[Tuple[float, float]]] = None
        best_spur_len = 0.0

        for br in candidate_bearings:
            spur_point = project_point(
                p_mid[0], p_mid[1], spur_dist, br
            )

            spur_out = kakao_walk_route(p_mid, spur_point)
            spur_back = kakao_walk_route(spur_point, p_mid)

            if not spur_out or not spur_back:
                continue

            spur_path = spur_out + spur_back[1:]
            spur_len = polyline_length_m(spur_path)

            # 스퍼 길이가 너무 짧으면 의미 없음
            if spur_len < 100.0:
                continue

            if spur_len > best_spur_len:
                best_spur_len = spur_len
                best_spur = spur_path

        if not best_spur:
            # 더 이상 유의미한 연장 불가
            break

        # 스퍼를 경로 중간에 삽입
        # route[idx_mid] == p_mid 이므로, 중복을 피하기 위해 스퍼 첫 점은 제외
        route = (
            route[: idx_mid + 1]
            + best_spur[1:]
            + route[idx_mid + 1 :]
        )

    # 마지막으로 target 근처인지 확인
    final_len = polyline_length_m(route)
    if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M:
        return route

    return None


# -----------------------------
# Area Loop 생성 (Two-Segment Hybrid)
# -----------------------------


def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """
    B안: 
    Start → viaA → viaB → Start 구조로
    '두 번 나가고 한 번 돌아오는' 루프를 만든다.

    - Out1: Start → viaA (Valhalla)
    - Out2: viaA → viaB (Valhalla)
    - Back: viaB → Start (Valhalla + Kakao 후보 중 선택)

    기존 메타 구조(meta dict)는 그대로 유지.
    """
    start_time = time.time()

    target_m = max(300.0, km * 1000.0)
    km_requested = km
    start = (lat, lng)

    # 시간 초과 방어
    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
        return [start], {
            "len": 0.0,
            "err": target_m,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": 0,
            "time_s": 0.0,
            "message": "경로 생성 요청이 시작하자마자 시간 제한(10초)을 초과했습니다.",
            "length_ok": False,
            "routes_checked": 0,
            "routes_processed": 0,
            "routes_validated": 0,
        }

    # 목표 거리 기준 이상적인 반지름
    R_ideal = target_m / (2.0 * math.pi)

    # R 후보 (기존 로직 그대로 사용)
    R_MIN = max(100.0, min(R_ideal * 0.3, 200.0))
    R_SMALL = max(200.0, min(R_ideal * 0.6, 400.0))
    R_MEDIUM = max(400.0, min(R_ideal * 1.0, 700.0))
    R_LARGE = max(700.0, min(R_ideal * 1.3, 1100.0))
    R_XLARGE = max(1100.0, min(R_ideal * 1.6, 1800.0))

    radii = list(sorted(list(set([R_MIN, R_SMALL, R_MEDIUM, R_LARGE, R_XLARGE]))))
    bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    candidate_routes: List[Dict[str, Any]] = []
    valhalla_calls = 0
    total_routes_checked = 0

    # ---------------------------------------------------
    # 1. 두 번 나가는 Out(Out1 + Out2) + Back 후보 생성
    #    (기존: Out1 + Back  →  지금: Out1 + Out2 + Back)
    # ---------------------------------------------------
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

            # Out1: Start → viaA
            via_a = project_point(lat, lng, R, br1)
            seg_out1 = valhalla_route(start, via_a)
            valhalla_calls += 1
            if not seg_out1 or len(seg_out1) < 2:
                continue

            pivot = seg_out1[-1]  # Out1 끝점 기준으로 두 번째 Out 시작

            for br2 in bearings:
                if valhalla_calls + 2 > MAX_TOTAL_CALLS:
                    break
                if time.time() - start_time >= GLOBAL_TIMEOUT_S:
                    break

                # Out2: viaA → viaB
                via_b = project_point(pivot[0], pivot[1], R, br2)
                seg_out2 = valhalla_route(pivot, via_b)
                valhalla_calls += 1
                if not seg_out2 or len(seg_out2) < 2:
                    continue

                # 전체 Out = Out1 + Out2 (겹치는 첫 점 제거)
                seg_out = seg_out1 + seg_out2[1:]
                comeback_point = seg_out[-1]

                # Back 후보들: Valhalla + Kakao
                back_segments: List[Dict[str, Any]] = []

                # 1) Valhalla Back (가능하면 1회)
                if valhalla_calls + 1 <= MAX_TOTAL_CALLS:
                    seg_back_v = valhalla_route(comeback_point, start)
                    valhalla_calls += 1
                    if seg_back_v and len(seg_back_v) >= 2:
                        back_segments.append({"seg": seg_back_v, "source": "Valhalla"})

                # 2) Kakao Back
                seg_back_k = kakao_walk_route(comeback_point, start)
                if seg_back_k and len(seg_back_k) >= 2:
                    back_segments.append({"seg": seg_back_k, "source": "Kakao"})

                # Back 후보 각각에 대해 루프 생성 + 점수 계산
                for back_seg_data in back_segments:
                    seg_back = back_seg_data["seg"]

                    # Out, Back 겹치는 정도에 대한 페널티 (기존 로직)
                    overlap_penalty = _calculate_overlap_penalty(seg_out, seg_back)
                    if overlap_penalty > 300.0:
                        continue

                    # 완전한 루프 구성
                    total_route = seg_out + seg_back[1:]
                    if total_route and total_route[0] != start:
                        total_route.insert(0, start)
                    if total_route and total_route[-1] != start:
                        total_route.append(start)

                    # 연속 중복 좌표 제거
                    temp_pts = [total_route[0]]
                    for p in total_route[1:]:
                        if p != temp_pts[-1]:
                            temp_pts.append(p)
                    total_route = temp_pts

                    # 길이/라운드니스 기반 기본 점수
                    score_base, _local_meta = _score_loop(total_route, target_m)
                    total_score = score_base + overlap_penalty

                    if polyline_length_m(total_route) > 0:
                        candidate_routes.append(
                            {
                                "route": total_route,
                                "valhalla_score": total_score,
                            }
                        )
                        total_routes_checked += 1

        if valhalla_calls + 3 > MAX_TOTAL_CALLS:
            break

    # ---------------------------------------------------
    # 2. 후보 경로들 중에서 최종 선택
    #    (카카오 단축은 일단 생략하고, 가장 스코어 좋은/길이 가까운 루프 선택)
    # ---------------------------------------------------
    final_validated_routes: List[Dict[str, Any]] = []
    candidate_routes.sort(key=lambda x: x["valhalla_score"])

    # 우선 상위 N개만 본다 (너무 많으면 시간 초과 위험)
    for i, candidate in enumerate(candidate_routes[:MAX_BEST_ROUTES_TO_TEST]):
        if time.time() - start_time >= GLOBAL_TIMEOUT_S:
            break

        current_route = candidate["route"]
        final_len = polyline_length_m(current_route)

        # 이미 ±99m 이내면 그대로 채택
        if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M:
            final_validated_routes.append(
                {
                    "route": current_route,
                    "score": _score_loop(current_route, target_m)[0],
                }
            )
            continue

        # (여기서 너무 길 때만 _try_shrink_path_kakao를 다시 붙여도 됨.
        #  일단은 단순 버전으로 두고, 나중에 길이 미세조정이 필요하면 그때 추가하자.)

    best_final_route: Optional[List[Tuple[float, float]]] = None

    if final_validated_routes:
        final_validated_routes.sort(key=lambda x: x["score"])
        best_final_route = final_valid_routes[0]["route"]
    elif candidate_routes:
        # ±99m 안에 드는 건 없어도, 그 중에서 가장 거리 오차가 적은 루프 선택
        min_error = float("inf")
        most_adjacent_route = None
        for candidate in candidate_routes:
            length = polyline_length_m(candidate["route"])
            error = abs(length - target_m)
            if error < min_error:
                min_error = error
                most_adjacent_route = candidate["route"]
        best_final_route = most_adjacent_route

    # ---------------------------------------------------
    # 3. 결과 반환 (가능하면 항상 루프 하나는 리턴)
    # ---------------------------------------------------
    if best_final_route:
        final_len = polyline_length_m(best_final_route)
        is_perfect = abs(final_len - target_m) <= MAX_LENGTH_ERROR_M

        meta = {
            "len": final_len,
            "err": abs(final_len - target_m),
            "roundness": _loop_roundness(best_final_route),
            "success": is_perfect,
            "used_fallback": False,
            "valhalla_calls": valhalla_calls,
            "time_s": round(time.time() - start_time, 2),
            "message": "최적의 경로가 도출되었습니다."
            if is_perfect
            else "요청 오차(±99m)를 초과하지만, 가장 인접한 경로를 반환합니다.",
            "length_ok": is_perfect,
            "routes_checked": total_routes_checked,
            "routes_processed": len(candidate_routes),
            "routes_validated": len(final_validated_routes),
        }

        return best_final_route, meta

    # 후보 자체가 하나도 없을 때 (Valhalla 죽었거나 지리적으로 막힌 경우)
    return [start], {
        "len": 0.0,
        "err": target_m,
        "success": False,
        "used_fallback": False,
        "valhalla_calls": valhalla_calls,
        "time_s": round(time.time() - start_time, 2),
        "message": "탐색 결과, 유효한 경로 후보를 찾을 수 없습니다. (Valhalla 통신 불가 또는 지리적 단절)",
        "length_ok": False,
        "routes_checked": total_routes_checked,
        "routes_processed": 0,
        "routes_validated": 0,
    }
