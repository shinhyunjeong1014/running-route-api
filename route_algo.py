import math
import os
import time
import logging
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
    """위도/경도에서 특정 거리(m), 방위각(deg)만큼 직선 이동한 지점을 근사 계산."""
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
    """ 현재는 안전성 기준을 사용하지 않음."""
    return True


def _try_shrink_path_kakao(
    current_route: List[Tuple[float, float]],
    target_m: float,
    valhalla_calls: int,
    start_time: float,
    global_timeout: float,
) -> Tuple[Optional[List[Tuple[float, float]]], int]:
    """
    현재 경로가 '너무 길 때만' 카카오 경로로 단축을 시도한다.
    - 목표보다 짧은 경우에는 여기서 처리하지 않고, 별도의 연장 로직을 사용해야 한다.
    - 성공 조건: 길이가 target_m 이상이면서, target_m + MAX_LENGTH_ERROR_M 이내인 경우만 True.
    """

    current_len = polyline_length_m(current_route)
    error_m = current_len - target_m

    if time.time() - start_time >= global_timeout:
        return None, valhalla_calls

    if error_m <= 0:
        return None, valhalla_calls

    pts = current_route
    idx_a = max(1, int(len(pts) * 0.40))
    idx_b = min(len(pts) - 2, int(len(pts) * 0.60))

    route_to_shrink = current_route[:]

    if idx_a < idx_b:
        p_a = pts[idx_a]
        p_b = pts[idx_b]

        MAX_SHRINK_ATTEMPTS = 3

        for _ in range(MAX_SHRINK_ATTEMPTS):
            if time.time() - start_time >= global_timeout:
                break

            current_len = polyline_length_m(route_to_shrink)
            error_m = current_len - target_m

            if 0.0 <= error_m <= MAX_LENGTH_ERROR_M:
                return route_to_shrink, valhalla_calls

            if error_m < 0:
                break

            reconnect_seg = kakao_walk_route(p_a, p_b)

            if reconnect_seg and len(reconnect_seg) >= 2:
                seg_len_original = polyline_length_m(pts[idx_a : idx_b + 1])
                seg_len_new = polyline_length_m(reconnect_seg)
                reduction = seg_len_original - seg_len_new

                if reduction > 0:
                    new_route = pts[:idx_a] + reconnect_seg + pts[idx_b + 1 :]
                    route_to_shrink = new_route[:]
                else:
                    break

    final_len = polyline_length_m(route_to_shrink)
    final_error = final_len - target_m

    if 0.0 <= final_error <= MAX_LENGTH_ERROR_M:
        return route_to_shrink, valhalla_calls
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
        overlap_ratio = overlap_count / seg_back_len
        return overlap_ratio * 1000.0

    return 0.0


def _extend_path_kakao_spur(
    current_route: List[Tuple[float, float]],
    target_m: float,
    start_time: float,
    global_timeout: float,
) -> List[Tuple[float, float]]:
    """
    현재 루프가 target_m 보다 짧을 때,
    중간 지점 근처에 작은 왕복 스퍼(spur)를 붙여 전체 길이를 늘린다.
    - 카카오 도보 경로만 사용 (Valhalla 호출 수 증가 없음)
    - 모양은 루프 + 작은 돌출부 형태가 되지만, '한 바퀴 돈 느낌'은 유지됨.
    """

    route = current_route[:]
    if len(route) < 3:
        return route

    MAX_EXTEND_ATTEMPTS = 3

    for _ in range(MAX_EXTEND_ATTEMPTS):
        if time.time() - start_time >= global_timeout:
            break

        current_len = polyline_length_m(route)
        if current_len >= target_m:
            break

        remaining = target_m - current_len
        if remaining < 30.0:
            break

        spur_straight = min(remaining + 50.0, 600.0)

        mid_idx = len(route) // 2
        mid_point = route[mid_idx]

        extended = False
        spur_bearings = [90.0, 270.0, 0.0, 180.0]

        for br in spur_bearings:
            if time.time() - start_time >= global_timeout:
                break

            spur_target = project_point(
                mid_point[0], mid_point[1], spur_straight / 2.0, br
            )
            spur_out = kakao_walk_route(mid_point, spur_target)
            spur_back = kakao_walk_route(spur_target, mid_point)

            if spur_out and spur_back and len(spur_out) >= 2 and len(spur_back) >= 2:
                spur_path = spur_out + spur_back[1:]
                new_route = (
                    route[: mid_idx + 1] + spur_path[1:] + route[mid_idx + 1 :]
                )

                cleaned = [new_route[0]]
                for p in new_route[1:]:
                    if p != cleaned[-1]:
                        cleaned.append(p)
                route = cleaned
                extended = True
                break

        if not extended:
            break

    return route


# -----------------------------
# Area Loop 생성 (B안: 거리 기반 Out1/Out2)
# -----------------------------


def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """
    B안:
    Start → Out1 → Out2 → Back → Start 구조.
    Out1, Out2는 '요청 거리 비율'을 기준으로 직접 거리(m)를 설정해 생성하고,
    Back은 최단 복귀(Valhalla/Kakao)를 사용한다.

    - Out1: Start → viaA (거리 기반 타겟 + Valhalla)
    - Out2: viaA → viaB (거리 기반 타겟 + Valhalla)
    - Back: viaB → Start (Valhalla + Kakao 후보 중 선택)
    """

    start_time = time.time()

    target_m = max(300.0, km * 1000.0)
    start = (lat, lng)

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

    # 거리 비율 조합 (Out1, Out2)
    # 합이 0.8~0.9 정도가 되도록 설정 (나머지는 Back으로 채움)
    frac_pairs = [
        (0.35, 0.35),
        (0.4, 0.35),
        (0.35, 0.4),
        (0.4, 0.4),
    ]

    bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    candidate_routes: List[Dict[str, Any]] = []
    valhalla_calls = 0
    total_routes_checked = 0

    # ---------------------------------------------------
    # 1. 거리 기반 Out1 + Out2 + Back 후보 생성
    # ---------------------------------------------------
    for (out1_frac, out2_frac) in frac_pairs:
        if time.time() - start_time >= GLOBAL_TIMEOUT_S:
            break

        out1_target = target_m * out1_frac
        out2_target = target_m * out2_frac

        for br1 in bearings:
            if time.time() - start_time >= GLOBAL_TIMEOUT_S:
                break
            if valhalla_calls + 2 > MAX_TOTAL_CALLS:
                break

            # Out1: Start → viaA (거리 기반 타겟)
            via_a_geo = project_point(lat, lng, out1_target, br1)
            seg_out1 = valhalla_route(start, via_a_geo)
            valhalla_calls += 1

            if not seg_out1 or len(seg_out1) < 2:
                continue

            pivot = seg_out1[-1]

            for br2 in bearings:
                if time.time() - start_time >= GLOBAL_TIMEOUT_S:
                    break
                if valhalla_calls + 2 > MAX_TOTAL_CALLS:
                    break

                # Out2: viaA → viaB (거리 기반 타겟)
                via_b_geo = project_point(pivot[0], pivot[1], out2_target, br2)
                seg_out2 = valhalla_route(pivot, via_b_geo)
                valhalla_calls += 1

                if not seg_out2 or len(seg_out2) < 2:
                    continue

                seg_out = seg_out1 + seg_out2[1:]
                comeback_point = seg_out[-1]

                # Out 구간이 너무 짧으면 (목표 대비 50% 미만) 무시
                out_len = polyline_length_m(seg_out)
                if out_len < target_m * 0.4:
                    continue

                back_segments: List[Dict[str, Any]] = []

                # 1) Valhalla Back
                if valhalla_calls + 1 <= MAX_TOTAL_CALLS:
                    seg_back_v = valhalla_route(comeback_point, start)
                    valhalla_calls += 1
                    if seg_back_v and len(seg_back_v) >= 2:
                        back_segments.append(
                            {"seg": seg_back_v, "source": "Valhalla"}
                        )

                # 2) Kakao Back
                seg_back_k = kakao_walk_route(comeback_point, start)
                if seg_back_k and len(seg_back_k) >= 2:
                    back_segments.append(
                        {"seg": seg_back_k, "source": "Kakao"}
                    )

                for back_seg_data in back_segments:
                    seg_back = back_seg_data["seg"]

                    overlap_penalty = _calculate_overlap_penalty(seg_out, seg_back)
                    if overlap_penalty > 300.0:
                        continue

                    total_route = seg_out + seg_back[1:]
                    if total_route and total_route[0] != start:
                        total_route.insert(0, start)
                    if total_route and total_route[-1] != start:
                        total_route.append(start)

                    # 너무 짧은 루프는 후보에서 제외 (예: target의 50% 미만)
                    total_len = polyline_length_m(total_route)
                    if total_len < target_m * 0.5:
                        continue

                    cleaned = [total_route[0]]
                    for p in total_route[1:]:
                        if p != cleaned[-1]:
                            cleaned.append(p)
                    total_route = cleaned

                    score_base, _local_meta = _score_loop(total_route, target_m)
                    total_score = score_base + overlap_penalty

                    if total_len > 0:
                        candidate_routes.append(
                            {
                                "route": total_route,
                                "valhalla_score": total_score,
                            }
                        )
                        total_routes_checked += 1

        if valhalla_calls >= MAX_TOTAL_CALLS:
            break

    # ---------------------------------------------------
    # 2. 후보 경로들 중에서 최종 선택 (단축/연장 포함)
    # ---------------------------------------------------
    final_validated_routes: List[Dict[str, Any]] = []
    candidate_routes.sort(key=lambda x: x["valhalla_score"])

    for i, candidate in enumerate(candidate_routes[:MAX_BEST_ROUTES_TO_TEST]):

        if time.time() - start_time >= GLOBAL_TIMEOUT_S:
            break

        base_route = candidate["route"]
        route_variant = base_route
        final_len = polyline_length_m(route_variant)
        error_m = final_len - target_m

        # 1) 이미 조건 만족: target_m 이상이면서 +99m 이내
        if 0.0 <= error_m <= MAX_LENGTH_ERROR_M:
            final_validated_routes.append(
                {
                    "route": route_variant,
                    "score": _score_loop(route_variant, target_m)[0],
                }
            )
            continue

        # 2) 너무 긴 경우 → 카카오 단축 시도
        if error_m > MAX_LENGTH_ERROR_M:
            shrunken_route, valhalla_calls = _try_shrink_path_kakao(
                route_variant,
                target_m,
                valhalla_calls,
                start_time,
                GLOBAL_TIMEOUT_S,
            )
            if shrunken_route:
                final_len = polyline_length_m(shrunken_route)
                error_m = final_len - target_m
                if 0.0 <= error_m <= MAX_LENGTH_ERROR_M:
                    final_score = _score_loop(shrunken_route, target_m)[0]
                    final_validated_routes.append(
                        {"route": shrunken_route, "score": final_score}
                    )
            continue

        # 3) 아직 짧은 경우 → 스퍼로 연장 시도
        if error_m < 0.0:
            extended_route = _extend_path_kakao_spur(
                route_variant, target_m, start_time, GLOBAL_TIMEOUT_S
            )
            final_len = polyline_length_m(extended_route)
            error_m = final_len - target_m

            # 3-1) 연장 후 바로 조건 만족
            if 0.0 <= error_m <= MAX_LENGTH_ERROR_M:
                final_score = _score_loop(extended_route, target_m)[0]
                final_validated_routes.append(
                    {"route": extended_route, "score": final_score}
                )
                continue

            # 3-2) 연장 후 너무 길어졌으면 다시 단축
            if error_m > MAX_LENGTH_ERROR_M:
                shrunken_route, valhalla_calls = _try_shrink_path_kakao(
                    extended_route,
                    target_m,
                    valhalla_calls,
                    start_time,
                    GLOBAL_TIMEOUT_S,
                )
                if shrunken_route:
                    final_len = polyline_length_m(shrunken_route)
                    error_m = final_len - target_m
                    if 0.0 <= error_m <= MAX_LENGTH_ERROR_M:
                        final_score = _score_loop(shrunken_route, target_m)[0]
                        final_validated_routes.append(
                            {"route": shrunken_route, "score": final_score}
                        )
            continue

    best_final_route: Optional[List[Tuple[float, float]]] = None

    if final_validated_routes:
        final_validated_routes.sort(key=lambda x: x["score"])
        best_final_route = final_validated_routes[0]["route"]
    elif candidate_routes:
        # 조건을 만족하는 후보는 없지만, 그 중에서 가장 target에 가까운 것을 선택
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
    # 3. 결과 반환
    # ---------------------------------------------------
    if best_final_route:
        final_len = polyline_length_m(best_final_route)
        is_perfect = (final_len >= target_m) and (
            final_len <= target_m + MAX_LENGTH_ERROR_M
        )

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

    # 후보 자체가 하나도 없을 때
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
