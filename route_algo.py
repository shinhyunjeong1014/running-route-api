"""
route_algo.py

러닝/워킹용 루프 경로를 생성하는 모듈.

핵심 아이디어
--------------
1) Valhalla 보행자 라우팅을 사용해 "편도 경로(start → anchor)"와
   "복귀 경로(anchor → start)"를 만든 뒤, 이를 이어서 루프를 만든다.

2) 목표 거리 km 에 맞추기 위해,
   - 한 개의 방향(0~360도)을 선택하고
   - 그 방향으로 anchor 의 직선 거리 r 을 조정(이분 탐색)하면서
   - 루프 길이 |route_out| + |route_back| 이 target_m 에 최대한 근접하도록 한다.

3) 여러 방향(기본 8개)에서 위 과정을 수행한 뒤,
   목표 거리와의 오차가 가장 작은 루프를 선택한다.

4) 기존 "면적 탐색 기반 루프 생성 방식(Strategy A)"도 남겨두고,
   새 "거리 제어 엔진 기반 루프 생성(Strategy B)"이 실패할 때
   최후의 보완책으로 사용할 수 있게 한다.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import requests

# -----------------------------
# 환경 설정 및 상수
# -----------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] route_algo: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

EARTH_RADIUS_M = 6371000.0

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002")  # 예: http://valhalla:8002
KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY")  # 필요 시 보행자 대체 라우팅에 사용

# 루프 거리 오차 허용 범위 (±)
LENGTH_TOLERANCE_M = 99.0

# Valhalla 호출 상한 – infra 보호용
MAX_VALHALLA_CALLS = 30


# -----------------------------
# 기본 지오메트리 유틸
# -----------------------------

def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 위경도 사이의 거리를 미터 단위로 계산."""
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    r_lat1 = math.radians(lat1)
    r_lat2 = math.radians(lat2)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def polyline_length_m(polyline: List[Tuple[float, float]]) -> float:
    """polyline 의 총 길이를 미터 단위로 계산."""
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lng1), (lat2, lng2) in zip(polyline[:-1], polyline[1:]):
        total += haversine_m(lat1, lng1, lat2, lng2)
    return total


def offset_coord(
    lat: float, lng: float, bearing_deg: float, distance_m: float
) -> Tuple[float, float]:
    """
    시작점(lat, lng)에서 bearing_deg 방향으로 distance_m 만큼 이동한 위경도 좌표를 계산.
    (단순한 구면 삼각법 활용)
    """
    if distance_m <= 0:
        return lat, lng

    br = math.radians(bearing_deg)
    d_r = distance_m / EARTH_RADIUS_M

    lat1 = math.radians(lat)
    lng1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_r)
        + math.cos(lat1) * math.sin(d_r) * math.cos(br)
    )
    lng2 = lng1 + math.atan2(
        math.sin(br) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lng2)


def compute_roundness(polyline: List[Tuple[float, float]]) -> float:
    """
    루프의 '둥근 정도'를 대략적으로 평가.
    - 첫/마지막 점이 가까울수록,
    - 전체 길이에 비해 둘레(볼록 껍질)가 짧을수록 둥글다고 판단.
    0 ~ 1 범위 값 (1에 가까울수록 좋음) 정도를 목표로 한다.
    """
    if len(polyline) < 3:
        return 0.0

    # 시작점과 끝점 간 거리
    start = polyline[0]
    end = polyline[-1]
    close_dist = haversine_m(start[0], start[1], end[0], end[1])

    total_len = polyline_length_m(polyline) + 1e-6

    # "시작-끝 거리 / 전체 길이" 가 작을수록 루프가 잘 닫히므로
    closure_score = max(0.0, 1.0 - close_dist / max(200.0, total_len))

    # TODO: 필요 시 convex hull 기반 둘레 비율도 추가 가능
    return closure_score


# -----------------------------
# Valhalla / Kakao 라우팅
# -----------------------------

def _decode_polyline(shape: str) -> List[Tuple[float, float]]:
    """
    Valhalla polyline(shape) 디코더.
    기본 precision=1e6 을 가정.
    """
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
            # sanity check
            if not (-90.0 <= current_lat <= 90.0 and -180.0 <= current_lng <= 180.0):
                logger.error(
                    "[Valhalla Decode] Sanity check failed: (%f, %f)",
                    current_lat,
                    current_lng,
                )
                return []
            coords.append((current_lat, current_lng))
    except IndexError:
        logger.error("[Valhalla Decode] Unexpected end of polyline string.")
        return []

    return coords


def _valhalla_walk_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    timeouts: Tuple[float, float] = (2.0, 5.0),
) -> Optional[List[Tuple[float, float]]]:
    """
    Valhalla 에 보행자 경로(start → end)를 요청하고 polyline 리스트를 반환.
    실패 시 None.
    """
    url = VALHALLA_URL.rstrip("/") + "/route"

    data = {
        "locations": [
            {"lat": float(start[0]), "lon": float(start[1])},
            {"lat": float(end[0]), "lon": float(end[1])},
        ],
        "costing": "pedestrian",
        "directions_options": {"units": "kilometers"},
    }

    try:
        resp = requests.post(
            url,
            data=json.dumps(data),
            timeout=timeouts,
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        logger.error("[Valhalla] Request failed (network): %s", e)
        return None

    if resp.status_code != 200:
        logger.error(
            "[Valhalla] Non-200 status: %s - %s", resp.status_code, resp.text[:200]
        )
        return None

    try:
        js = resp.json()
    except Exception as e:
        logger.error("[Valhalla] JSON decode error: %s", e)
        return None

    try:
        legs = js.get("trip", {}).get("legs", [])
        if not legs:
            logger.error("[Valhalla] No legs in response.")
            return None
        polyline: List[Tuple[float, float]] = []
        for leg in legs:
            shape = leg.get("shape")
            if not shape:
                continue
            coords = _decode_polyline(shape)
            if coords:
                if polyline and polyline[-1] == coords[0]:
                    polyline.extend(coords[1:])
                else:
                    polyline.extend(coords)
        if len(polyline) < 2:
            logger.error("[Valhalla] Decoded polyline too short.")
            return None
        return polyline
    except Exception as e:
        logger.error("[Valhalla] Parsing error: %s", e)
        return None


def _kakao_walk_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    timeouts: Tuple[float, float] = (2.0, 5.0),
) -> Optional[List[Tuple[float, float]]]:
    """
    Kakao 보행자 경로 API (선택적, Valhalla 장애 시 fallback 용).
    실제 서비스에서는 Kakao developers 문서 기준으로 URL/파라미터를 맞춰야 한다.
    """
    if not KAKAO_API_KEY:
        return None

    # 아래 URL/파라미터는 예시. 실제 Kakao API 스펙에 맞게 수정 필요.
    url = "https://apis-navi.kakaomobility.com/v1/directions/walking"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {
        "origin": f"{start[1]},{start[0]}",
        "destination": f"{end[1]},{end[0]}",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeouts)
    except Exception as e:
        logger.error("[Kakao] Request failed: %s", e)
        return None

    if resp.status_code != 200:
        logger.error("[Kakao] Non-200 status: %s - %s", resp.status_code, resp.text[:200])
        return None

    try:
        js = resp.json()
    except Exception as e:
        logger.error("[Kakao] JSON decode error: %s", e)
        return None

    try:
        # Kakao 응답 포맷에 맞게 polyline 추출 (예시)
        routes = js.get("routes", [])
        if not routes:
            return None
        first = routes[0]
        sections = first.get("sections", [])
        coords: List[Tuple[float, float]] = []
        for sec in sections:
            for p in sec.get("roads", []):
                for vertex in p.get("vertexes", []):
                    # 실제 포맷에 맞게 lat/lng 추출 로직 조정 필요
                    pass  # 이 부분은 실제 사용 시 구현
        # 여기서는 Kakao fallback 을 아직 사용하지 않도록 None 반환
        return None
    except Exception as e:
        logger.error("[Kakao] Parsing error: %s", e)
        return None


def _route_between(
    start: Tuple[float, float],
    end: Tuple[float, float],
    prefer: str = "valhalla",
    timeouts: Tuple[float, float] = (2.0, 5.0),
) -> Optional[List[Tuple[float, float]]]:
    """
    start → end 경로를 얻기 위한 공통 라우터.
    - 우선 Valhalla 를 사용
    - 실패 시 Kakao 보행자 API 를 fallback 으로 시도 (현재는 미구현 상태)
    """
    if prefer == "valhalla":
        poly = _valhalla_walk_route(start, end, timeouts=timeouts)
        if poly:
            return poly
        # fallback (현재는 항상 None)
        poly = _kakao_walk_route(start, end, timeouts=timeouts)
        return poly
    else:
        # 다른 전략이 필요하면 여기 확장
        return _valhalla_walk_route(start, end, timeouts=timeouts)


# -----------------------------
# Strategy B: 거리 제어 엔진 기반 루프 생성
# -----------------------------

def _build_loop_via_anchor(
    start: Tuple[float, float],
    anchor: Tuple[float, float],
    valhalla_budget: Dict[str, int],
) -> Optional[List[Tuple[float, float]]]:
    """
    anchor 를 경유하는 루프:
        start → anchor, anchor → start
    를 만들고 polyline 으로 병합.
    """
    if valhalla_budget["used"] >= MAX_VALHALLA_CALLS:
        return None

    out_poly = _route_between(start, anchor)
    if not out_poly:
        valhalla_budget["used"] += 1  # 실패도 1회로 센다 (대략적인 카운트용)
        return None
    valhalla_budget["used"] += 1

    if valhalla_budget["used"] >= MAX_VALHALLA_CALLS:
        return None
    back_poly = _route_between(anchor, start)
    if not back_poly:
        valhalla_budget["used"] += 1
        return None
    valhalla_budget["used"] += 1

    # 병합 (중복되는 모서리 제거)
    loop_poly: List[Tuple[float, float]] = list(out_poly)
    if loop_poly and back_poly:
        if loop_poly[-1] == back_poly[0]:
            loop_poly.extend(back_poly[1:])
        else:
            loop_poly.extend(back_poly)
    return loop_poly


def _generate_loop_distance_control(
    start_lat: float,
    start_lng: float,
    km: float,
    *,
    directions: int = 8,
    binary_search_steps: int = 3,
) -> Tuple[Optional[List[Tuple[float, float]]], Dict]:
    """
    Strategy B: 목표 거리 km 에 맞는 루프를 생성하는 거리 제어 엔진.

    - 여러 방향(예: 0, 45, 90, ..., 315도)에 대해
      anchor 의 직선 거리 r 을 이분 탐색하여
      루프 길이가 target_m 에 가깝도록 맞춘다.
    - 각 방향마다 best candidate 를 모은 뒤, 전체 중 가장 오차가 작은 루프를 선택.
    """
    start = (float(start_lat), float(start_lng))
    target_m = float(km) * 1000.0

    # 너무 작은 거리는 Valhalla 결과도 불안정하므로 최소 500m 정도로 클램핑
    target_m = max(target_m, 500.0)

    valhalla_budget = {"used": 0}

    best_poly: Optional[List[Tuple[float, float]]] = None
    best_err = float("inf")
    best_roundness = 0.0
    loops_checked = 0

    # anchor 거리의 대략적인 범위 설정
    #    - 루프 길이 ≈ 2 * (start~anchor 실제 경로 길이) 라고 가정하면
    #      anchor 까지의 직선거리는 대략 target_m / 4 ~ target_m / 2 정도에서 결정된다.
    min_anchor_r = max(150.0, target_m * 0.20)  # 20% 정도
    max_anchor_r = min(2500.0, target_m * 0.60)  # 너무 멀리 나가지 않도록 상한

    bearings = [360.0 * i / directions for i in range(directions)]

    for bearing in bearings:
        # 각 방향마다 이분 탐색
        low_r = min_anchor_r
        high_r = max_anchor_r

        dir_best_poly: Optional[List[Tuple[float, float]]] = None
        dir_best_err = float("inf")
        dir_best_round = 0.0

        for _ in range(binary_search_steps):
            if valhalla_budget["used"] >= MAX_VALHALLA_CALLS:
                break

            mid_r = (low_r + high_r) / 2.0
            anchor = offset_coord(start[0], start[1], bearing, mid_r)

            loop_poly = _build_loop_via_anchor(start, anchor, valhalla_budget)
            if not loop_poly:
                # 이 방향/거리 조합 실패 → anchor 를 좀 당겨보거나 멀려보게 조정
                high_r = (low_r + high_r) / 2.0
            else:
                loops_checked += 1
                loop_len = polyline_length_m(loop_poly)
                err = abs(loop_len - target_m)
                roundness = compute_roundness(loop_poly)

                # 방향별 best 업데이트
                if err < dir_best_err or (
                    abs(err - dir_best_err) < 30.0 and roundness > dir_best_round
                ):
                    dir_best_poly = loop_poly
                    dir_best_err = err
                    dir_best_round = roundness

                # 전체 best 업데이트
                if err < best_err or (
                    abs(err - best_err) < 30.0 and roundness > best_roundness
                ):
                    best_poly = loop_poly
                    best_err = err
                    best_roundness = roundness

                # 이분 탐색 갱신
                if loop_len < target_m:
                    # 루프 길이가 짧으니 anchor 를 더 멀리
                    low_r = mid_r
                else:
                    # 길이가 너무 길면 anchor 를 조금 당김
                    high_r = mid_r

        # 각 방향별 best 가 target 에 비해 너무 짧으면,
        # 마지막으로 anchor 를 high_r 쪽으로 한 번 더 시도해볼 수 있음
        if (
            valhalla_budget["used"] < MAX_VALHALLA_CALLS
            and dir_best_poly is not None
            and dir_best_err > LENGTH_TOLERANCE_M
        ):
            # high_r 근처에서 한 번 더 anchor 를 잡고 시도
            anchor = offset_coord(start[0], start[1], bearing, high_r)
            loop_poly = _build_loop_via_anchor(start, anchor, valhalla_budget)
            if loop_poly:
                loops_checked += 1
                loop_len = polyline_length_m(loop_poly)
                err = abs(loop_len - target_m)
                roundness = compute_roundness(loop_poly)
                if err < best_err or (
                    abs(err - best_err) < 30.0 and roundness > best_roundness
                ):
                    best_poly = loop_poly
                    best_err = err
                    best_roundness = roundness

    meta = {
        "km_requested": float(km),
        "target_m": target_m,
        "len": float(polyline_length_m(best_poly)) if best_poly else 0.0,
        "err": float(best_err if best_poly else float("inf")),
        "roundness": float(best_roundness),
        "length_ok": bool(best_poly and best_err <= LENGTH_TOLERANCE_M),
        "success": bool(best_poly is not None),
        "routes_checked": int(loops_checked),
        "routes_processed": int(loops_checked),
        "routes_validated": int(loops_checked if best_poly else 0),
        "valhalla_calls": int(valhalla_budget["used"]),
        "used_fallback": False,
        "time_s": 0.0,  # 상위 wrapper 에서 세팅
        "message": "",
        "strategy": "distance_control",
    }

    if not best_poly:
        meta["message"] = (
            "거리 제어 엔진(Strategy B)으로 유효한 루프를 찾지 못했습니다."
        )
    elif not meta["length_ok"]:
        meta["message"] = (
            "거리 제어 엔진으로 생성한 루프가 요청 오차(±%.0fm)를 만족하지 못했습니다."
            % LENGTH_TOLERANCE_M
        )
    else:
        meta["message"] = "거리 제어 엔진(Strategy B)으로 생성한 루프입니다."

    return best_poly, meta


# -----------------------------
# Strategy A: 기존 면적 기반 루프 (간소화 버전)
# -----------------------------

def _generate_loop_area_simple(
    start_lat: float,
    start_lng: float,
    km: float,
) -> Tuple[Optional[List[Tuple[float, float]]], Dict]:
    """
    Strategy A: 기존 방식과 유사하지만, 구현을 단순화한 영역 탐색 루프.
    - 여러 반경 R, 여러 방향에 대해 anchor 를 찍고
    - start → anchor → start 루프를 만들어,
      거리 오차 + 둥근 정도로 스코어링하여 best 를 선택한다.
    - 거리 제어 엔진이 완전히 실패할 때 마지막 fallback 용도로만 사용.
    """
    start = (float(start_lat), float(start_lng))
    target_m = float(km) * 1000.0
    target_m = max(target_m, 500.0)

    valhalla_budget = {"used": 0}

    best_poly: Optional[List[Tuple[float, float]]] = None
    best_err = float("inf")
    best_round = 0.0
    loops_checked = 0

    # 예전 코드에서 사용하던 개념을 간략화:
    #   - 이상적인 반경 R ≈ target_m / (2π)
    R_ideal = target_m / (2.0 * math.pi)
    radii = [
        max(100.0, R_ideal * r_scale)
        for r_scale in [0.6, 0.8, 1.0, 1.2, 1.5]
    ]
    radii = [min(r, 3000.0) for r in radii]

    bearings = [i * 30.0 for i in range(12)]  # 0, 30, 60, ..., 330

    for R in radii:
        for bearing in bearings:
            if valhalla_budget["used"] >= MAX_VALHALLA_CALLS:
                break

            anchor = offset_coord(start[0], start[1], bearing, R)
            loop_poly = _build_loop_via_anchor(start, anchor, valhalla_budget)
            if not loop_poly:
                continue

            loops_checked += 1
            loop_len = polyline_length_m(loop_poly)
            err = abs(loop_len - target_m)
            roundness = compute_roundness(loop_poly)

            # 간단한 스코어: err 를 우선, 비슷하면 roundness 가 높은 쪽
            if (
                err < best_err
                or (abs(err - best_err) < 30.0 and roundness > best_round)
            ):
                best_poly = loop_poly
                best_err = err
                best_round = roundness

    meta = {
        "km_requested": float(km),
        "target_m": target_m,
        "len": float(polyline_length_m(best_poly)) if best_poly else 0.0,
        "err": float(best_err if best_poly else float("inf")),
        "roundness": float(best_round),
        "length_ok": bool(best_poly and best_err <= LENGTH_TOLERANCE_M),
        "success": bool(best_poly is not None),
        "routes_checked": int(loops_checked),
        "routes_processed": int(loops_checked),
        "routes_validated": int(loops_checked if best_poly else 0),
        "valhalla_calls": int(valhalla_budget["used"]),
        "used_fallback": False,
        "time_s": 0.0,  # 상위 wrapper 에서 세팅
        "message": "",
        "strategy": "area_simple",
    }

    if not best_poly:
        meta["message"] = "면적 기반 탐색(Strategy A)으로도 루프를 찾지 못했습니다."
    elif not meta["length_ok"]:
        meta["message"] = (
            "면적 기반 탐색(Strategy A)으로 생성한 루프가 요청 오차(±%.0fm)를 만족하지 못했습니다."
            % LENGTH_TOLERANCE_M
        )
    else:
        meta["message"] = "면적 기반 탐색(Strategy A)으로 생성한 루프입니다."

    return best_poly, meta


# -----------------------------
# 외부에서 호출하는 메인 함수
# -----------------------------

def generate_area_loop_route(
    start_lat: float,
    start_lng: float,
    km: float,
) -> Tuple[Optional[List[Tuple[float, float]]], Dict]:
    """
    외부에서 사용하는 메인 엔트리 포인트.

    1) 우선 "거리 제어 엔진 기반 루프(Strategy B)"로 시도
    2) 실패하거나, 유효한 루프를 찾지 못한 경우
       - 보완 차원에서 "면적 기반 루프(Strategy A)"를 시도
    3) 두 전략 중, 목표 거리와의 오차가 더 작은 쪽을 최종 선택
    """
    t0 = time.time()

    # Strategy B 우선
    poly_b, meta_b = _generate_loop_distance_control(start_lat, start_lng, km)
    meta_b["time_s"] = time.time() - t0

    # B가 완전히 실패하지 않았고, 오차도 어느 정도(예: 2 * tolerance) 이하라면 그대로 사용
    if poly_b and meta_b["err"] <= max(LENGTH_TOLERANCE_M * 2.0, 250.0):
        return poly_b, meta_b

    # 아니면 Strategy A 로 보완 시도
    t1 = time.time()
    poly_a, meta_a = _generate_loop_area_simple(start_lat, start_lng, km)
    meta_a["time_s"] = time.time() - t1

    # 두 결과 중 더 좋은 것을 선택
    candidates = []
    if poly_b:
        candidates.append(("B", poly_b, meta_b))
    if poly_a:
        candidates.append(("A", poly_a, meta_a))

    if not candidates:
        # 완전히 실패한 경우 – B의 meta 를 기준으로 실패 정보만 반환
        fail_meta = meta_b
        fail_meta["success"] = False
        fail_meta["message"] = (
            "거리 제어 엔진(B), 면적 기반(A) 모두 유효한 루프를 찾지 못했습니다."
        )
        return None, fail_meta

    # err 기준으로 최소값을 선택, 비슷하면 roundness 큰 쪽
    best_name, best_poly, best_meta = None, None, None
    for name, poly, meta in candidates:
        if best_meta is None:
            best_name, best_poly, best_meta = name, poly, meta
            continue
        if meta["err"] < best_meta["err"] or (
            abs(meta["err"] - best_meta["err"]) < 30.0
            and meta["roundness"] > best_meta["roundness"]
        ):
            best_name, best_poly, best_meta = name, poly, meta

    # 최종 meta 에 어떤 전략이 쓰였는지 표시
    best_meta = dict(best_meta)  # copy
    best_meta["final_strategy"] = best_name

    # 전체 수행 시간 (A, B 모두 포함)
    best_meta["time_s_total"] = time.time() - t0

    # 오류 메시지/성공 메시지 보정
    if best_meta["err"] <= LENGTH_TOLERANCE_M:
        best_meta["length_ok"] = True
        if best_name == "B":
            best_meta["message"] = (
                "거리 제어 엔진(Strategy B)으로 요청 거리(±%.0fm)를 만족하는 루프를 생성했습니다."
                % LENGTH_TOLERANCE_M
            )
        else:
            best_meta["message"] = (
                "면적 기반 탐색(Strategy A)으로 요청 거리(±%.0fm)를 만족하는 루프를 생성했습니다."
                % LENGTH_TOLERANCE_M
            )
    else:
        best_meta["length_ok"] = False
        best_meta["message"] = (
            "요청 오차(±%.0fm)를 만족하지는 않지만, 두 전략 중 가장 인접한 루프를 반환합니다."
            % LENGTH_TOLERANCE_M
        )

    return best_poly, best_meta


if __name__ == "__main__":
    # 간단 수동 테스트용
    lat, lng = 37.44847, 126.64964
    for km in [1, 3, 5]:
        poly, meta = generate_area_loop_route(lat, lng, km)
        print(f"=== km={km} ===")
        print("len:", meta.get("len"), "err:", meta.get("err"))
        print("strategy:", meta.get("final_strategy", meta.get("strategy")))
        print("message:", meta.get("message"))
        print("points:", len(poly) if poly else 0)
        print()
