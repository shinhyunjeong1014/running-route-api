# ============================
# route_algo.py — Part 1/4
# ============================

import math
import os
import logging
from typing import List, Tuple, Dict, Optional

import requests

# ----------------------------
# 설정
# ----------------------------

# Valhalla HTTP 엔드포인트
VALHALLA_ROUTE_URL = os.getenv("VALHALLA_ROUTE_URL", "http://localhost:8002/route")

# 러닝 루프 설정
DEFAULT_COSTING = "pedestrian"
EARTH_RADIUS_M = 6371000.0  # meters


# ----------------------------
# 기본 지오메트리 유틸
# ----------------------------

LatLng = Tuple[float, float]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 위경도 좌표 사이의 거리 (meter) 반환
    """
    r = EARTH_RADIUS_M
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _polyline_length_m(poly: List[LatLng]) -> float:
    """
    폴리라인 전체 길이(m) 계산
    """
    if len(poly) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(poly) - 1):
        lat1, lon1 = poly[i]
        lat2, lon2 = poly[i + 1]
        dist += _haversine_m(lat1, lon1, lat2, lon2)
    return dist


def _to_local_xy(points: List[LatLng]) -> List[Tuple[float, float]]:
    """
    위경도 좌표들을 첫 번째 점 기준의 local 평면 좌표 (x, y, meter)로 변환.
    스파이크 감지용 각도 계산에 사용.
    """
    if not points:
        return []

    lat0, lon0 = points[0]
    lat0_rad = math.radians(lat0)

    res = []
    for lat, lon in points:
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        x = EARTH_RADIUS_M * dlon * math.cos(lat0_rad)
        y = EARTH_RADIUS_M * dlat
        res.append((x, y))

    return res
# ============================
# route_algo.py — Part 2/4
# ============================

# ----------------------------
# Polyline6 디코딩 (Valhalla shape)
# ----------------------------

def _decode_polyline6(encoded: str) -> List[LatLng]:
    """
    Valhalla의 polyline6 포맷 디코딩.
    (Google polyline과 유사하지만 정밀도 1e-6)
    """
    if not encoded:
        return []

    coordinates: List[LatLng] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)

    while index < length:
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
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

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
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / 1e6, lng / 1e6))

    return coordinates


# ----------------------------
# Valhalla 호출 헬퍼
# ----------------------------

def _call_valhalla_route(
    start: LatLng,
    end: LatLng,
    costing: str = DEFAULT_COSTING,
    timeout: float = 5.0
) -> Optional[List[LatLng]]:
    """
    Valhalla /route 호출 → shape(폴리라인) 반환
    실패 시 None
    """
    lat1, lon1 = start
    lat2, lon2 = end

    payload = {
        "locations": [
            {"lat": lat1, "lon": lon1},
            {"lat": lat2, "lon": lon2},
        ],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }

    try:
        resp = requests.post(VALHALLA_ROUTE_URL, json=payload, timeout=timeout)
    except Exception as e:
        logging.warning(f"Valhalla request exception: {e}")
        return None

    if not resp.ok:
        logging.warning(f"Valhalla HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        data = resp.json()
    except ValueError:
        logging.warning("Valhalla response is not JSON")
        return None

    trip = data.get("trip")
    if not trip:
        logging.warning("Valhalla response has no 'trip'")
        return None

    shape = trip.get("shape")
    if isinstance(shape, str):
        # polyline6 인코딩
        poly = _decode_polyline6(shape)
    elif isinstance(shape, list):
        # 이미 좌표 리스트 형태
        poly = [(pt["lat"], pt["lon"]) for pt in shape]
    else:
        poly = []

    if not poly:
        logging.warning("Valhalla returned empty polyline")
        return None

    return poly


def _merge_out_and_back(out_poly: List[LatLng], back_poly: List[LatLng]) -> List[LatLng]:
    """
    start → pivot, pivot → start 경로를 하나의 루프 폴리라인으로 병합.
    back_poly의 첫 점(=pivot) 중복 제거.
    """
    if not out_poly:
        return back_poly[:] if back_poly else []
    if not back_poly:
        return out_poly[:]

    merged = out_poly[:]
    # back_poly 첫 점이 out_poly 끝 점과 같다면 중복 제거
    if back_poly and merged[-1] == back_poly[0]:
        merged.extend(back_poly[1:])
    else:
        merged.extend(back_poly)

    return merged
    
# ============================
# route_algo.py — Part 3/4
# ============================

# ----------------------------
# Pivot 후보 생성
# ----------------------------

def _generate_pivot_candidates(
    start: LatLng,
    target_m: float,
    n_rings: int = 3,
    n_bearings: int = 12
) -> List[LatLng]:
    """
    start 기준으로 원형 링 위에 pivot 후보들을 생성.
    target_m의 절반 근처에서 ring을 여러 개 생성해서
    start→pivot→start 길이가 target과 비슷해지도록 유도.
    """
    lat0, lon0 = start
    lat0_rad = math.radians(lat0)

    # 대략적으로 out + back = target => out ≈ target/2
    base_radius = max(target_m * 0.45, 200.0)  # 최소 200m는 나가도록

    pivots: List[LatLng] = []
    for ring in range(n_rings):
        radius = base_radius * (0.7 + 0.2 * ring)  # 0.7, 0.9, 1.1 배
        for k in range(n_bearings):
            theta = 2.0 * math.pi * (k / n_bearings)
            # 위경도 오프셋
            dlat = (radius / EARTH_RADIUS_M) * math.cos(theta)
            dlon = (radius / (EARTH_RADIUS_M * math.cos(lat0_rad))) * math.sin(theta)
            plat = lat0 + math.degrees(dlat)
            plon = lon0 + math.degrees(dlon)
            pivots.append((plat, plon))

    return pivots


# ----------------------------
# 루프 품질 평가
# ----------------------------

def _compute_shape_jaggedness(poly: List[LatLng]) -> float:
    """
    폴리라인의 '지그재그 정도'를 [0, 1] 범위로 반환.
    0에 가까울수록 부드러운 경로, 1에 가까울수록 꺾임이 심함.
    """
    n = len(poly)
    if n < 3:
        return 0.0

    xy = _to_local_xy(poly)
    total_angle = 0.0
    count = 0

    for i in range(1, n - 1):
        x0, y0 = xy[i - 1]
        x1, y1 = xy[i]
        x2, y2 = xy[i + 1]

        v1 = (x0 - x1, y0 - y1)
        v2 = (x2 - x1, y2 - y1)

        norm1 = math.hypot(*v1)
        norm2 = math.hypot(*v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
        dot = max(-1.0, min(1.0, dot))
        angle = math.degrees(math.acos(dot))  # 0 = straight, 180 = U-turn
        # 직선(0°)에서 멀어질수록 더 벌점
        total_angle += angle
        count += 1

    if count == 0:
        return 0.0

    avg_angle = total_angle / count  # 0~180
    # 0°일 때 0, 90°일 때 0.5, 180°일 때 1.0 쯤 되도록 정규화
    jag = avg_angle / 180.0
    return max(0.0, min(1.0, jag))


def _score_loop(
    poly: List[LatLng],
    target_m: float,
    w_dist: float = 0.7,
    w_shape: float = 0.3
) -> float:
    """
    루프 하나에 대해 '점수' 계산.
    높을수록 좋은 루프.

    - 거리 오차가 작을수록 좋음
    - 꺾임(지그재그)이 적을수록 좋음
    """
    if len(poly) < 2:
        return -1e9

    length_m = _polyline_length_m(poly)
    if length_m <= 0:
        return -1e9

    # 거리 오차 비율
    dist_err_ratio = abs(length_m - target_m) / max(target_m, 1.0)
    # 0일수록 좋음 → (1 - error)
    dist_score = 1.0 - min(dist_err_ratio, 1.0)

    # 형상 점수 (0~1, 1이 제일 좋게 뒤집기)
    jag = _compute_shape_jaggedness(poly)  # 0~1, 클수록 나쁨
    shape_score = 1.0 - jag

    total = w_dist * dist_score + w_shape * shape_score
    return total


# ----------------------------
# pivot 기반 루프 생성
# ----------------------------

def _build_loop_via_pivot(start: LatLng, pivot: LatLng) -> Optional[List[LatLng]]:
    """
    Valhalla를 이용해 start→pivot, pivot→start 경로를 구하고 병합해서 루프 생성.
    실패 시 None
    """
    out_poly = _call_valhalla_route(start, pivot)
    if not out_poly:
        return None

    back_poly = _call_valhalla_route(pivot, start)
    if not back_poly:
        return None

    return _merge_out_and_back(out_poly, back_poly)


def _search_best_loop(
    start: LatLng,
    target_m: float,
    quality_first: bool = True
) -> Optional[Tuple[List[LatLng], float]]:
    """
    여러 pivot 후보를 탐색해서 가장 점수가 높은 루프를 선택.
    반환: (best_polyline, best_score) 또는 None
    """
    # 후보 개수 조절
    if quality_first:
        n_rings = 4
        n_bearings = 16
    else:
        n_rings = 2
        n_bearings = 10

    pivots = _generate_pivot_candidates(start, target_m, n_rings=n_rings, n_bearings=n_bearings)

    best_poly: Optional[List[LatLng]] = None
    best_score: float = -1e9

    # 거리 오차가 아주 좋은 루프를 찾으면 조기 종료할 기준
    EARLY_STOP_DIST_ERR = 0.05  # 5% 이내
    EARLY_STOP_SCORE = 0.92     # 어느 정도 점수 기준

    for pivot in pivots:
        loop_poly = _build_loop_via_pivot(start, pivot)
        if not loop_poly or len(loop_poly) < 2:
            continue

        score = _score_loop(loop_poly, target_m)
        if score > best_score:
            best_score = score
            best_poly = loop_poly

        # 스피드 모드에선 괜찮은 루프 나오면 바로 끝냄
        if not quality_first and best_poly is not None:
            length_m = _polyline_length_m(best_poly)
            dist_err = abs(length_m - target_m) / max(target_m, 1.0)
            if dist_err < EARLY_STOP_DIST_ERR and score > EARLY_STOP_SCORE:
                break

    if best_poly is None:
        return None

    return best_poly, best_score
    
# ============================
# route_algo.py — Part 4/4
# ============================

# ----------------------------
# 스파이크 제거 필터
# ----------------------------

def _remove_spikes(
    poly: List[LatLng],
    angle_thresh_deg: float = 150.0,
    dist_thresh_m: float = 50.0
) -> List[LatLng]:
    """
    짧고 각도가 날카로운 스파이크를 제거.

    스파이크 정의:
      - 각도 > angle_thresh_deg (U턴급 큰 꺾임)
      - in+out 거리 합이 dist_thresh_m 미만

    점 i가 스파이크 조건이면 그 점을 삭제하면서
    반복적으로 스무딩.
    """
    if len(poly) < 5:
        return poly

    changed = True
    pts = poly[:]

    while changed:
        changed = False
        new_pts: List[LatLng] = [pts[0]]

        for i in range(1, len(pts) - 1):
            p0 = new_pts[-1]
            p1 = pts[i]
            p2 = pts[i + 1]

            # 거리 계산 (in, out)
            a = _haversine_m(p1[0], p1[1], p0[0], p0[1])
            b = _haversine_m(p1[0], p1[1], p2[0], p2[1])
            if a < 1e-3 or b < 1e-3:
                new_pts.append(p1)
                continue

            # local XY로 각도 계산
            xy = _to_local_xy([p0, p1, p2])
            (x0, y0), (x1, y1), (x2, y2) = xy
            v1 = (x0 - x1, y0 - y1)
            v2 = (x2 - x1, y2 - y1)
            norm1 = math.hypot(*v1)
            norm2 = math.hypot(*v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                new_pts.append(p1)
                continue

            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
            dot = max(-1.0, min(1.0, dot))
            angle_deg = math.degrees(math.acos(dot))

            # angle_deg: 0 = 직선, 180 = U턴
            # 150° 이상 + 짧은 거리 ⇒ 스파이크로 간주
            if angle_deg > angle_thresh_deg and (a + b) < dist_thresh_m:
                # p1 제거
                changed = True
                # append 안함
            else:
                new_pts.append(p1)

        new_pts.append(pts[-1])
        pts = new_pts

    return pts


# ----------------------------
# 퍼블릭 API: 러닝 루프 생성
# ----------------------------

def generate_running_route(
    lat: float,
    lng: float,
    km: float,
    quality_first: bool = True
) -> Dict:
    """
    메인 엔트리 포인트.

    인자:
      - lat, lng: 시작 위치
      - km: 목표 거리(km)
      - quality_first: True면 후보 pivot 많이 돌면서 품질 우선,
                       False면 적당히 괜찮은 루프 찾으면 일찍 종료.

    반환:
      {
        "status": "ok" / "error",
        "start": {"lat": ..., "lng": ...},
        "polyline": [{"lat": ..., "lng": ...}, ...],
        "distance_km": ...,
        "message": "..."
      }
    """
    start = (float(lat), float(lng))
    target_km = max(float(km), 0.5)   # 최소 0.5km
    target_m = target_km * 1000.0

    # 1) pivot 기반 루프 탐색
    result = _search_best_loop(start, target_m, quality_first=quality_first)

    if result is None:
        # fallback: 그냥 한 방향 왕복 (start→pivot→start) 시도
        pivots = _generate_pivot_candidates(start, target_m, n_rings=1, n_bearings=8)
        fallback_poly: Optional[List[LatLng]] = None

        for pivot in pivots:
            loop_poly = _build_loop_via_pivot(start, pivot)
            if loop_poly and len(loop_poly) > 1:
                fallback_poly = loop_poly
                break

        if fallback_poly is None:
            return {
                "status": "error",
                "start": {"lat": start[0], "lng": start[1]},
                "polyline": [],
                "distance_km": 0.0,
                "message": "루프 생성에 실패했습니다. (Valhalla 경로 없음)",
            }

        polyline = fallback_poly
        msg = "안전한 루프를 찾지 못해 단순 왕복 루프를 사용했습니다."
    else:
        polyline, _score = result
        msg = "고품질 러닝 루프를 생성했습니다."

    # 2) 스파이크 제거 후처리
    polyline = _remove_spikes(polyline)

    # 3) 거리 재계산
    dist_m = _polyline_length_m(polyline)
    dist_km = dist_m / 1000.0

    # 4) 형식 맞춰 반환
    poly_output = [{"lat": lat, "lng": lon} for (lat, lon) in polyline]

    return {
        "status": "ok",
        "start": {"lat": start[0], "lng": start[1]},
        "polyline": poly_output,
        "distance_km": round(dist_km, 3),
        "message": msg,
    }