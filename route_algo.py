import math
import os
import logging
from typing import List, Tuple, Dict, Optional
import requests

# ----------------------------
# 설정 및 상수
# ----------------------------
LatLng = Tuple[float, float]
EARTH_RADIUS_M = 6371000.0

# 로컬 uvicorn 실행 환경에 맞춰 localhost로 설정
VALHALLA_ROUTE_URL = os.getenv("VALHALLA_ROUTE_URL", "http://localhost:8002/route")
VALHALLA_LOCATE_URL = os.getenv("VALHALLA_LOCATE_URL", "http://localhost:8002/locate")

DEFAULT_COSTING = "pedestrian"

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)


# ----------------------------
# 기본 거리/좌표 유틸
# ----------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이의 구면 거리(m) 계산"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlam/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_M * c


def _polyline_length_m(poly: List[LatLng]) -> float:
    """폴리라인 전체 길이(m) 계산"""
    if len(poly) < 2:
        return 0.0
    d = 0
    for i in range(len(poly) - 1):
        d += _haversine_m(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
    return d


def _to_local_xy(points: List[LatLng]) -> List[Tuple[float, float]]:
    """좌표계를 미터 단위의 로컬 평면 좌표(x, y)로 근사 변환 (각도 계산용)"""
    if not points:
        return []
    lat0, lon0 = points[0]
    lat0r = math.radians(lat0)
    res = []
    for lat, lon in points:
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        x = EARTH_RADIUS_M * dlon * math.cos(lat0r)
        y = EARTH_RADIUS_M * dlat
        res.append((x, y))
    return res


# ----------------------------
# polyline6 디코딩 (Valhalla 포맷)
# ----------------------------
def _decode_polyline6(encoded: str) -> List[LatLng]:
    """Valhalla의 encoded shape string(6자리 정밀도)을 좌표 리스트로 변환"""
    if not encoded:
        return []
    coords: List[LatLng] = []
    index = 0
    lat = 0
    lng = 0
    L = len(encoded)

    while index < L:
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

        coords.append((lat/1e6, lng/1e6))

    return coords


# ----------------------------
# /locate 기반 스냅 (도로망 매칭)
# ----------------------------
def _snap_to_road(lat: float, lon: float) -> LatLng:
    """좌표를 가장 가까운 도보 가능한 도로로 스냅"""
    try:
        resp = requests.post(
            VALHALLA_LOCATE_URL,
            json={"locations": [{"lat": lat, "lon": lon}], "costing": "pedestrian"},
            timeout=1.5
        )
    except Exception:
        return (lat, lon)

    if not resp.ok:
        return (lat, lon)

    try:
        j = resp.json()
    except Exception:
        return (lat, lon)

    if not isinstance(j, list) or not j:
        return (lat, lon)

    obj = j[0]

    # edges 기반 스냅 (도로 선형 위)
    edges = obj.get("edges", [])
    if edges:
        e = edges[0]
        return (
            e.get("correlated_lat", lat),
            e.get("correlated_lon", lon),
        )

    # nodes 기반 (교차로 등)
    nodes = obj.get("nodes", [])
    if nodes:
        n = nodes[0]
        return (
            n.get("lat", lat),
            n.get("lon", lon),
        )

    return (lat, lon)


# ----------------------------
# Valhalla /route 호출 (핵심 수정됨)
# ----------------------------
def _call_valhalla_route(start: LatLng, end: LatLng, costing: str = DEFAULT_COSTING) -> Optional[List[LatLng]]:
    """두 지점 간의 경로 탐색 요청"""
    payload = {
        "locations": [
            {"lat": start[0], "lon": start[1]},
            {"lat": end[0], "lon": end[1]},
        ],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }

    try:
        resp = requests.post(VALHALLA_ROUTE_URL, json=payload, timeout=8.0)
    except Exception as e:
        logger.warning(f"Valhalla request exception: {e}")
        return None

    if not resp.ok:
        logger.warning(f"Valhalla returned error: {resp.status_code} - {resp.text}")
        return None

    try:
        j = resp.json()
    except Exception:
        return None

    trip = j.get("trip")
    if not trip:
        return None

    # [수정] trip 바로 아래가 아니라 legs 리스트 내부에서 shape를 찾아야 합니다.
    legs = trip.get("legs", [])
    poly = []

    if legs:
        for leg in legs:
            shape = leg.get("shape")
            if isinstance(shape, str):
                decoded = _decode_polyline6(shape)
                poly.extend(decoded)
    else:
        # Fallback: 혹시 구조가 다른 경우
        shape = trip.get("shape")
        if isinstance(shape, str):
            poly = _decode_polyline6(shape)

    if not poly:
        return None

    return poly


def _merge_out_and_back(out_poly: List[LatLng], back_poly: List[LatLng]) -> List[LatLng]:
    """가는 경로와 오는 경로를 하나로 합침"""
    if not out_poly:
        return back_poly[:] if back_poly else []
    if not back_poly:
        return out_poly[:]
    
    merged = out_poly[:]
    # 연결 부위 중복 제거
    if merged[-1] == back_poly[0]:
        merged.extend(back_poly[1:])
    else:
        merged.extend(back_poly)
    return merged


# ----------------------------
# 루프 생성 로직 1: Pivot 방식 (원형/삼각형 형태 유도)
# ----------------------------
def _generate_pivot_candidates(
    start: LatLng,
    target_m: float,
    n_rings: int = 4,
    n_bearings: int = 16
) -> List[LatLng]:
    lat0, lon0 = start
    lat0r = math.radians(lat0)
    
    # 목표 거리의 약 절반 지점을 반환점으로 잡음 (왕복 고려)
    # 0.45를 곱하는 이유는 직선거리가 아닌 도로거리 오차 고려
    base_r = max(target_m * 0.45, 200.0)

    pivots: List[LatLng] = []

    for ring in range(n_rings):
        # 여러 반경으로 후보군 탐색
        radius = base_r * (0.70 + 0.20 * ring)
        for k in range(n_bearings):
            theta = 2 * math.pi * (k / n_bearings)
            dlat = (radius / EARTH_RADIUS_M) * math.cos(theta)
            dlon = (radius / (EARTH_RADIUS_M * math.cos(lat0r))) * math.sin(theta)

            plat = lat0 + math.degrees(dlat)
            plon = lon0 + math.degrees(dlon)

            snapped = _snap_to_road(plat, plon)

            # start → pivot 경로가 실제로 존재하는지 가볍게 체크 (선택사항, 성능 위해 생략 가능하나 정확도 위해 유지)
            # 여기서는 pivot 좌표만 수집하고 실제 경로는 나중에 계산
            pivots.append(snapped)

    return pivots


# ----------------------------
# 루프 품질 평가 점수
# ----------------------------
def _compute_shape_jaggedness(poly: List[LatLng]) -> float:
    """경로가 얼마나 꼬불꼬불한지(Jaggedness) 계산. 낮을수록 좋음(0~1)"""
    if len(poly) < 3:
        return 0.0

    xy = _to_local_xy(poly)
    total_angle = 0.0
    count = 0

    for i in range(1, len(poly) - 1):
        x0, y0 = xy[i - 1]
        x1, y1 = xy[i]
        x2, y2 = xy[i + 1]

        v1 = (x0 - x1, y0 - y1)
        v2 = (x2 - x1, y2 - y1)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        dot = (v1[0]*v2[0] + v1[1]*v2[1])/(n1*n2)
        dot = max(-1.0, min(1.0, dot))
        ang = math.degrees(math.acos(dot))

        total_angle += ang
        count += 1

    if count == 0:
        return 0.0

    avg = total_angle / count
    return min(1.0, avg / 180.0)


def _score_loop(poly: List[LatLng], target_m: float) -> float:
    """거리 정확도와 모양 단순성을 종합하여 점수 매기기"""
    if not poly:
        return -1e9

    length_m = _polyline_length_m(poly)
    # 거리 오차율 (작을수록 좋음)
    dist_err = abs(length_m - target_m) / max(target_m, 1.0)
    dist_score = 1.0 - min(dist_err, 1.0)

    # 모양 점수 (직선에 가까울수록 좋음)
    jag = _compute_shape_jaggedness(poly)
    shape_score = 1.0 - jag

    # 가중치: 거리 70%, 모양 30%
    return 0.7 * dist_score + 0.3 * shape_score


def _build_loop_via_pivot(start: LatLng, pivot: LatLng) -> Optional[List[LatLng]]:
    """Start -> Pivot -> Start 로 이어지는 루프 생성"""
    outp = _call_valhalla_route(start, pivot)
    if not outp:
        return None

    backp = _call_valhalla_route(pivot, start)
    if not backp:
        return None

    return _merge_out_and_back(outp, backp)


def _search_best_loop(
    start: LatLng,
    target_m: float,
    quality_first: bool = True
) -> Optional[List[LatLng]]:
    """여러 Pivot 후보 중 가장 점수가 높은 루프 탐색"""
    n_rings = 4 if quality_first else 2
    n_bearings = 16 if quality_first else 10

    pivots = _generate_pivot_candidates(start, target_m, n_rings, n_bearings)
    if not pivots:
        return None

    best_poly: Optional[List[LatLng]] = None
    best_score = -1e9

    # 모든 후보에 대해 경로를 다 찍어보면 느릴 수 있으므로, 실제로는 
    # 랜덤 샘플링하거나 거리 계산을 먼저 해볼 수도 있으나, 여기선 전체 탐색
    for pivot in pivots:
        loop = _build_loop_via_pivot(start, pivot)
        if not loop:
            continue

        score = _score_loop(loop, target_m)
        if score > best_score:
            best_score = score
            best_poly = loop

    return best_poly


# ----------------------------
# 루프 생성 로직 2: 단순 왕복 (Fallback)
# ----------------------------
def _build_simple_out_and_back(start: LatLng, target_m: float) -> Optional[List[LatLng]]:
    """단순히 한 방향으로 갔다가 되돌아오는 코스"""
    lat0, lon0 = start
    lat0r = math.radians(lat0)

    # target/2 근처까지 한 방향으로 나갔다 오는 루프 만들기
    base_r = max(target_m * 0.5, 300.0)
    candidates: List[Tuple[List[LatLng], float]] = []
    n_bearings = 12

    for k in range(n_bearings):
        theta = 2 * math.pi * (k / n_bearings)
        dlat = (base_r / EARTH_RADIUS_M) * math.cos(theta)
        dlon = (base_r / (EARTH_RADIUS_M * math.cos(lat0r))) * math.sin(theta)

        plat = lat0 + math.degrees(dlat)
        plon = lon0 + math.degrees(dlon)

        snapped = _snap_to_road(plat, plon)
        outp = _call_valhalla_route(start, snapped)
        if not outp:
            continue

        length_m = _polyline_length_m(outp)
        if length_m < target_m * 0.25:
            # 너무 짧으면 패스
            continue

        candidates.append((outp, length_m))

    if not candidates:
        return None

    # 목표 거리 절반(편도)에 가장 가까운 경로 선택
    best_out, _ = min(
        candidates,
        key=lambda t: abs(t[1] - target_m * 0.5)
    )

    # 왕복 루프: out + reverse(out)
    back = list(reversed(best_out[:-1]))
    loop = best_out + back
    return loop


# ----------------------------
# 후처리: 스파이크(불필요한 뾰족한 구간) 제거
# ----------------------------
def _remove_spikes(
    poly: List[LatLng],
    angle_thresh_deg: float = 150.0,
    dist_thresh_m: float = 50.0
) -> List[LatLng]:
    """갔던 길을 바로 되돌아오거나 너무 급격하게 꺾이는 짧은 구간 제거"""
    if len(poly) < 5:
        return poly

    pts = poly[:]
    changed = True

    while changed:
        changed = False
        new_pts: List[LatLng] = [pts[0]]

        for i in range(1, len(pts)-1):
            p0 = new_pts[-1]
            p1 = pts[i]
            p2 = pts[i+1]

            a = _haversine_m(p1[0],p1[1], p0[0],p0[1])
            b = _haversine_m(p1[0],p1[1], p2[0],p2[1])

            if a < 1e-3 or b < 1e-3:
                new_pts.append(p1)
                continue

            xy = _to_local_xy([p0,p1,p2])
            (x0,y0),(x1,y1),(x2,y2) = xy
            v1 = (x0-x1, y0-y1)
            v2 = (x2-x1, y2-y1)
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)

            if n1 < 1e-6 or n2 < 1e-6:
                new_pts.append(p1)
                continue

            dot = (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))

            # 각도가 급하고(거의 U턴), 그 구간 길이가 짧으면 스파이크로 간주
            if angle > angle_thresh_deg and (a+b) < dist_thresh_m:
                changed = True
                # p1을 추가하지 않음으로써 제거 효과
            else:
                new_pts.append(p1)

        new_pts.append(pts[-1])
        pts = new_pts

    return pts


# ----------------------------
# Public API (Main Entry Point)
# ----------------------------
def generate_running_route(
    lat: float,
    lng: float,
    km: float,
    quality_first: bool = True
) -> Dict:
    start = (float(lat), float(lng))
    target_m = max(float(km), 0.5) * 1000.0

    # 1) pivot 기반 고품질 루프 시도
    loop = _search_best_loop(start, target_m, quality_first)
    if loop:
        msg = "고품질 러닝 루프 생성 완료"
    else:
        # 2) 실패 시: 단순 out-and-back 루프 생성 (성공률 매우 높음)
        loop = _build_simple_out_and_back(start, target_m)
        if loop:
            msg = "안전한 단순 왕복 루프로 루트를 생성했습니다."
        else:
            return {
                "status": "error",
                "message": "루프 생성 실패 (Valhalla 경로 탐색 불가)",
                "start": {"lat": lat, "lng": lng},
                "polyline": [],
                "distance_km": 0.0,
            }

    # 스파이크 제거 (불필요한 꼬임 정리)
    loop = _remove_spikes(loop)

    dist_km = _polyline_length_m(loop) / 1000.0

    return {
        "status": "ok",
        "message": msg,
        "start": {"lat": lat, "lng": lng},
        "polyline": [{"lat": a, "lng": b} for (a,b) in loop],
        "distance_km": round(dist_km, 3),
    }
