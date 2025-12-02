# ============================================
# route_algo.py (Final Stable Version)
# ============================================

import math
import os
import logging
from typing import List, Tuple, Dict, Optional
import requests

# --------------------------------------------
# 기본 설정
# --------------------------------------------
LatLng = Tuple[float, float]
EARTH_RADIUS_M = 6371000.0

VALHALLA_ROUTE_URL = os.getenv("VALHALLA_ROUTE_URL", "http://localhost:8002/route")
VALHALLA_LOCATE_URL = os.getenv("VALHALLA_LOCATE_URL", "http://localhost:8002/locate")

DEFAULT_COSTING = "pedestrian"

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)


# --------------------------------------------
# 거리 / geometry
# --------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlam/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_M * c


def _polyline_length_m(poly: List[LatLng]) -> float:
    if len(poly) < 2:
        return 0.0
    d = 0
    for i in range(len(poly) - 1):
        d += _haversine_m(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
    return d


def _to_local_xy(points: List[LatLng]) -> List[Tuple[float, float]]:
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


# --------------------------------------------
# polyline6 decoder
# --------------------------------------------
def _decode_polyline6(encoded: str) -> List[LatLng]:
    if not encoded:
        return []
    coords = []
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


# --------------------------------------------
# /locate 기반 pivot 스냅 (edges 우선)
# --------------------------------------------
def _snap_to_road(lat: float, lon: float) -> LatLng:
    try:
        resp = requests.post(
            VALHALLA_LOCATE_URL,
            json={"locations": [{"lat": lat, "lon": lon}]},
            timeout=1.5
        )
    except Exception:
        return (lat, lon)

    if not resp.ok:
        return (lat, lon)

    try:
        j = resp.json()
    except:
        return (lat, lon)

    if not isinstance(j, list) or not j:
        return (lat, lon)

    obj = j[0]

    # ----- edges 기반 스냅 -----
    edges = obj.get("edges", [])
    if edges:
        e = edges[0]
        return (
            e.get("correlated_lat", lat),
            e.get("correlated_lon", lon)
        )

    # ----- nodes 기반 스냅 (fallback) -----
    nodes = obj.get("nodes", [])
    if nodes:
        n = nodes[0]
        return (
            n.get("lat", lat),
            n.get("lon", lon)
        )

    return (lat, lon)


# --------------------------------------------
# Valhalla /route 호출
# --------------------------------------------
def _call_valhalla_route(start: LatLng, end: LatLng, costing=DEFAULT_COSTING) -> Optional[List[LatLng]]:
    payload = {
        "locations": [
            {"lat": start[0], "lon": start[1]},
            {"lat": end[0], "lon": end[1]},
        ],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }

    try:
        resp = requests.post(VALHALLA_ROUTE_URL, json=payload, timeout=4.0)
    except Exception as e:
        logger.warning(f"Valhalla request exception: {e}")
        return None

    if not resp.ok:
        return None

    try:
        j = resp.json()
    except:
        return None

    trip = j.get("trip")
    if not trip:
        return None

    shape = trip.get("shape", "")
    if isinstance(shape, str):
        poly = _decode_polyline6(shape)
    elif isinstance(shape, list):
        poly = [(p["lat"], p["lon"]) for p in shape]
    else:
        poly = []

    if not poly:
        return None

    return poly


def _merge_out_and_back(out_poly, back_poly):
    if not out_poly:
        return back_poly[:] if back_poly else []
    if not back_poly:
        return out_poly[:]

    merged = out_poly[:]
    if merged[-1] == back_poly[0]:
        merged.extend(back_poly[1:])
    else:
        merged.extend(back_poly)
    return merged


# --------------------------------------------
# pivot 후보 생성 + 스냅 + 연결성(경로 가능 여부) 필터링
# --------------------------------------------
def _generate_pivot_candidates(start: LatLng, target_m: float, n_rings=4, n_bearings=16):
    lat0, lon0 = start
    lat0r = math.radians(lat0)

    base_r = max(target_m * 0.45, 200.0)

    pivots = []
    for ring in range(n_rings):
        radius = base_r * (0.70 + 0.20 * ring)

        for k in range(n_bearings):
            theta = 2 * math.pi * (k / n_bearings)
            dlat = (radius / EARTH_RADIUS_M) * math.cos(theta)
            dlon = (radius / (EARTH_RADIUS_M * math.cos(lat0r))) * math.sin(theta)

            plat = lat0 + math.degrees(dlat)
            plon = lon0 + math.degrees(dlon)

            # 1) 스냅 (edges 기반)
            snapped = _snap_to_road(plat, plon)

            # 2) 연결성 검사 (out-route 가능해야 함)
            test = _call_valhalla_route(start, snapped)
            if test is None:
                continue

            pivots.append(snapped)

    return pivots


# --------------------------------------------
# 루프 품질
# --------------------------------------------
def _compute_shape_jaggedness(poly):
    if len(poly) < 3:
        return 0.0
    xy = _to_local_xy(poly)
    total_angle = 0
    count = 0

    for i in range(1, len(poly)-1):
        x0,y0 = xy[i-1]
        x1,y1 = xy[i]
        x2,y2 = xy[i+1]

        v1 = (x0-x1, y0-y1)
        v2 = (x2-x1, y2-y1)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        dot = (v1[0]*v2[0] + v1[1]*v2[1])/(n1*n2)
        dot = max(-1,min(1,dot))
        ang = math.degrees(math.acos(dot))

        total_angle += ang
        count += 1

    if count == 0:
        return 0.0

    avg = total_angle / count
    return min(1.0, avg / 180.0)


def _score_loop(poly, target_m):
    if not poly:
        return -1e9

    length_m = _polyline_length_m(poly)
    dist_err = abs(length_m - target_m) / max(target_m, 1.0)
    dist_score = 1 - min(dist_err, 1.0)

    jag = _compute_shape_jaggedness(poly)
    shape_score = 1 - jag

    return 0.7 * dist_score + 0.3 * shape_score


# --------------------------------------------
# 루프 생성
# --------------------------------------------
def _build_loop_via_pivot(start, pivot):
    outp = _call_valhalla_route(start, pivot)
    if not outp:
        return None

    backp = _call_valhalla_route(pivot, start)
    if not backp:
        return None

    return _merge_out_and_back(outp, backp)


def _search_best_loop(start, target_m, quality_first=True):
    n_rings = 4 if quality_first else 2
    n_bearings = 16 if quality_first else 10

    pivots = _generate_pivot_candidates(start, target_m, n_rings, n_bearings)

    best_poly = None
    best_score = -1e9

    for pivot in pivots:
        loop = _build_loop_via_pivot(start, pivot)
        if not loop:
            continue

        score = _score_loop(loop, target_m)
        if score > best_score:
            best_score = score
            best_poly = loop

    if best_poly is None:
        return None

    return (best_poly, best_score)


# --------------------------------------------
# 스파이크 제거
# --------------------------------------------
def _remove_spikes(poly, angle_thresh_deg=150.0, dist_thresh_m=50.0):
    if len(poly) < 5:
        return poly

    pts = poly[:]
    changed = True

    while changed:
        changed = False
        new_pts = [pts[0]]

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
            dot = max(-1,min(1,dot))
            angle = math.degrees(math.acos(dot))

            if angle > angle_thresh_deg and (a+b) < dist_thresh_m:
                changed = True
            else:
                new_pts.append(p1)

        new_pts.append(pts[-1])
        pts = new_pts

    return pts


# --------------------------------------------
# Public API
# --------------------------------------------
def generate_running_route(lat, lng, km, quality_first=True):
    start = (float(lat), float(lng))
    target_m = max(float(km), 0.5) * 1000

    result = _search_best_loop(start, target_m, quality_first)

    if result is None:
        return {
            "status": "error",
            "message": "루프 생성 실패 (pivot 경로 없음)",
            "start": {"lat": lat, "lng": lng},
            "polyline": [],
            "distance_km": 0.0
        }

    poly, score = result
    poly = _remove_spikes(poly)

    dist_km = _polyline_length_m(poly) / 1000

    return {
        "status": "ok",
        "message": "고품질 러닝 루프 생성 완료",
        "start": {"lat": lat, "lng": lng},
        "polyline": [{"lat": a, "lng": b} for (a,b) in poly],
        "distance_km": round(dist_km, 3)
    }