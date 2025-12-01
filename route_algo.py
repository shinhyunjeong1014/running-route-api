import math
import random
import time
from typing import List, Tuple, Dict, Optional

import requests


###############################################################################
# Utility helpers
###############################################################################


LatLng = Tuple[float, float]


def safe_float(x: float) -> Optional[float]:
    """Return a JSON-serialisable float (NaN/inf -> None)."""
    try:
        if isinstance(x, (float, int)):
            if math.isnan(x) or math.isinf(x):
                return None
        return x
    except Exception:
        return None


def _safe_any(obj):
    if isinstance(obj, dict):
        return {k: _safe_any(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_any(v) for v in obj]
    if isinstance(obj, float):
        return safe_float(obj)
    return obj


def safe_dict(d: Dict) -> Dict:
    return _safe_any(d)


def safe_list(lst: List) -> List:
    return _safe_any(lst)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def polyline_length_m(polyline: List[LatLng]) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    dist = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        dist += haversine(lat1, lon1, lat2, lon2)
    if math.isnan(dist) or math.isinf(dist):
        return 0.0
    return dist


def _project_to_local_xy(polyline: List[LatLng], ref_lat: Optional[float] = None) -> List[Tuple[float, float]]:
    """Project lat/lng to a local planar system (meters) using simple equirect."""
    if not polyline:
        return []
    if ref_lat is None:
        ref_lat = polyline[0][0]
    ref_lat_rad = math.radians(ref_lat)
    R = 6371000.0
    xs: List[Tuple[float, float]] = []
    lat0, lon0 = polyline[0]
    for lat, lon in polyline:
        x = R * math.radians(lon - lon0) * math.cos(ref_lat_rad)
        y = R * math.radians(lat - lat0)
        xs.append((x, y))
    return xs


def polygon_roundness(polyline: List[LatLng]) -> float:
    """4πA / P² using projected coordinates. 0~1, 1 = perfect circle."""
    if len(polyline) < 4:
        return 0.0
    pts = _project_to_local_xy(polyline)
    if not pts:
        return 0.0

    # close polygon
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]

    area = 0.0
    peri = 0.0
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        area += x1 * y2 - x2 * y1
        peri += math.hypot(x2 - x1, y2 - y1)

    area = abs(area) * 0.5
    if area <= 0.0 or peri <= 0.0:
        return 0.0
    r = 4.0 * math.pi * area / (peri * peri)
    if math.isnan(r) or math.isinf(r):
        return 0.0
    return r


def _bearing_deg(a: LatLng, b: LatLng) -> float:
    """Initial bearing in degrees from point a to b."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


###############################################################################
# Valhalla routing (walk profile)
###############################################################################


VALHALLA_URL = "http://127.0.0.1:8002/route"


def _valhalla_request(locations: List[LatLng]) -> Optional[List[LatLng]]:
    """
    Call local Valhalla instance for pedestrian route.
    This assumes the Docker container is exposing /route on 8002.
    """
    if len(locations) < 2:
        return None

    loc_objs = [{"lat": lat, "lon": lon} for lat, lon in locations]
    body = {
        "locations": loc_objs,
        "costing": "pedestrian",
        "directions_options": {"units": "kilometers"},
    }

    try:
        resp = requests.post(VALHALLA_URL, json=body, timeout=5)
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    data = resp.json()
    try:
        shape: List[LatLng] = []
        for trip in data.get("trip", {}).get("legs", []):
            for edge in trip.get("shape", []):
                # When shape is already list of dicts {"lat":..,"lon":..}
                if isinstance(edge, dict) and "lat" in edge and "lon" in edge:
                    shape.append((float(edge["lat"]), float(edge["lon"])))
        if not shape:
            # Some Valhalla builds return encoded polyline "shape" string.
            # In that case we simply give up instead of re-implementing decoding.
            return None
        return shape
    except Exception:
        return None


###############################################################################
# Candidate loop generation – inspired by Random/RUNAMIC/WSRP24
###############################################################################


def _radial_targets(start: LatLng, target_m: float, n: int = 16) -> List[LatLng]:
    """
    Sample endpoints on a noisy circle around the start.
    Radius is chosen so that shortest-path distance start→end→start
    is likely to be close to target_m.
    """
    lat0, lng0 = start
    R = 6371000.0
    # rough radius: half of target, scaled down because road distance > straight line
    radius = max(80.0, target_m / 3.4)

    pts: List[LatLng] = []
    for i in range(n):
        angle = 2.0 * math.pi * (i / n) + random.uniform(-0.25, 0.25)
        d = radius * random.uniform(0.85, 1.15)
        dlat = (d * math.cos(angle)) / R
        dlng = (d * math.sin(angle)) / (R * math.cos(math.radians(lat0)))
        pts.append((lat0 + math.degrees(dlat), lng0 + math.degrees(dlng)))
    return pts


def _chunk_polyline(poly: List[LatLng]) -> List[Tuple[LatLng, LatLng]]:
    return list(zip(poly[:-1], poly[1:])) if len(poly) >= 2 else []


def _segment_overlap_score(poly: List[LatLng]) -> float:
    """
    Very lightweight self-overlap penalty.
    Count how many times short segments share almost the same midpoint.
    """
    if len(poly) < 4:
        return 0.0

    segs = _chunk_polyline(poly)
    buckets: Dict[Tuple[int, int], int] = {}
    for (a, b) in segs:
        mid_lat = (a[0] + b[0]) * 0.5
        mid_lng = (a[1] + b[1]) * 0.5
        key = (int(mid_lat * 1e4), int(mid_lng * 1e4))
        buckets[key] = buckets.get(key, 0) + 1

    overlap = sum(c - 1 for c in buckets.values() if c > 1)
    return float(overlap)


def _curvature_score(poly: List[LatLng]) -> float:
    """
    Penalise extremely sharp turns (U-turn-like).
    Lower score is better; will be subtracted from roundness.
    """
    if len(poly) < 3:
        return 0.0
    pts = poly
    penalty = 0.0
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        br1 = _bearing_deg(a, b)
        br2 = _bearing_deg(b, c)
        diff = abs(br2 - br1)
        if diff > 180.0:
            diff = 360.0 - diff
        # If turning more than 135 deg, treat as almost U-turn
        if diff > 135.0:
            penalty += (diff - 135.0) / 45.0
    return penalty


def _loop_quality(poly: List[LatLng], target_m: float) -> Dict[str, float]:
    """Compute error, roundness, overlap, and overall score."""
    length = polyline_length_m(poly)
    if length <= 0.0:
        return {
            "length": 0.0,
            "err": float("inf"),
            "roundness": 0.0,
            "overlap": float("inf"),
            "score": -1e9,
        }

    err = abs(length - target_m)
    roundness = polygon_roundness(poly)
    overlap = _segment_overlap_score(poly)
    curve_pen = _curvature_score(poly)

    # Normalise error relative to target (so 50m vs 5km is small)
    rel_err = err / max(target_m, 1.0)

    # Higher is better
    score = (
        +1.0 * roundness
        - 3.0 * rel_err
        - 0.3 * overlap
        - 0.5 * curve_pen
    )

    return {
        "length": length,
        "err": err,
        "roundness": roundness,
        "overlap": overlap,
        "curve_penalty": curve_pen,
        "score": score,
    }


###############################################################################
# Fallback: simple geometric square loop (last resort)
###############################################################################


def _fallback_square_loop(start: LatLng, km: float) -> List[LatLng]:
    """Pure geometric fallback – used only when all routing fails."""
    lat, lng = start
    target_m = km * 1000.0
    # half-diagonal of square ~ target/4, so side ~ target/4*sqrt(2)
    half_side = max(80.0, target_m / 4.0 / math.sqrt(2.0))
    dlat = (half_side / 6371000.0) * 180.0 / math.pi
    dlng = dlat / math.cos(math.radians(lat))

    a = (lat + dlat, lng)
    b = (lat, lng + dlng)
    c = (lat - dlat, lng)
    d = (lat, lng - dlng)
    poly = [a, b, c, d, a]
    return poly


###############################################################################
# Main public API – used by app.py
###############################################################################


def generate_area_loop(lat: float, lng: float, km: float):
    """
    Generate a running loop around (lat, lng) of approximately `km` kilometers.

    Algorithmic ideas mixed from:
      - Random round-trip route generation on pedestrian graphs
      - RUNAMIC style rod + detour cycles with cost poisoning
      - WSRP24 cycle quality metrics (roundness, overlap)
    but implemented in a lightweight way that fits a single Python file.
    """
    start_time = time.time()
    start: LatLng = (lat, lng)
    target_m = km * 1000.0

    best_poly: Optional[List[LatLng]] = None
    best_metrics: Optional[Dict[str, float]] = None
    valhalla_calls = 0
    kakao_calls = 0  # kept for compatibility in meta
    routes_checked = 0
    routes_validated = 0

    # 1) sample radial endpoints
    endpoints = _radial_targets(start, target_m, n=20)

    # 2) build candidate loops of increasing complexity
    #    (single detour rod, then two-point rods) – RUNAMIC style
    candidates: List[List[LatLng]] = []

    # 2-1: simple out-and-back rods
    for p in endpoints:
        fwd = _valhalla_request([start, p])
        valhalla_calls += 1
        if not fwd or len(fwd) < 2:
            continue
        back = _valhalla_request([p, start])
        valhalla_calls += 1
        if not back or len(back) < 2:
            continue

        # join (Random-paper flavour: 현재는 단순 연결, 나중에 내부 세그먼트 섞는 것도 가능)
        poly = fwd + back[1:]
        candidates.append(poly)

    # 2-2: two-point cycles: start → p_i → p_j → start
    if len(endpoints) >= 3:
        pair_indices = [
            (i, j)
            for i in range(len(endpoints))
            for j in range(len(endpoints))
            if i != j
        ]
        random_pairs = random.sample(
            pair_indices,
            k=min(30, len(pair_indices)),
        )
        for i, j in random_pairs:
            p1 = endpoints[i]
            p2 = endpoints[j]
            f1 = _valhalla_request([start, p1])
            valhalla_calls += 1
            if not f1 or len(f1) < 2:
                continue
            f2 = _valhalla_request([p1, p2])
            valhalla_calls += 1
            if not f2 or len(f2) < 2:
                continue
            f3 = _valhalla_request([p2, start])
            valhalla_calls += 1
            if not f3 or len(f3) < 2:
                continue
            poly = f1 + f2[1:] + f3[1:]
            candidates.append(poly)

    # 3) Evaluate candidates with multi-criteria score
    for poly in candidates:
        routes_checked += 1
        metrics = _loop_quality(poly, target_m)
        if not math.isfinite(metrics["err"]):
            continue
        routes_validated += 1
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_metrics = metrics
            best_poly = poly

    used_fallback = False

    # 4) If we still don't have anything usable or error is huge, fallback
    if best_poly is None or best_metrics is None or best_metrics["err"] > 200.0:
        used_fallback = True
        best_poly = _fallback_square_loop(start, km)
        best_metrics = _loop_quality(best_poly, target_m)

    elapsed = time.time() - start_time

    # 5) Build meta
    meta: Dict[str, object] = {
        "len": safe_float(best_metrics.get("length", 0.0)),
        "err": safe_float(best_metrics.get("err", 0.0)),
        "roundness": safe_float(best_metrics.get("roundness", 0.0)),
        "overlap": safe_float(best_metrics.get("overlap", 0.0)),
        "curve_penalty": safe_float(best_metrics.get("curve_penalty", 0.0)),
        "score": safe_float(best_metrics.get("score", 0.0)),
        "success": not used_fallback,
        "used_fallback": used_fallback,
        "valhalla_calls": valhalla_calls,
        "kakao_calls": kakao_calls,
        "routes_checked": routes_checked,
        "routes_validated": routes_validated,
        "km_requested": km,
        "target_m": safe_float(target_m),
        "time_s": safe_float(elapsed),
        "message": (
            "논문 스타일 Valhalla 러닝 루프 생성 성공"
            if not used_fallback
            else "Valhalla 기반 후보가 부족해 기하학적 사각형 루프를 사용했습니다."
        ),
    }

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)

    return safe_poly, safe_meta
