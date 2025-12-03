from __future__ import annotations

import math
import os
import logging
import random
import time
import requests
import json
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

# ----------------------------
# 로거 및 기본 설정
# ----------------------------
logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

EARTH_RADIUS_M = 6371000.0

# Valhalla 설정
VALHALLA_ROUTE_URL = os.getenv("VALHALLA_ROUTE_URL", "http://localhost:8002/route")
VALHALLA_LOCATE_URL = os.getenv("VALHALLA_LOCATE_URL", "http://localhost:8002/locate")
DEFAULT_COSTING = "pedestrian"

# OSMnx 로드 (설치 안 된 경우 대비)
try:
    import osmnx as ox
except Exception:
    ox = None
    logger.warning("osmnx를 불러올 수 없습니다. 2km 미만 경로 생성에 제한이 있을 수 있습니다.")

# 타입 힌트
LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ============================================================
# 1. 공통 유틸리티 (거리 계산, 좌표 변환)
# ============================================================

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 거리 (meter)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlam/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_M * c


def _polyline_length_m(poly: Polyline) -> float:
    """폴리라인의 총 길이 (meter)."""
    if not poly or len(poly) < 2:
        return 0.0
    d = 0.0
    for i in range(len(poly) - 1):
        d += _haversine_m(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
    return d


def _to_local_xy(points: List[LatLng]) -> List[Tuple[float, float]]:
    """위경도를 첫 점 기준 평면 좌표계(x, y)로 근사 변환."""
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


# ============================================================
# 2. JSON Safe 유틸 (OSMnx 메타데이터 처리용)
# ============================================================

def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return default
    return x

def _safe_list(lst: Any) -> list:
    if not isinstance(lst, (list, tuple)):
        return []
    out = []
    for v in lst:
        if isinstance(v, (list, tuple)):
            out.append(_safe_list(v))
        elif isinstance(v, dict):
            out.append(_safe_dict(v))
        else:
            out.append(_safe_float(v, v))
    return out

def _safe_dict(d: Any) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _safe_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = _safe_list(v)
        else:
            out[k] = _safe_float(v, v)
    return out


# ============================================================
# 3. Valhalla 기반 로직 (2km 이상)
# ============================================================

def _decode_polyline6(encoded: str) -> List[LatLng]:
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
    except Exception:
        return (lat, lon)

    if not isinstance(j, list) or not j:
        return (lat, lon)

    obj = j[0]
    edges = obj.get("edges", [])
    if edges:
        e = edges[0]
        return (e.get("correlated_lat", lat), e.get("correlated_lon", lon))
    
    nodes = obj.get("nodes", [])
    if nodes:
        n = nodes[0]
        return (n.get("lat", lat), n.get("lon", lon))

    return (lat, lon)


def _call_valhalla_route(start: LatLng, end: LatLng, costing: str = DEFAULT_COSTING) -> Optional[List[LatLng]]:
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
    except requests.exceptions.RequestException as e:
        logger.warning(f"Valhalla request exception: {e}")
        return None

    if not resp.ok:
        logger.warning(f"Valhalla status {resp.status_code}: {resp.text[:100]}")
        return None

    try:
        j = resp.json()
    except Exception:
        return None

    trip = j.get("trip")
    if not trip:
        return None

    shape_data = trip.get("shape")
    if isinstance(shape_data, str):
        poly = _decode_polyline6(shape_data)
    elif isinstance(shape_data, list):
        poly = [(p["lat"], p["lon"]) for p in shape_data]
    else:
        poly = []

    if not poly:
        return None

    return poly


def _merge_out_and_back(out_poly: List[LatLng], back_poly: List[LatLng]) -> List[LatLng]:
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


def _compute_shape_jaggedness(poly: List[LatLng]) -> float:
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


def _score_loop_valhalla(poly: List[LatLng], target_m: float) -> float:
    if not poly:
        return -1e9
    length_m = _polyline_length_m(poly)
    dist_err = abs(length_m - target_m) / max(target_m, 1.0)
    dist_score = 1.0 - min(dist_err, 1.0)
    jag = _compute_shape_jaggedness(poly)
    shape_score = 1.0 - jag
    return 0.7 * dist_score + 0.3 * shape_score


def _generate_pivot_candidates(start: LatLng, target_m: float, n_rings: int = 4, n_bearings: int = 16) -> List[LatLng]:
    lat0, lon0 = start
    lat0r = math.radians(lat0)
    base_r = max(target_m * 0.45, 200.0)
    pivots: List[LatLng] = []

    for ring in range(n_rings):
        radius = base_r * (0.70 + 0.20 * ring)
        for k in range(n_bearings):
            theta = 2 * math.pi * (k / n_bearings)
            dlat = (radius / EARTH_RADIUS_M) * math.cos(theta)
            dlon = (radius / (EARTH_RADIUS_M * math.cos(lat0r))) * math.sin(theta)
            plat = lat0 + math.degrees(dlat)
            plon = lon0 + math.degrees(dlon)
            
            snapped = _snap_to_road(plat, plon)
            # Check basic connectivity
            if _call_valhalla_route(start, snapped) is not None:
                pivots.append(snapped)
    return pivots


def _build_loop_via_pivot(start: LatLng, pivot: LatLng) -> Optional[List[LatLng]]:
    outp = _call_valhalla_route(start, pivot)
    if not outp:
        return None
    backp = _call_valhalla_route(pivot, start)
    if not backp:
        return None
    return _merge_out_and_back(outp, backp)


def _search_best_loop_valhalla(start: LatLng, target_m: float, quality_first: bool = True) -> Optional[List[LatLng]]:
    n_rings = 4 if quality_first else 2
    n_bearings = 16 if quality_first else 10
    pivots = _generate_pivot_candidates(start, target_m, n_rings, n_bearings)
    
    best_poly: Optional[List[LatLng]] = None
    best_score = -1e9

    for pivot in pivots:
        loop = _build_loop_via_pivot(start, pivot)
        if not loop:
            continue
        score = _score_loop_valhalla(loop, target_m)
        if score > best_score:
            best_score = score
            best_poly = loop
    return best_poly


def _build_simple_out_and_back(start: LatLng, target_m: float) -> Optional[List[LatLng]]:
    lat0, lon0 = start
    lat0r = math.radians(lat0)
    base_r = max(target_m * 0.5, 300.0)
    candidates = []
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
            continue
        candidates.append((outp, length_m))

    if not candidates:
        return None

    best_out, _ = min(candidates, key=lambda t: abs(t[1] - target_m * 0.5))
    back = list(reversed(best_out[:-1]))
    return best_out + back


def _remove_spikes(poly: List[LatLng], angle_thresh_deg: float = 150.0, dist_thresh_m: float = 50.0) -> List[LatLng]:
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
            
            a = _haversine_m(p1[0], p1[1], p0[0], p0[1])
            b = _haversine_m(p1[0], p1[1], p2[0], p2[1])
            
            if a < 1e-3 or b < 1e-3:
                new_pts.append(p1)
                continue

            xy = _to_local_xy([p0, p1, p2])
            (x0, y0), (x1, y1), (x2, y2) = xy
            v1 = (x0-x1, y0-y1)
            v2 = (x2-x1, y2-y1)
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            
            if n1 < 1e-6 or n2 < 1e-6:
                new_pts.append(p1)
                continue
                
            dot = (v1[0]*v2[0] + v1[1]*v2[1])/(n1*n2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))
            
            if angle > angle_thresh_deg and (a+b) < dist_thresh_m:
                changed = True
            else:
                new_pts.append(p1)
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def _generate_valhalla_loop(lat: float, lng: float, km: float, quality_first: bool = True) -> Dict:
    """2km 이상 Valhalla 경로 생성 로직."""
    start_time = time.time()
    start = (float(lat), float(lng))
    target_m = max(float(km), 0.5) * 1000.0

    # 1) Pivot 루프 시도
    loop = _search_best_loop_valhalla(start, target_m, quality_first)
    msg = "고품질 러닝 루프 생성 완료 (Valhalla/Pivot)"

    # 2) 실패 시 Out-and-Back 루프 시도
    if not loop:
        loop = _build_simple_out_and_back(start, target_m)
        msg = "안전한 단순 왕복 루프로 루트를 생성했습니다. (Valhalla/Out-and-Back)"

    if not loop:
        return {
            "status": "error",
            "message": "루프 생성 실패 (Valhalla 경로 없음)",
            "start": {"lat": lat, "lng": lng},
            "polyline": [{"lat": lat, "lng": lng}],
            "distance_km": 0.0,
            "time_s": round(time.time() - start_time, 3),
        }

    loop = _remove_spikes(loop)
    dist_km = _polyline_length_m(loop) / 1000.0

    return {
        "status": "ok",
        "message": msg,
        "start": {"lat": lat, "lng": lng},
        "polyline": [{"lat": a, "lng": b} for (a,b) in loop],
        "distance_km": round(dist_km, 3),
        "time_s": round(time.time() - start_time, 3),
    }


# ============================================================
# 4. OSMnx/Graph 기반 로직 (2km 미만)
# ============================================================

# Redzone 관련 전역 변수 (Lazy Loading)
REDZONE_POLYGONS: List[Any] = []
REDZONE_TREE: Optional[STRtree] = None
_REDZONES_LOADED = False

def _load_redzones(path: str = "redzones.geojson") -> None:
    """2km 미만 로직에서만 사용되는 Redzone 데이터를 Lazy Loading."""
    global REDZONE_POLYGONS, REDZONE_TREE, _REDZONES_LOADED
    
    if _REDZONES_LOADED:
        return

    _REDZONES_LOADED = True # 시도 표시
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    polys = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom: continue
        try:
            shp = shape(geom)
            if shp.geom_type == "Polygon":
                polys.append(shp)
            elif shp.geom_type == "MultiPolygon":
                for p in shp.geoms:
                    polys.append(p)
        except Exception:
            continue
            
    REDZONE_POLYGONS = polys
    if polys:
        REDZONE_TREE = STRtree(polys)


def _is_in_redzone(lat: float, lon: float) -> bool:
    if not REDZONE_POLYGONS:
        return False
    pt = Point(lon, lat)
    candidates = REDZONE_TREE.query(pt) if REDZONE_TREE else REDZONE_POLYGONS
    for poly in candidates:
        if poly.contains(pt):
            return True
    return False


def _polyline_hits_redzone(poly: Polyline) -> bool:
    if not REDZONE_POLYGONS or not poly:
        return False
    step = max(1, len(poly) // 50)
    for i in range(0, len(poly), step):
        if _is_in_redzone(poly[i][0], poly[i][1]):
            return True
    if _is_in_redzone(poly[-1][0], poly[-1][1]):
        return True
    return False


def _polygon_roundness(poly: Polyline) -> float:
    if len(poly) < 3: return 0.0
    xy = _to_local_xy(poly)
    if xy[0] != xy[-1]: xy.append(xy[0])
    
    area = 0.0
    perimeter = 0.0
    for i in range(len(xy)-1):
        x1, y1 = xy[i]
        x2, y2 = xy[i+1]
        area += x1*y2 - x2*y1
        perimeter += math.hypot(x2-x1, y2-y1)
    
    area = abs(area) * 0.5
    if area == 0 or perimeter == 0: return 0.0
    return float(4 * math.pi * area / (perimeter**2))


def _edge_overlap_fraction(node_path: List[int]) -> float:
    if len(node_path) < 2: return 0.0
    counts = {}
    for u, v in zip(node_path[:-1], node_path[1:]):
        if u == v: continue
        e = tuple(sorted((u, v)))
        counts[e] = counts.get(e, 0) + 1
    if not counts: return 0.0
    overlaps = sum(1 for c in counts.values() if c > 1)
    return overlaps / len(counts)


def _curve_penalty(node_path: List[int], G: nx.Graph) -> float:
    if len(node_path) < 3: return 0.0
    coords = {}
    for n in node_path:
        if n not in coords:
            coords[n] = (G.nodes[n]['y'], G.nodes[n]['x'])
    
    penalty = 0.0
    for i in range(1, len(node_path)-1):
        p0 = coords[node_path[i-1]]
        p1 = coords[node_path[i]]
        p2 = coords[node_path[i+1]]
        
        # Simple vector angle calculation without projecting everything to XY
        # (Using local approximation logic inline or simplified)
        # Using existing _to_local_xy logic for consistency
        xy = _to_local_xy([p0, p1, p2])
        (x0, y0), (x1, y1), (x2, y2) = xy
        
        v1 = (x0-x1, y0-y1)
        v2 = (x2-x1, y2-y1)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6: continue
        
        dot = (v1[0]*v2[0] + v1[1]*v2[1])/(n1*n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)
        
        if theta < math.pi / 3.0:
            penalty += (math.pi/3.0 - theta)
    return penalty


def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> Polyline:
    return [(float(G.nodes[n]['y']), float(G.nodes[n]['x'])) for n in nodes]


def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0
    dlat = (side / 111111.0)
    dlng = side / (111111.0 * math.cos(math.radians(lat)))
    
    p0 = (lat, lng)
    p1 = (lat, lng + dlng)
    p2 = (lat + dlat, lng + dlng)
    p3 = (lat + dlat, lng)
    
    poly = [p0, p1, p2, p3, p0]
    return poly, _polyline_length_m(poly), _polygon_roundness(poly)


def _generate_local_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """2km 미만 OSMnx 기반 로직."""
    start_time = time.time()
    
    # 1. Redzone 데이터 로딩 (필요 시점에만 실행)
    _load_redzones()
    
    target_m = max(300.0, km * 1000.0)
    meta = {
        "len": 0, "err": 0, "roundness": 0, "score": -1e18, 
        "success": False, "length_ok": False, "used_fallback": False,
        "message": ""
    }

    if ox is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(len=length, score=r, used_fallback=True, message="OSMnx missing")
        return poly, _safe_dict(meta)

    try:
        radius_m = max(300.0, km * 600.0 + 300.0)
        G = ox.graph_from_point((lat, lng), dist=radius_m, network_type="walk", simplify=True)
        UG = ox.utils_graph.get_undirected(G)
        start_node = ox.distance.nearest_nodes(UG, X=lng, Y=lat)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(len=length, score=r, used_fallback=True, message=f"Graph load failed: {e}")
        return poly, _safe_dict(meta)

    # Dijkstra & Loop finding
    try:
        dist_map = nx.single_source_dijkstra_path_length(UG, start_node, cutoff=target_m, weight="length")
    except Exception:
        dist_map = {}

    endpoints = [n for n, d in dist_map.items() 
                 if target_m*0.3 <= d <= target_m and n != start_node]
    
    # Filter endpoints in redzone
    endpoints = [n for n in endpoints if not _is_in_redzone(UG.nodes[n]['y'], UG.nodes[n]['x'])]
    
    random.shuffle(endpoints)
    endpoints = endpoints[:80]
    
    best_poly = None
    best_score = -1e18
    best_stats = {}

    for u in endpoints:
        try:
            path1 = nx.shortest_path(UG, start_node, u, weight="length")
            # Skip if path1 goes through redzone
            if any(_is_in_redzone(UG.nodes[n]['y'], UG.nodes[n]['x']) for n in path1):
                continue
                
            path1_len = sum(nx.shortest_path_length(UG, u, v, weight="length") for u, v in zip(path1[:-1], path1[1:]))

            for v in endpoints:
                if u == v: continue
                path2 = nx.shortest_path(UG, u, v, weight="length")
                path3 = nx.shortest_path(UG, v, start_node, weight="length")
                
                full_nodes = path1 + path2[1:] + path3[1:]
                poly = _nodes_to_polyline(UG, full_nodes)
                
                if _polyline_hits_redzone(poly): continue

                length_m = _polyline_length_m(poly)
                err = abs(length_m - target_m)
                if err > target_m * 0.25: continue

                r = _polygon_roundness(poly)
                ov = _edge_overlap_fraction(full_nodes)
                cp = _curve_penalty(full_nodes, UG)
                
                length_pen = err / (max(1.0, target_m * 0.05))
                score = (2.5 * r) - (2.0 * ov) - (0.3 * cp) - (7.0 * length_pen)

                length_ok = (err <= target_m * 0.05)
                
                if score > best_score:
                    best_score = score
                    best_poly = poly
                    best_stats = {
                        "len": length_m, "err": err, "roundness": r,
                        "score": score, "length_ok": length_ok
                    }
        except Exception:
            continue

    if best_poly is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(len=length, score=r, used_fallback=True, message="No loop found")
        return poly, _safe_dict(meta)

    # Anchor start/end
    if _haversine_m(lat, lng, best_poly[0][0], best_poly[0][1]) > 1.0:
        best_poly.insert(0, (lat, lng))
    if _haversine_m(lat, lng, best_poly[-1][0], best_poly[-1][1]) > 1.0:
        best_poly.append((lat, lng))

    meta.update(best_stats)
    meta["success"] = best_stats.get("length_ok", False)
    meta["message"] = "근거리 최적 루프 생성 완료"
    meta["time_s"] = time.time() - start_time
    
    return best_poly, _safe_dict(meta)


# ============================================================
# 5. Main Entry Point
# ============================================================

def generate_running_route(
    lat: float,
    lng: float,
    km: float,
    quality_first: bool = True,
    threshold_km: float = 2.0
) -> Dict:
    """
    요청 거리(km)에 따라 분기하여 최적의 러닝 루프를 생성합니다.
    - km < 2.0: OSMnx 기반 정밀 탐색 (Redzone 회피, 원형성 고려)
    - km >= 2.0: Valhalla 기반 고속/장거리 탐색 (도로망 중심)
    """
    
    if km < threshold_km:
        # --- 2km 미만 로직 (OSMnx) ---
        poly, meta = _generate_local_loop(lat, lng, km)
        dist_km = _polyline_length_m(poly) / 1000.0
        
        return {
            "status": "ok" if meta.get("success") else "warning",
            "message": meta.get("message", "근거리 루프 생성 완료"),
            "start": {"lat": lat, "lng": lng},
            "polyline": [{"lat": lat, "lng": lng} for lat, lng in poly],
            "distance_km": round(dist_km, 3),
            "meta": meta,
        }
        
    else:
        # --- 2km 이상 로직 (Valhalla) ---
        # 이 분기에서는 OSMnx/Redzone 관련 로직이 전혀 실행되지 않습니다.
        return _generate_valhalla_loop(lat, lng, km, quality_first)
