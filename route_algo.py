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
# 공통 설정 및 로거
# ----------------------------
logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

LatLng = Tuple[float, float]
Polyline = List[LatLng]

# Valhalla 설정
VALHALLA_ROUTE_URL = os.getenv("VALHALLA_ROUTE_URL", "http://localhost:8002/route")
VALHALLA_LOCATE_URL = os.getenv("VALHALLA_LOCATE_URL", "http://localhost:8002/locate")
DEFAULT_COSTING = "pedestrian"

# OSMnx 로드 (2km 미만용)
try:
    import osmnx as ox
except Exception:
    ox = None
    logger.warning("osmnx import failed.")


# ============================================================
# [PART A] 2km 이상: Valhalla 로직 (원본 코드 1 보존)
# ※ 이 영역의 함수들은 2km 미만 로직과 일절 공유되지 않습니다.
# ============================================================

EARTH_RADIUS_M = 6371000.0

def _haversine_m(lat1, lon1, lat2, lon2):
    """[Valhalla 전용] 거리 계산"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlam/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_M * c


def _polyline_length_m(poly: List[LatLng]) -> float:
    """[Valhalla 전용] 폴리라인 길이"""
    if len(poly) < 2:
        return 0.0
    d = 0
    for i in range(len(poly) - 1):
        d += _haversine_m(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
    return d


def _to_local_xy(points: List[LatLng]) -> List[Tuple[float, float]]:
    """[Valhalla 전용] 좌표 변환"""
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


def _decode_polyline6(encoded: str) -> List[LatLng]:
    """[Valhalla 전용] Polyline6 디코딩"""
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
    """[Valhalla 전용] Locate 호출"""
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
    """[Valhalla 전용] Route 호출"""
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
        logger.warning(f"Valhalla API Error: {resp.status_code}")
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
            if _call_valhalla_route(start, snapped) is None:
                continue
            pivots.append(snapped)
    return pivots


def _compute_shape_jaggedness(poly: List[LatLng]) -> float:
    if len(poly) < 3: return 0.0
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
        if n1 < 1e-6 or n2 < 1e-6: continue
        dot = (v1[0]*v2[0] + v1[1]*v2[1])/(n1*n2)
        dot = max(-1.0, min(1.0, dot))
        ang = math.degrees(math.acos(dot))
        total_angle += ang
        count += 1
    if count == 0: return 0.0
    avg = total_angle / count
    return min(1.0, avg / 180.0)


def _score_loop(poly: List[LatLng], target_m: float) -> float:
    if not poly: return -1e9
    length_m = _polyline_length_m(poly)
    dist_err = abs(length_m - target_m) / max(target_m, 1.0)
    dist_score = 1.0 - min(dist_err, 1.0)
    jag = _compute_shape_jaggedness(poly)
    shape_score = 1.0 - jag
    return 0.7 * dist_score + 0.3 * shape_score


def _build_loop_via_pivot(start: LatLng, pivot: LatLng) -> Optional[List[LatLng]]:
    outp = _call_valhalla_route(start, pivot)
    if not outp: return None
    backp = _call_valhalla_route(pivot, start)
    if not backp: return None
    return _merge_out_and_back(outp, backp)


def _search_best_loop(start: LatLng, target_m: float, quality_first: bool = True) -> Optional[List[LatLng]]:
    n_rings = 4 if quality_first else 2
    n_bearings = 16 if quality_first else 10
    pivots = _generate_pivot_candidates(start, target_m, n_rings, n_bearings)
    if not pivots: return None
    best_poly = None
    best_score = -1e9
    for pivot in pivots:
        loop = _build_loop_via_pivot(start, pivot)
        if not loop: continue
        score = _score_loop(loop, target_m)
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
        if not outp: continue

        length_m = _polyline_length_m(outp)
        if length_m < target_m * 0.25: continue
        candidates.append((outp, length_m))

    if not candidates: return None
    best_out, _ = min(candidates, key=lambda t: abs(t[1] - target_m * 0.5))
    back = list(reversed(best_out[:-1]))
    return best_out + back


def _remove_spikes(poly: List[LatLng], angle_thresh_deg: float = 150.0, dist_thresh_m: float = 50.0) -> List[LatLng]:
    if len(poly) < 5: return poly
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
                new_pts.append(p1); continue
            
            # 여기서 _to_local_xy는 Valhalla용 함수를 사용
            xy = _to_local_xy([p0,p1,p2])
            (x0,y0),(x1,y1),(x2,y2) = xy
            v1 = (x0-x1, y0-y1)
            v2 = (x2-x1, y2-y1)
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 < 1e-6 or n2 < 1e-6:
                new_pts.append(p1); continue
            dot = (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))
            if angle > angle_thresh_deg and (a+b) < dist_thresh_m:
                changed = True
            else:
                new_pts.append(p1)
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


# [Public Wrapper for Valhalla]
def _generate_valhalla_loop(lat: float, lng: float, km: float, quality_first: bool = True) -> Dict:
    start_time = time.time()
    start = (float(lat), float(lng))
    target_m = max(float(km), 0.5) * 1000.0

    loop = _search_best_loop(start, target_m, quality_first)
    if loop:
        msg = "고품질 러닝 루프 생성 완료"
    else:
        loop = _build_simple_out_and_back(start, target_m)
        if loop:
            msg = "안전한 단순 왕복 루프로 루트를 생성했습니다."
        else:
            return {
                "status": "error",
                "message": "루프 생성 실패 (Valhalla 경로 없음)",
                "start": {"lat": lat, "lng": lng},
                "polyline": [{"lat": lat, "lng": lng}],
                "distance_km": 0.0,
                "time_s": round(time.time() - start_time, 3)
            }

    loop = _remove_spikes(loop)
    dist_km = _polyline_length_m(loop) / 1000.0

    return {
        "status": "ok",
        "message": msg,
        "start": {"lat": lat, "lng": lng},
        "polyline": [{"lat": a, "lng": b} for (a,b) in loop],
        "distance_km": round(dist_km, 3),
        "time_s": round(time.time() - start_time, 3)
    }


# ============================================================
# [PART B] 2km 미만: OSMnx 로직 (원본 코드 2 기반, 격리 적용)
# ※ 충돌 방지를 위해 내부 함수명에 모두 '_osm_' 접두어 적용
# ============================================================

def _osm_safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x): return default
    return x

def _osm_safe_list(lst: Any) -> list:
    if not isinstance(lst, (list, tuple)): return []
    out = []
    for v in lst:
        if isinstance(v, (list, tuple)): out.append(_osm_safe_list(v))
        elif isinstance(v, dict): out.append(_osm_safe_dict(v))
        else: out.append(_osm_safe_float(v, v))
    return out

def _osm_safe_dict(d: Any) -> dict:
    if not isinstance(d, dict): return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out[k] = _osm_safe_dict(v)
        elif isinstance(v, (list, tuple)): out[k] = _osm_safe_list(v)
        else: out[k] = _osm_safe_float(v, v)
    return out

def _osm_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _osm_polyline_length_m(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 2: return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(polyline[:-1], polyline[1:]):
        total += _osm_haversine(la1, lo1, la2, lo2)
    return total

def _osm_to_local_xy(polyline: Polyline) -> List[Tuple[float, float]]:
    if not polyline: return []
    lats = [p[0] for p in polyline]
    lngs = [p[1] for p in polyline]
    lat0 = sum(lats) / len(lats)
    lng0 = sum(lngs) / len(lngs)
    R = 6371000.0
    res = []
    for lat, lng in polyline:
        d_lat = math.radians(lat - lat0)
        d_lng = math.radians(lng - lng0)
        x = R * d_lng * math.cos(math.radians(lat0))
        y = R * d_lat
        res.append((x, y))
    return res

def _osm_polygon_roundness(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 3: return 0.0
    xy = _osm_to_local_xy(polyline)
    if not xy: return 0.0
    if xy[0] != xy[-1]: xy = xy + [xy[0]]
    area = 0.0
    perimeter = 0.0
    for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
        area += x1 * y2 - x2 * y1
        perimeter += math.hypot(x2 - x1, y2 - y1)
    area = abs(area) * 0.5
    if area == 0.0 or perimeter == 0.0: return 0.0
    r = 4 * math.pi * area / (perimeter ** 2)
    return float(r) if not (math.isinf(r) or math.isnan(r)) else 0.0

def _osm_edge_overlap_fraction(node_path: List[int]) -> float:
    if not node_path or len(node_path) < 2: return 0.0
    edge_counts = {}
    for u, v in zip(node_path[:-1], node_path[1:]):
        if u == v: continue
        e = (u, v) if u <= v else (v, u)
        edge_counts[e] = edge_counts.get(e, 0) + 1
    if not edge_counts: return 0.0
    overlap_edges = sum(1 for c in edge_counts.values() if c > 1)
    return overlap_edges / len(edge_counts)

def _osm_curve_penalty(node_path: List[int], G: nx.Graph) -> float:
    if len(node_path) < 3: return 0.0
    coords = {}
    for n in node_path:
        if n in coords: continue
        node = G.nodes[n]
        coords[n] = (float(node.get("y")), float(node.get("x")))
    
    penalty = 0.0
    R = 6371000.0
    
    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]; b = node_path[i]; c = node_path[i + 1]
        lat_a, lng_a = coords[a]
        lat_b, lng_b = coords[b]
        lat_c, lng_c = coords[c]
        
        # Local conversion (Inline)
        d_lat1 = math.radians(lat_a - lat_b)
        d_lng1 = math.radians(lng_a - lng_b)
        x1 = R * d_lng1 * math.cos(math.radians(lat_b))
        y1 = R * d_lat1
        
        d_lat2 = math.radians(lat_c - lat_b)
        d_lng2 = math.radians(lng_c - lng_b)
        x2 = R * d_lng2 * math.cos(math.radians(lat_b))
        y2 = R * d_lat2
        
        n1 = math.hypot(x1, y1); n2 = math.hypot(x2, y2)
        if n1 == 0 or n2 == 0: continue
        
        dot = (x1 * x2 + y1 * y2) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)
    return penalty

def _osm_path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    if not nodes or len(nodes) < 2: return 0.0
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v): return 0.0
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total

# Redzone 전역 변수 (Lazy Loading 유지)
REDZONE_POLYGONS: List[Any] = []
REDZONE_TREE: Optional[STRtree] = None
_REDZONES_LOADED = False

def _osm_load_redzones(path: str = "redzones.geojson") -> None:
    global REDZONE_POLYGONS, REDZONE_TREE, _REDZONES_LOADED
    if _REDZONES_LOADED: return
    
    _REDZONES_LOADED = True
    if not os.path.exists(path): return

    try:
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    except Exception: return

    polys = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom: continue
        try:
            shp = shape(geom)
            if shp.geom_type == "Polygon": polys.append(shp)
            elif shp.geom_type == "MultiPolygon":
                for p in shp.geoms: polys.append(p)
        except Exception: continue
    
    REDZONE_POLYGONS = polys
    if polys: REDZONE_TREE = STRtree(polys)

def _osm_is_in_redzone(lat: float, lon: float) -> bool:
    if not REDZONE_POLYGONS: return False
    pt = Point(lon, lat)
    candidates = REDZONE_TREE.query(pt) if REDZONE_TREE else REDZONE_POLYGONS
    for poly in candidates:
        if hasattr(poly, "contains") and poly.contains(pt): return True
    return False

def _osm_polyline_hits_redzone(poly: Polyline) -> bool:
    if not REDZONE_POLYGONS or not poly: return False
    step = max(1, len(poly) // 50)
    for i in range(0, len(poly), step):
        if _osm_is_in_redzone(poly[i][0], poly[i][1]): return True
    if _osm_is_in_redzone(poly[-1][0], poly[-1][1]): return True
    return False

def _osm_fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0
    d_lat = (side / 111111.0)
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))
    p = [(lat, lng), (lat + d_lat, lng), (lat + d_lat, lng + d_lng), (lat, lng + d_lng), (lat, lng)]
    return p, _osm_polyline_length_m(p), _osm_polygon_roundness(p)

def _osm_nodes_to_polyline(G: nx.MultiDiGraph, nodes: List[int]) -> Polyline:
    poly = []
    for n in nodes:
        node = G.nodes[n]
        poly.append((float(node.get("y")), float(node.get("x"))))
    return poly

def _generate_local_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """2km 미만 루프 생성기 (OSMnx)"""
    start_time = time.time()
    target_m = max(300.0, km * 1000.0)
    
    _osm_load_redzones() # Redzone 로딩
    
    meta = {
        "len": 0, "err": 0, "roundness": 0, "score": -1e18, 
        "success": False, "length_ok": False, "used_fallback": False, "message": ""
    }

    if ox is None:
        poly, length, r = _osm_fallback_square_loop(lat, lng, km)
        meta.update(len=length, score=r, used_fallback=True, message="OSMnx missing")
        meta["time_s"] = time.time() - start_time
        return poly, _osm_safe_dict(meta)

    try:
        radius_m = max(300.0, km * 600.0 + 300.0)
        G = ox.graph_from_point((lat, lng), dist=radius_m, network_type="walk", simplify=True, retain_all=False)
        UG = ox.utils_graph.get_undirected(G)
        start_node = ox.distance.nearest_nodes(UG, X=lng, Y=lat)
    except Exception as e:
        poly, length, r = _osm_fallback_square_loop(lat, lng, km)
        meta.update(len=length, score=r, used_fallback=True, message=f"Graph/Snap fail: {e}")
        meta["time_s"] = time.time() - start_time
        return poly, _osm_safe_dict(meta)

    try:
        dist_map = nx.single_source_dijkstra_path_length(UG, start_node, cutoff=target_m, weight="length")
    except Exception:
        dist_map = {}

    endpoints = [n for n, d in dist_map.items() if target_m*0.3 <= d <= target_m and n != start_node]
    endpoints = [n for n in endpoints if not _osm_is_in_redzone(float(UG.nodes[n]['y']), float(UG.nodes[n]['x']))]

    if not endpoints:
        poly, length, r = _osm_fallback_square_loop(lat, lng, km)
        meta.update(len=length, score=r, used_fallback=True, message="No endpoints")
        meta["time_s"] = time.time() - start_time
        return poly, _osm_safe_dict(meta)

    random.shuffle(endpoints)
    endpoints = endpoints[:80]
    
    best_poly = None
    best_score = -1e18
    best_stats = {}

    for u in endpoints:
        try:
            path1 = nx.shortest_path(UG, start_node, u, weight="length")
            if any(_osm_is_in_redzone(float(UG.nodes[n]['y']), float(UG.nodes[n]['x'])) for n in path1): continue
            
            for v in endpoints:
                if u == v: continue
                path2 = nx.shortest_path(UG, u, v, weight="length")
                path3 = nx.shortest_path(UG, v, start_node, weight="length")
                full_nodes = path1 + path2[1:] + path3[1:]
                
                poly = _osm_nodes_to_polyline(UG, full_nodes)
                if _osm_polyline_hits_redzone(poly): continue
                
                length_m = _osm_polyline_length_m(poly)
                err = abs(length_m - target_m)
                if err > target_m * 0.25: continue
                
                r = _osm_polygon_roundness(poly)
                ov = _osm_edge_overlap_fraction(full_nodes)
                cp = _osm_curve_penalty(full_nodes, UG)
                length_pen = err / (max(1.0, target_m * 0.05))
                
                score = (2.5 * r) - (2.0 * ov) - (0.3 * cp) - (7.0 * length_pen)
                length_ok = (err <= target_m * 0.05)
                
                if score > best_score:
                    best_score = score
                    best_poly = poly
                    best_stats = {
                        "len": length_m, "err": err, "roundness": r, "score": score, "length_ok": length_ok
                    }
        except Exception: continue

    if best_poly is None:
        poly, length, r = _osm_fallback_square_loop(lat, lng, km)
        meta.update(len=length, score=r, used_fallback=True, message="Loop gen failed")
        meta["time_s"] = time.time() - start_time
        return poly, _osm_safe_dict(meta)

    # 앵커링 (OSM 로직 전용 haversine 사용)
    if _osm_haversine(lat, lng, best_poly[0][0], best_poly[0][1]) > 1.0: best_poly.insert(0, (lat, lng))
    if _osm_haversine(lat, lng, best_poly[-1][0], best_poly[-1][1]) > 1.0: best_poly.append((lat, lng))
    
    meta.update(best_stats)
    meta["success"] = best_stats.get("length_ok", False)
    meta["message"] = "근거리 최적 루프 생성 완료"
    meta["time_s"] = time.time() - start_time
    return best_poly, _osm_safe_dict(meta)


# ============================================================
# [MAIN] 통합 진입점 (Dispatcher)
# ============================================================

def generate_running_route(
    lat: float,
    lng: float,
    km: float,
    quality_first: bool = True,
    threshold_km: float = 2.0
) -> Dict:
    """
    2km 미만: OSMnx(Local) 로직 호출
    2km 이상: Valhalla(Global) 로직 호출
    """
    if km < threshold_km:
        # [OSMnx Path]
        poly, meta = _generate_local_loop(lat, lng, km)
        dist_km = _osm_polyline_length_m(poly) / 1000.0
        return {
            "status": "ok" if meta.get("success") else "warning",
            "message": meta.get("message", "근거리 생성"),
            "start": {"lat": lat, "lng": lng},
            "polyline": [{"lat": lat, "lng": lng} for lat, lng in poly],
            "distance_km": round(dist_km, 3),
            "meta": meta,
        }
    else:
        # [Valhalla Path]
        return _generate_valhalla_loop(lat, lng, km, quality_first)
