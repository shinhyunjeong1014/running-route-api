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
# 상수 및 환경 설정
# ----------------------------
LatLng = Tuple[float, float]
Polyline = List[LatLng]
EARTH_RADIUS_M = 6371000.0

VALHALLA_ROUTE_URL = os.getenv("VALHALLA_ROUTE_URL", "http://localhost:8002/route")
VALHALLA_LOCATE_URL = os.getenv("VALHALLA_LOCATE_URL", "http://localhost:8002/locate")
DEFAULT_COSTING = "pedestrian"

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

try:
    import osmnx as ox
except Exception:
    ox = None
    logger.warning("osmnx를 불러올 수 없습니다. 2km 이상 경로 생성에 문제가 발생할 수 있습니다.")


# ============================================================
# 📐 기본 거리/좌표 유틸리티 (두 코드 공통/정리)
# ============================================================

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 거리 (meter)."""
    R = EARTH_RADIUS_M
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _polyline_length_m(polyline: Polyline) -> float:
    """폴리라인의 총 길이 (meter)."""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(polyline[:-1], polyline[1:]):
        total += _haversine_m(la1, lo1, la2, lo2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


def _to_local_xy(points: List[LatLng]) -> List[Tuple[float, float]]:
    """위경도를 평면 좌표계로 근사 변환 (첫 점 기준)."""
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
# ⚠️ JSON-safe 변환 유틸 (두 번째 코드)
# ============================================================

def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """NaN / Inf 값을 JSON 직렬화 가능한 값으로 변환."""
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
# 🛣️ Valhalla 기반 유틸리티 (첫 번째 코드)
# ============================================================

def _decode_polyline6(encoded: str) -> List[LatLng]:
    """polyline6 디코딩."""
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
    """/locate 기반 스냅 (edges 우선)."""
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
        return (
            e.get("correlated_lat", lat),
            e.get("correlated_lon", lon),
        )

    nodes = obj.get("nodes", [])
    if nodes:
        n = nodes[0]
        return (
            n.get("lat", lat),
            n.get("lon", lon),
        )

    return (lat, lon)


def _call_valhalla_route(start: LatLng, end: LatLng, costing: str = DEFAULT_COSTING) -> Optional[List[LatLng]]:
    """Valhalla /route 호출."""
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
    """두 경로를 병합 (중복점 제거)."""
    if not out_poly:
        return back_poly[:] if back_poly else []
    if not back_poly:
        return out_poly[:]
    merged = out_poly[:]
    # 첫 점/마지막 점이 일치하면 중복 제거
    if merged[-1] == back_poly[0]:
        merged.extend(back_poly[1:])
    else:
        merged.extend(back_poly)
    return merged


def _compute_shape_jaggedness(poly: List[LatLng]) -> float:
    """Valhalla 코드의 '루프 품질 점수' 중 하나 (0에 가까울수록 부드러움)."""
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
    # 0.0 (직선) ~ 1.0 (급격한 꺾임)
    return min(1.0, avg / 180.0)


def _score_loop_valhalla(poly: List[LatLng], target_m: float) -> float:
    """Valhalla 기반의 루프 품질 점수 (높을수록 좋음)."""
    if not poly:
        return -1e9

    length_m = _polyline_length_m(poly)
    dist_err = abs(length_m - target_m) / max(target_m, 1.0)
    dist_score = 1.0 - min(dist_err, 1.0)

    jag = _compute_shape_jaggedness(poly)
    shape_score = 1.0 - jag

    # 거리 오차에 가중치 (0.7)
    return 0.7 * dist_score + 0.3 * shape_score


def _generate_pivot_candidates(
    start: LatLng,
    target_m: float,
    n_rings: int = 4,
    n_bearings: int = 16
) -> List[LatLng]:
    """pivot 후보 생성 + 연결성 필터링."""
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

            # start → pivot 경로가 실제로 존재하는 pivot만 사용
            test = _call_valhalla_route(start, snapped)
            if test is None:
                continue

            pivots.append(snapped)

    return pivots


def _build_loop_via_pivot(start: LatLng, pivot: LatLng) -> Optional[List[LatLng]]:
    """start → pivot → start 루프 생성."""
    outp = _call_valhalla_route(start, pivot)
    if not outp:
        return None

    backp = _call_valhalla_route(pivot, start)
    if not backp:
        return None

    return _merge_out_and_back(outp, backp)


def _search_best_loop_valhalla(
    start: LatLng,
    target_m: float,
    quality_first: bool = True
) -> Optional[List[LatLng]]:
    """Valhalla 기반의 최적 루프 탐색 (2km 이상)."""
    n_rings = 4 if quality_first else 2
    n_bearings = 16 if quality_first else 10

    pivots = _generate_pivot_candidates(start, target_m, n_rings, n_bearings)
    if not pivots:
        return None

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
    """단순 out-and-back fallback (Valhalla 기반)."""
    lat0, lon0 = start
    lat0r = math.radians(lat0)

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
            continue

        candidates.append((outp, length_m))

    if not candidates:
        return None

    best_out, _ = min(
        candidates,
        key=lambda t: abs(t[1] - target_m * 0.5)
    )

    back = list(reversed(best_out[:-1]))
    loop = best_out + back
    return loop


def _remove_spikes(
    poly: List[LatLng],
    angle_thresh_deg: float = 150.0,
    dist_thresh_m: float = 50.0
) -> List[LatLng]:
    """스파이크 제거."""
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

            # 로컬 XY 변환
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

            if angle > angle_thresh_deg and (a+b) < dist_thresh_m:
                # 스파이크로 판단 → p1 제거
                changed = True
            else:
                new_pts.append(p1)

        new_pts.append(pts[-1])
        pts = new_pts

    return pts


# ============================================================
# 🗺️ OSMnx/Graph 기반 유틸리티 (두 번째 코드)
# ============================================================

def _polygon_roundness(polyline: Polyline) -> float:
    """
    isoperimetric quotient 기반 원형도: 4πA / P^2
    (1에 가까울수록 원형, 0에 가까울수록 찌그러진 형태)
    """
    if not polyline or len(polyline) < 3:
        return 0.0
    # _to_local_xy는 첫 점을 기준으로 변환하므로 전체 좌표계가 로컬하게 평면 근사됨
    xy = _to_local_xy(polyline)
    if not xy:
        return 0.0
    if xy[0] != xy[-1]:
        xy = xy + [xy[0]]

    area = 0.0
    perimeter = 0.0
    for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
        area += x1 * y2 - x2 * y1
        perimeter += math.hypot(x2 - x1, y2 - y1)
    area = abs(area) * 0.5
    if area == 0.0 or perimeter == 0.0:
        return 0.0
    r = 4 * math.pi * area / (perimeter ** 2)
    if math.isinf(r) or math.isnan(r):
        return 0.0
    return float(r)


def _edge_overlap_fraction(node_path: List[int]) -> float:
    """노드 시퀀스에서 같은 간선을 여러 번 쓰는 비율 (0에 가까울수록 좋음)."""
    if not node_path or len(node_path) < 2:
        return 0.0
    edge_counts: Dict[Tuple[int, int], int] = {}
    for u, v in zip(node_path[:-1], node_path[1:]):
        if u == v:
            continue
        e = (u, v) if u <= v else (v, u)
        edge_counts[e] = edge_counts.get(e, 0) + 1
    if not edge_counts:
        return 0.0
    overlap_edges = sum(1 for c in edge_counts.values() if c > 1)
    return overlap_edges / len(edge_counts)


def _curve_penalty(node_path: List[int], G: nx.Graph) -> float:
    """연속 세 점의 각도가 너무 예리하면 페널티를 부여 (작을수록 좋음)."""
    if len(node_path) < 3:
        return 0.0

    coords: Dict[int, Tuple[float, float]] = {}
    for n in node_path:
        if n not in coords:
            node = G.nodes.get(n, {})
            coords[n] = (float(node.get("y", 0)), float(node.get("x", 0)))

    penalty = 0.0
    R = EARTH_RADIUS_M

    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        lat_a, lng_a = coords[a]
        lat_b, lng_b = coords[b]
        lat_c, lng_c = coords[c]

        def _to_xy_local(lat, lng, lat0, lng0):
            d_lat = math.radians(lat - lat0)
            d_lng = math.radians(lng - lng0)
            x = R * d_lng * math.cos(math.radians(lat0))
            y = R * d_lat
            return x, y

        x1, y1 = _to_xy_local(lat_a, lng_a, lat_b, lng_b)
        x2, y2 = _to_xy_local(lat_c, lng_c, lat_b, lng_b)

        v1x, v1y = x1, y1
        v2x, v2y = x2, y2
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0 or n2 == 0:
            continue
        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)  # 라디안

        # 60도(π/3)보다 예리한 코너에 비례하여 페널티
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    """그래프 상에서 노드 시퀀스의 길이(미터)."""
    if not nodes or len(nodes) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            # Directed graph일 경우 역방향 엣지를 찾아본다.
            if G.has_edge(v, u):
                data = next(iter(G[v][u].values()))
                total += float(data.get("length", 0.0))
            else:
                return 0.0
        else:
            data = next(iter(G[u][v].values()))
            total += float(data.get("length", 0.0))
    return total


def _apply_route_poison(G: nx.MultiGraph, path_nodes: List[int], factor: float = 8.0) -> nx.MultiGraph:
    """forward 경로의 엣지 length를 늘려서 되돌아올 때 다른 길을 쓰도록 유도."""
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not G2.has_edge(u, v):
            continue
        # u->v 엣지 poison
        for key in list(G2[u][v].keys()):
            data = G2[u][v][key]
            if "length" in data:
                data["length"] = float(data["length"]) * factor
        # v->u 엣지 poison (undirected에서 양방향 모두)
        if G2.has_edge(v, u):
            for key in list(G2[v][u].keys()):
                data = G2[v][u][key]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
    return G2


REDZONE_POLYGONS: List[Any] = []
REDZONE_TREE: Optional[STRtree] = None


def _load_redzones(path: str = "redzones.geojson") -> None:
    """redzones.geojson을 읽어 Polygon/MultiPolygon을 shapely Polygon 리스트로 로드하고 STRtree 구성."""
    global REDZONE_POLYGONS, REDZONE_TREE

    if not os.path.exists(path):
        REDZONE_POLYGONS = []
        REDZONE_TREE = None
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        REDZONE_POLYGONS = []
        REDZONE_TREE = None
        return

    features = data.get("features", [])
    polys: List[Any] = []

    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            shp = shape(geom)
        except Exception:
            continue

        if shp.geom_type == "Polygon":
            polys.append(shp)
        elif shp.geom_type == "MultiPolygon":
            for p in shp.geoms:
                polys.append(p)

    REDZONE_POLYGONS = polys
    if polys:
        REDZONE_TREE = STRtree(polys)
    else:
        REDZONE_TREE = None

_load_redzones() # import 시점에 한 번 로드


def _is_in_redzone(lat: float, lon: float) -> bool:
    """한 점이 redzone polygon 내부에 있으면 True."""
    if not REDZONE_POLYGONS:
        return False
    pt = Point(lon, lat)  # (x, y) = (lon, lat)

    if REDZONE_TREE is not None:
        try:
            candidates = REDZONE_TREE.query(pt)
        except Exception:
            candidates = REDZONE_POLYGONS
    else:
        candidates = REDZONE_POLYGONS

    for poly in candidates:
        if hasattr(poly, "contains") and poly.contains(pt):
            return True
    return False


def _polyline_hits_redzone(poly: Polyline) -> bool:
    """폴리라인 상의 점 중 하나라도 redzone 안에 들어가면 True."""
    if not REDZONE_POLYGONS or not poly:
        return False

    # 최대 50포인트 정도만 검사
    step = max(1, len(poly) // 50)
    for i in range(0, len(poly), step):
        lat, lon = poly[i]
        if _is_in_redzone(lat, lon):
            return True

    # 마지막 점도 한 번 더 확인
    lat, lon = poly[-1]
    if _is_in_redzone(lat, lon):
        return True

    return False


def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """OSMnx 'walk' 네트워크 타입만 사용하여 안정적인 보행자 그래프를 생성."""
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    radius_m = max(700.0, km * 500.0 + 700.0)

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )
    if not G.nodes:
        raise RuntimeError("OSM 보행자 네트워크를 생성하지 못했습니다.")
    return G


def _nodes_to_polyline(G: nx.MultiDiGraph, nodes: List[int]) -> Polyline:
    """그래프 노드 시퀀스를 위경도 폴리라인으로 변환."""
    poly: Polyline = []
    for n in nodes:
        node = G.nodes[n]
        lat = float(node.get("y"))
        lng = float(node.get("x"))
        poly.append((lat, lng))
    return poly


def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """OSM/그래프를 전혀 쓰지 못할 때 사용하는 매우 단순한 정사각형 루프."""
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0

    d_lat = (side / 111111.0)
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))

    a = (lat + d_lat, lng)
    b = (lat + d_lat, lng + d_lng)
    c = (lat,        lng + d_lng)
    d = (lat,        lng)

    poly: Polyline = [d, a, b, c, d]
    length = _polyline_length_m(poly)
    r = _polygon_roundness(poly)
    return poly, length, r


# ============================================================
# 🚶‍♂️ 2km 미만: Local Loop Builder (OSMnx/Graph 기반)
# ============================================================
def _generate_local_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """1.8km 이하 요청 시 사용하는 '근거리 루프 생성기'."""

    start_time = time.time()
    target_m = max(300.0, km * 1000.0)

    # 튜닝 상수 (두 번째 코드에서 가져옴)
    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_WEIGHT = 0.3
    LENGTH_TOL_FRAC = 0.05   # ±5%
    HARD_ERR_FRAC = 0.25     # ±25%는 폐기
    LEN_PEN_WEIGHT = 7.0
    MAX_ENDPOINTS = 80
    CUTOFF_FACTOR = 0.8

    meta: Dict[str, Any] = dict(
        len=0, err=0, roundness=0, overlap=0, curve_penalty=0,
        score=-1e18, success=False, length_ok=False, used_fallback=False,
        routes_checked=0, routes_validated=0,
        km_requested=km, target_m=target_m,
        time_s=0.0, message=""
    )

    # 1) 보행자 그래프 (근거리)
    try:
        radius_m = max(300.0, km * 600.0 + 300.0)
        if ox is None:
            raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

        G = ox.graph_from_point(
            (lat, lng),
            dist=radius_m,
            network_type="walk",
            simplify=True,
            retain_all=False,
        )
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length - target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message=f"local graph load 실패: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return poly, _safe_dict(meta)

    if not G.nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length - target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message="local graph empty"
        )
        meta["time_s"] = time.time() - start_time
        return poly, _safe_dict(meta)

    try:
        UG = ox.utils_graph.get_undirected(G)
    except Exception:
        UG = G.to_undirected()

    # 2) start node
    try:
        start_node = ox.distance.nearest_nodes(UG, X=lng, Y=lat)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length - target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message=f"local start snap 실패: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return poly, _safe_dict(meta)

    # 3) start에서 Dijkstra
    try:
        dist_map = nx.single_source_dijkstra_path_length(
            UG, start_node,
            cutoff=max(300.0, target_m * CUTOFF_FACTOR),
            weight="length"
        )
    except Exception:
        dist_map = {}

    if not dist_map:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length - target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message="local dijkstra empty"
        )
        meta["time_s"] = time.time() - start_time
        return poly, _safe_dict(meta)

    min_forward = target_m * 0.3
    max_forward = target_m * 1.0

    endpoints = [n for n, d in dist_map.items()
                 if min_forward <= d <= max_forward and n != start_node]

    # redzone 제외
    filtered = []
    for n in endpoints:
        la_n = float(UG.nodes[n]["y"])
        lo_n = float(UG.nodes[n]["x"])
        if not _is_in_redzone(la_n, lo_n):
            filtered.append(n)
    endpoints = filtered

    if len(endpoints) == 0:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length - target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message="local endpoints 없음"
        )
        meta["time_s"] = time.time() - start_time
        return poly, _safe_dict(meta)

    random.shuffle(endpoints)
    endpoints = endpoints[:MAX_ENDPOINTS]

    best_poly: Optional[Polyline] = None
    best_score = -1e18
    best_stats: Dict[str, Any] = {}

    # 4) u, v 쌍을 이용해 loop 생성
    for u in endpoints:
        try:
            path1 = nx.shortest_path(UG, start_node, u, weight="length")
        except Exception:
            continue

        # path1 redzone 체크
        skip = False
        for n in path1:
            la_n = float(UG.nodes[n]["y"])
            lo_n = float(UG.nodes[n]["x"])
            if _is_in_redzone(la_n, lo_n):
                skip = True
                break
        if skip:
            continue

        path1_len = _path_length_on_graph(UG, path1)
        if path1_len <= 0:
            continue

        for v in endpoints:
            if u == v:
                continue
            try:
                path2 = nx.shortest_path(UG, u, v, weight="length")
                path3 = nx.shortest_path(UG, v, start_node, weight="length")
            except Exception:
                continue

            full_nodes = path1 + path2[1:] + path3[1:]
            meta["routes_checked"] += 1

            poly = _nodes_to_polyline(UG, full_nodes)
            length_m = _polyline_length_m(poly)
            if length_m <= 0:
                continue

            # polyline 레드존 검사
            if _polyline_hits_redzone(poly):
                continue

            err = abs(length_m - target_m)
            if err > target_m * HARD_ERR_FRAC:
                continue

            r = _polygon_roundness(poly)
            ov = _edge_overlap_fraction(full_nodes)
            cp = _curve_penalty(full_nodes, UG)

            length_pen = err / (max(1.0, target_m * LENGTH_TOL_FRAC))
            score = (
                ROUNDNESS_WEIGHT * r
                - OVERLAP_PENALTY * ov
                - CURVE_WEIGHT * cp
                - LEN_PEN_WEIGHT * length_pen
            )

            length_ok = (err <= target_m * LENGTH_TOL_FRAC)
            if length_ok:
                meta["routes_validated"] += 1

            if score > best_score:
                best_score = score
                best_poly = poly
                best_stats = dict(
                    len=length_m, err=err, roundness=r,
                    overlap=ov, curve_penalty=cp,
                    score=score, length_ok=length_ok
                )

    if best_poly is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        best_stats = dict(
            len=length, err=abs(length - target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, length_ok=False
        )
        meta.update(best_stats)
        meta["used_fallback"] = True
        meta["message"] = "local loop 생성 실패(fallback)"
        meta["time_s"] = time.time() - start_time
        return poly, _safe_dict(meta)

    # 시작점 앵커링
    first_la, first_lo = best_poly[0]
    if _haversine_m(lat, lng, first_la, first_lo) > 1.0:
        best_poly.insert(0, (lat, lng))

    last_la, last_lo = best_poly[-1]
    if _haversine_m(lat, lng, last_la, last_lo) > 1.0:
        best_poly.append((lat, lng))

    length2 = _polyline_length_m(best_poly)
    err2 = abs(length2 - target_m)
    best_stats["len"] = length2
    best_stats["err"] = err2
    best_stats["length_ok"] = (err2 <= target_m * LENGTH_TOL_FRAC)

    meta.update(best_stats)
    meta["success"] = best_stats["length_ok"]
    meta["message"] = "근거리 최적 루프 생성 완료"
    meta["time_s"] = time.time() - start_time

    return best_poly, _safe_dict(meta)


# ============================================================
# 🚗 2km 이상: Valhalla 기반 (장거리)
# ============================================================
def _generate_valhalla_loop(lat: float, lng: float, km: float, quality_first: bool = True) -> Dict:
    """Valhalla 기반의 장거리 루프 생성기 (2km 이상)."""

    start_time = time.time()
    start = (float(lat), float(lng))
    target_m = max(float(km), 0.5) * 1000.0

    # 1) pivot 기반 고품질 루프 시도
    loop = _search_best_loop_valhalla(start, target_m, quality_first)
    if loop:
        msg = "고품질 러닝 루프 생성 완료 (Valhalla/Pivot)"
    else:
        # 2) 실패 시: 단순 out-and-back 루프 생성 (성공률 매우 높음)
        loop = _build_simple_out_and_back(start, target_m)
        if loop:
            msg = "안전한 단순 왕복 루프로 루트를 생성했습니다. (Valhalla/Out-and-Back)"
        else:
            return {
                "status": "error",
                "message": "루프 생성 실패 (Valhalla 경로 없음)",
                "start": {"lat": lat, "lng": lng},
                "polyline": [],
                "distance_km": 0.0,
            }

    # 스파이크 제거
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
# 🏁 Public API: 2km 기준 분기
# ============================================================
def generate_running_route(
    lat: float,
    lng: float,
    km: float,
    quality_first: bool = True,
    threshold_km: float = 2.0
) -> Dict:
    """
    요청 좌표와 목표 거리를 기반으로 러닝 루프를 생성합니다.
    - km < threshold_km (기본 2.0km): OSMnx/Graph 기반 (근거리 최적화)
    - km >= threshold_km (기본 2.0km): Valhalla 기반 (장거리/안정성 최적화)
    """
    if km < threshold_km:
        # 2km 미만: OSMnx/Graph 기반
        poly, meta = _generate_local_loop(lat, lng, km)
        dist_km = polyline_length_m(poly) / 1000.0
        return {
            "status": "ok" if meta.get("success") else "warning",
            "message": meta.get("message", "근거리 루프 생성 완료"),
            "start": {"lat": lat, "lng": lng},
            "polyline": [{"lat": a, "lng": b} for (a,b) in poly],
            "distance_km": round(dist_km, 3),
            "meta": meta,
        }
    else:
        # 2km 이상: Valhalla 기반
        return _generate_valhalla_loop(lat, lng, km, quality_first)

# ----------------------------
# ⚠️ 주의: 두 번째 코드의 generate_area_loop 함수는
# 2km 미만과 초과 로직을 모두 포함하고 있으나,
# 요청사항에 따라 Valhalla 기반의 generate_running_route에 통합합니다.
# ----------------------------

# ----------------------------
# ⚠️ 참고: 두 번째 코드에서 사용된 OSMnx 기반의 2km 초과 로직 (rod/poisoning)은
# 첫 번째 코드의 Valhalla 기반 장거리 로직으로 대체되었습니다.
# 따라서 _generate_area_loop 내의 1.8km 초과 로직은 제거되었습니다.
# ----------------------------
