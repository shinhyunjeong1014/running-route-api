from __future__ import annotations

import math
import os
import time
import json
import random
import logging
from typing import List, Tuple, Dict, Any, Optional

import requests
import networkx as nx

try:
    import osmnx as ox
except Exception:
    ox = None

from shapely.geometry import shape, Point
from shapely.strtree import STRtree

# -----------------------------------
# 공통 타입 정의
# -----------------------------------
LatLng = Tuple[float, float]
Polyline = List[LatLng]

EARTH_RADIUS_M = 6371000.0

# Valhalla 엔드포인트
VALHALLA_ROUTE_URL = os.getenv("VALHALLA_ROUTE_URL", "http://localhost:8002/route")
VALHALLA_LOCATE_URL = os.getenv("VALHALLA_LOCATE_URL", "http://localhost:8002/locate")

DEFAULT_COSTING = "pedestrian"

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)


# ==========================
# JSON-safe 변환 유틸
# ==========================
def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """NaN / Inf 값을 JSON 직렬화 가능한 값으로 변환."""
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return default
    return x


def safe_list(lst: Any) -> list:
    if not isinstance(lst, (list, tuple)):
        return []
    out = []
    for v in lst:
        if isinstance(v, (list, tuple)):
            out.append(safe_list(v))
        elif isinstance(v, dict):
            out.append(safe_dict(v))
        else:
            out.append(safe_float(v, v))
    return out


def safe_dict(d: Any) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = safe_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = safe_list(v)
        else:
            out[k] = safe_float(v, v)
    return out


# ==========================
# 거리 / 길이 유틸
# ==========================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def polyline_length_m(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(la1, lo1, la2, lo2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


def _to_local_xy(polyline: Polyline) -> List[Tuple[float, float]]:
    """위경도를 평면 좌표계로 근사 변환."""
    if not polyline:
        return []
    lats = [p[0] for p in polyline]
    lngs = [p[1] for p in polyline]
    lat0 = sum(lats) / len(lats)
    lng0 = sum(lngs) / len(lngs)
    R = EARTH_RADIUS_M
    res = []
    for lat, lng in polyline:
        d_lat = math.radians(lat - lat0)
        d_lng = math.radians(lng - lng0)
        x = R * d_lng * math.cos(math.radians(lat0))
        y = R * d_lat
        res.append((x, y))
    return res


def polygon_roundness(polyline: Polyline) -> float:
    """
    isoperimetric quotient 기반 원형도: 4πA / P^2
    (1에 가까울수록 원형, 0에 가까울수록 찌그러진 형태)
    """
    if not polyline or len(polyline) < 3:
        return 0.0
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


# ==========================
# 그래프 기반 부가 유틸
# ==========================
def _edge_overlap_fraction(node_path: List[int]) -> float:
    """
    노드 시퀀스에서 같은 간선을 여러 번 쓰는 비율.
    (0에 가까울수록 더 '한 번씩만' 지나는 좋은 루프)
    """
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
    """
    연속 세 점의 각도가 너무 예리하면 페널티를 부여.
    러너가 꺾어야 하는 '급코너' 개념을 근사.
    """
    if len(node_path) < 3:
        return 0.0

    coords: Dict[int, Tuple[float, float]] = {}
    for n in node_path:
        if n in coords:
            continue
        node = G.nodes[n]
        coords[n] = (float(node.get("y")), float(node.get("x")))

    penalty = 0.0
    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        lat_a, lng_a = coords[a]
        lat_b, lng_b = coords[b]
        lat_c, lng_c = coords[c]

        R = EARTH_RADIUS_M

        def _to_xy(lat, lng, lat0, lng0):
            d_lat = math.radians(lat - lat0)
            d_lng = math.radians(lng - lng0)
            x = R * d_lng * math.cos(math.radians(lat0))
            y = R * d_lat
            return x, y

        x1, y1 = _to_xy(lat_a, lng_a, lat_b, lng_b)
        x2, y2 = _to_xy(lat_c, lng_c, lat_b, lng_b)

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
            return 0.0
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total


# ==========================
# fallback: 기하학적 사각형 루프
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """
    OSM/Valhalla 모두 실패 시 사용하는 매우 단순한 정사각형 루프.
    """
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0

    # 위도 1m ≈ 1/111111 deg
    d_lat = (side / 111111.0)
    # 경도 1m ≈ 1/(111111 cos φ) deg
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))

    a = (lat + d_lat, lng)
    b = (lat + d_lat, lng + d_lng)
    c = (lat,        lng + d_lng)
    d = (lat,        lng)

    poly: Polyline = [d, a, b, c, d]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# redzones.geojson 로딩 + R-tree
# ==========================
REDZONE_POLYGONS: List[Any] = []
REDZONE_TREE: Optional[STRtree] = None


def load_redzones(path: str = "redzones.geojson") -> None:
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


# import 시점에 한 번 로드
load_redzones()


def is_in_redzone(lat: float, lon: float) -> bool:
    """한 점이 redzone polygon 내부에 있으면 True."""
    if not REDZONE_POLYGONS:
        return False
    pt = Point(lon, lat)  # (x, y) = (lon, lat)

    # R-tree로 후보 좁히기
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


def polyline_hits_redzone(poly: Polyline) -> bool:
    """
    폴리라인 상의 점 중 하나라도 redzone 안에 들어가면 True.
    (성능을 위해 모든 점이 아닌 일정 간격으로 샘플링)
    """
    if not REDZONE_POLYGONS or not poly:
        return False

    step = max(1, len(poly) // 50)  # 최대 50포인트 정도만 검사
    for i in range(0, len(poly), step):
        lat, lon = poly[i]
        if is_in_redzone(lat, lon):
            return True

    lat, lon = poly[-1]
    if is_in_redzone(lat, lon):
        return True

    return False


# ==========================
# OSM 보행자 그래프 구축 (근거리 전용)
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    OSMnx 'walk' 네트워크 타입만 사용하여
    안정적인 보행자 그래프를 생성.
    """
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
    poly: Polyline = []
    for n in nodes:
        node = G.nodes[n]
        lat = float(node.get("y"))
        lng = float(node.get("x"))
        poly.append((lat, lng))
    return poly


# ============================================================
# 1.8km 이하 전용 Local Loop Builder (OSM + redzone)
# ============================================================
def _generate_local_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    1.8km 이하 요청 시 사용하는 '근거리 루프 생성기'.
    - OSM 보행자 그래프 기반
    - redzone 완전 회피
    """

    start_time = time.time()
    target_m = max(300.0, km * 1000.0)

    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_WEIGHT = 0.3
    LENGTH_TOL_FRAC = 0.05   # ±5%
    HARD_ERR_FRAC = 0.25     # ±25%는 폐기
    LEN_PEN_WEIGHT = 7.0

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
        return poly, safe_dict(meta)

    if not G.nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length - target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message="local graph empty"
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

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
        return poly, safe_dict(meta)

    # 3) start에서 Dijkstra
    try:
        dist_map = nx.single_source_dijkstra_path_length(
            UG, start_node,
            cutoff=max(300.0, target_m * 0.8),
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
        return poly, safe_dict(meta)

    min_forward = target_m * 0.3
    max_forward = target_m * 1.0

    endpoints = [n for n, d in dist_map.items()
                 if min_forward <= d <= max_forward and n != start_node]

    # redzone 제외
    filtered = []
    for n in endpoints:
        la_n = float(UG.nodes[n]["y"])
        lo_n = float(UG.nodes[n]["x"])
        if not is_in_redzone(la_n, lo_n):
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
        return poly, safe_dict(meta)

    random.shuffle(endpoints)
    endpoints = endpoints[:80]

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
            if is_in_redzone(la_n, lo_n):
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
            length_m = polyline_length_m(poly)
            if length_m <= 0:
                continue

            # polyline 레드존 검사
            if polyline_hits_redzone(poly):
                continue

            err = abs(length_m - target_m)
            if err > target_m * HARD_ERR_FRAC:
                continue

            r = polygon_roundness(poly)
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
        return poly, safe_dict(meta)

    # 시작점 앵커링
    first_la, first_lo = best_poly[0]
    if haversine(lat, lng, first_la, first_lo) > 1.0:
        best_poly.insert(0, (lat, lng))

    last_la, last_lo = best_poly[-1]
    if haversine(lat, lng, last_la, last_lo) > 1.0:
        best_poly.append((lat, lng))

    length2 = polyline_length_m(best_poly)
    err2 = abs(length2 - target_m)
    best_stats["len"] = length2
    best_stats["err"] = err2
    best_stats["length_ok"] = (err2 <= target_m * LENGTH_TOL_FRAC)

    meta.update(best_stats)
    meta["success"] = best_stats["length_ok"]
    meta["message"] = "근거리 최적 루프 생성 완료"
    meta["time_s"] = time.time() - start_time

    return best_poly, safe_dict(meta)


# ============================================================
# Valhalla 기반 루프 생성기 (2km 이상 전용)
#  - 예전에 잘 되던 코드 + timeout/피벗 개수 조정
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

        coords.append((lat / 1e6, lng / 1e6))

    return coords


def _snap_to_road(lat: float, lon: float) -> LatLng:
    """
    Valhalla /locate 기반 스냅.
    timeout을 5초로 늘려 과부하 상황에서도 pivot 발생률을 올림.
    """
    try:
        resp = requests.post(
            VALHALLA_LOCATE_URL,
            json={"locations": [{"lat": lat, "lon": lon}]},
            timeout=5.0,  # (기존 1.5초 → 5초)
        )
    except Exception as e:
        logger.warning(f"Valhalla locate exception: {e}")
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
    payload = {
        "locations": [
            {"lat": start[0], "lon": start[1]},
            {"lat": end[0], "lon": end[1]},
        ],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }

    try:
        resp = requests.post(VALHALLA_ROUTE_URL, json=payload, timeout=15.0)  # (기존 8초 → 15초)
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

    shape = trip.get("shape")
    if isinstance(shape, str):
        poly = _decode_polyline6(shape)
    elif isinstance(shape, list):
        poly = [(p["lat"], p["lon"]) for p in shape]
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
    """
    평균 각도로 지그재그 정도를 측정.
    0에 가까울수록 부드러운 루프, 1에 가까울수록 지그재그 심함.
    """
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

        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        ang = math.degrees(math.acos(dot))

        total_angle += ang
        count += 1

    if count == 0:
        return 0.0

    avg = total_angle / count
    return min(1.0, avg / 180.0)


def _score_valhalla_loop(poly: List[LatLng], target_m: float) -> float:
    """
    Valhalla 기반 루프 스코어:
      - 거리 정확도 0.7
      - 모양 부드러움(지그재그 적음) 0.3
    """
    if not poly:
        return -1e9

    length_m = polyline_length_m(poly)
    dist_err = abs(length_m - target_m) / max(target_m, 1.0)
    dist_score = 1.0 - min(dist_err, 1.0)

    jag = _compute_shape_jaggedness(poly)
    shape_score = 1.0 - jag

    return 0.7 * dist_score + 0.3 * shape_score


def _generate_pivot_candidates(
    start: LatLng,
    target_m: float,
    n_rings: int = 2,
    n_bearings: int = 12,
) -> List[LatLng]:
    """
    pivot 후보 생성:
      - rings: 2
      - bearings: 12 (총 24개)
      - 각 pivot은 /locate + /route 로 실제 연결 가능한 것만 채택
    """
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
    outp = _call_valhalla_route(start, pivot)
    if not outp:
        return None

    backp = _call_valhalla_route(pivot, start)
    if not backp:
        return None

    return _merge_out_and_back(outp, backp)


def _search_best_valhalla_loop(
    start: LatLng,
    target_m: float,
) -> Optional[List[LatLng]]:
    """
    Valhalla pivot 기반 고품질 루프 탐색.
    """
    pivots = _generate_pivot_candidates(start, target_m, n_rings=2, n_bearings=12)
    if not pivots:
        return None

    best_poly: Optional[List[LatLng]] = None
    best_score = -1e9

    for pivot in pivots:
        loop = _build_loop_via_pivot(start, pivot)
        if not loop:
            continue

        score = _score_valhalla_loop(loop, target_m)
        if score > best_score:
            best_score = score
            best_poly = loop

    return best_poly


def _build_simple_out_and_back(start: LatLng, target_m: float) -> Optional[List[LatLng]]:
    """
    pivot 실패 시 사용되는 안전한 out-and-back 루프 (여전히 Valhalla 사용).
    """
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

        length_m = polyline_length_m(outp)
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
    """
    불필요하게 톡 튀어나온 스파이크 구간 제거.
    """
    if len(poly) < 5:
        return poly

    pts = poly[:]
    changed = True

    while changed:
        changed = False
        new_pts: List[LatLng] = [pts[0]]

        for i in range(1, len(pts) - 1):
            p0 = new_pts[-1]
            p1 = pts[i]
            p2 = pts[i + 1]

            a = haversine(p1[0], p1[1], p0[0], p0[1])
            b = haversine(p1[0], p1[1], p2[0], p2[1])

            if a < 1e-3 or b < 1e-3:
                new_pts.append(p1)
                continue

            xy = _to_local_xy([p0, p1, p2])
            (x0, y0), (x1, y1), (x2, y2) = xy
            v1 = (x0 - x1, y0 - y1)
            v2 = (x2 - x1, y2 - y1)
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)

            if n1 < 1e-6 or n2 < 1e-6:
                new_pts.append(p1)
                continue

            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))

            if angle > angle_thresh_deg and (a + b) < dist_thresh_m:
                # 스파이크로 판단 → p1 제거
                changed = True
            else:
                new_pts.append(p1)

        new_pts.append(pts[-1])
        pts = new_pts

    return pts


def _generate_valhalla_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    2km 이상 요청 시 사용하는 Valhalla 기반 러닝 루프 생성기.
    - pivot 기반 고품질 루프 + out-and-back fallback
    - 마지막까지 실패 시에만 사각형 fallback 사용.
    """
    start_time = time.time()
    start = (float(lat), float(lng))
    target_m = max(float(km), 0.5) * 1000.0
    LENGTH_TOL_FRAC = 0.05

    meta: Dict[str, Any] = {
        "len": 0.0,
        "err": 0.0,
        "roundness": 0.0,
        "overlap": 0.0,
        "curve_penalty": 0.0,
        "score": 0.0,
        "success": False,
        "length_ok": False,
        "used_fallback": False,
        "km_requested": km,
        "target_m": target_m,
        "routes_checked": 0,
        "routes_validated": 0,
        "valhalla_calls": 0,
        "kakao_calls": 0,
        "time_s": 0.0,
        "message": "",
    }

    # 1) pivot 기반 고품질 루프
    loop = _search_best_valhalla_loop(start, target_m)
    used_fallback = False
    msg = ""

    if loop:
        msg = "Valhalla 피벗 기반 고품질 러닝 루프 생성 완료"
    else:
        # 2) out-and-back fallback
        loop = _build_simple_out_and_back(start, target_m)
        if loop:
            used_fallback = True
            msg = "Valhalla 단순 왕복 루프로 러닝 루트를 생성했습니다."
        else:
            # 3) 최종 실패: 사각형 fallback
            poly, length, r = _fallback_square_loop(lat, lng, km)
            err = abs(length - target_m)
            meta.update(
                len=length,
                err=err,
                roundness=r,
                overlap=0.0,
                curve_penalty=0.0,
                score=r,
                success=False,
                length_ok=(err <= target_m * LENGTH_TOL_FRAC),
                used_fallback=True,
                message="Valhalla 루프 생성 실패 (fallback 사각형)",
            )
            meta["time_s"] = time.time() - start_time
            return poly, safe_dict(meta)

    # 스파이크 제거
    loop = _remove_spikes(loop)

    length = polyline_length_m(loop)
    err = abs(length - target_m)
    roundness = polygon_roundness(loop)
    score = roundness  # 간단히 roundness를 점수로 사용
    length_ok = (err <= target_m * LENGTH_TOL_FRAC)

    meta.update(
        len=length,
        err=err,
        roundness=roundness,
        overlap=0.0,
        curve_penalty=0.0,
        score=score,
        success=length_ok,
        length_ok=length_ok,
        used_fallback=used_fallback,
        message=msg,
    )
    meta["time_s"] = time.time() - start_time

    return loop, safe_dict(meta)


# ============================================================
# Public API: generate_area_loop
#  - app.py에서 이 함수를 호출한다고 가정
# ============================================================
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    요청 좌표(lat, lng)와 목표 거리(km)를 기반으로 러닝 루프를 생성한다.

    - km <= 1.8  : OSMnx + redzone 근거리 루프 (_generate_local_loop)
    - km >  1.8  : Valhalla pivot + out-and-back 기반 루프 (_generate_valhalla_loop)
    """
    if km <= 1.8:
        poly, meta = _generate_local_loop(lat, lng, km)
        return safe_list(poly), safe_dict(meta)
    else:
        poly, meta = _generate_valhalla_loop(lat, lng, km)
        return safe_list(poly), safe_dict(meta)