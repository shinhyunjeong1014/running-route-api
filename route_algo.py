from __future__ import annotations

import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx

try:
    import osmnx as ox
except Exception:
    ox = None

from shapely.geometry import shape, Point
from shapely.strtree import STRtree
import json
import os

LatLng = Tuple[float, float]
Polyline = List[LatLng]


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
    R = 6371000.0
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


# ==========================
# roundness / overlap / 곡률 페널티
# ==========================
def _to_local_xy(polyline: Polyline) -> List[Tuple[float, float]]:
    """위경도를 평면 좌표계로 근사 변환."""
    if not polyline:
        return []
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

        R = 6371000.0

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


def _apply_route_poison(G: nx.MultiGraph, path_nodes: List[int], factor: float = 8.0) -> nx.MultiGraph:
    """
    forward 경로의 엣지 length를 늘려서
    되돌아올 때는 가급적 다른 길을 쓰도록 유도.
    (factor가 클수록 '다른 길'을 더 강하게 선호)
    """
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not G2.has_edge(u, v):
            continue
        for key in list(G2[u][v].keys()):
            data = G2[u][v][key]
            if "length" in data:
                data["length"] = float(data["length"]) * factor
        if G2.has_edge(v, u):
            for key in list(G2[v][u].keys()):
                data = G2[v][u][key]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
    return G2


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
        # 방어적 체크: poly가 Polygon 객체인지 확인
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

    # 마지막 점도 한 번 더 확인
    lat, lon = poly[-1]
    if is_in_redzone(lat, lon):
        return True

    return False


# ==========================
# OSM 보행자 그래프 구축
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


# ==========================
# fallback: 기하학적 사각형 루프
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """
    OSM/그래프를 전혀 쓰지 못할 때 사용하는 매우 단순한 정사각형 루프.
    - 실제 도로망과 맞지 않을 수 있지만, API가 완전히 죽었을 때의 최후 수단.
    """
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0

    d_lat = (side / 111111.0)
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))

    a = (lat + d_lat, lng)
    b = (lat + d_lat, lng + d_lng)
    c = (lat,        lng + d_lng)
    d = (lat,        lng)

    poly: Polyline = [d, a, b, c, d]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ============================================================
# 1.8km 이하 전용 Local Loop Builder
# ============================================================
def _generate_local_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    1.8km 이하 요청 시 사용하는 '근거리 루프 생성기'.
    - rod/poisoning 사용 안함
    - 반경 r 내의 subgraph에서 모든 노드-노드 루프 탐색
    - roundness / overlap / curve_penalty 기반 최적 루프 선택
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
# 메인: 러닝 루프 생성기 (1.8km 이하 / 이상 통합)
# ============================================================
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    요청 좌표(lat, lng)와 목표 거리(km)를 기반으로
    '요청거리 정확도'와 '루프 모양'을 동시에 고려한 러닝 루프를 생성한다.

    - km <= 1.8  : 근거리 Local Loop Builder (_generate_local_loop)
    - km >  1.8  : rod + poisoning 기반 루프
    """

    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

    # 1.8km 이하: 근거리 루프 전용
    if km <= 1.8:
        poly, meta = _generate_local_loop(lat, lng, km)
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # 1.8km 초과: 기존 알고리즘
    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3

    LENGTH_TOL_FRAC = 0.05
    HARD_ERR_FRAC = 0.30
    LENGTH_PENALTY_WEIGHT = 8.0

    meta: Dict[str, Any] = {
        "len": 0.0,
        "err": 0.0,
        "roundness": 0.0,
        "overlap": 0.0,
        "curve_penalty": 0.0,
        "score": -1e18,
        "success": False,
        "length_ok": False,
        "used_fallback": False,
        "valhalla_calls": 0,
        "kakao_calls": 0,
        "routes_checked": 0,
        "routes_validated": 0,
        "km_requested": km,
        "target_m": target_m,
        "time_s": 0.0,
        "message": "",
    }

    # 1) 보행자 그래프
    try:
        G = _build_pedestrian_graph(lat, lng, km)
    except Exception as e:
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
            message=f"OSM 보행자 그래프 생성 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    try:
        undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    # 1-1) redzone 노드 제거
    remove_nodes = []
    for n, data in list(undirected.nodes(data=True)):
        la = float(data.get("y"))
        lo = float(data.get("x"))
        if is_in_redzone(la, lo):
            remove_nodes.append(n)
    if remove_nodes:
        undirected.remove_nodes_from(remove_nodes)

    if undirected.number_of_nodes() == 0:
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
            message="redzone 필터링 후 사용 가능한 노드가 없어 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # 2) start node
    try:
        start_node = ox.distance.nearest_nodes(undirected, X=lng, Y=lat)
    except Exception as e:
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
            message=f"시작 좌표를 그래프에 스냅하지 못해 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # 3) start에서 rod endpoint 후보 탐색
    try:
        dist_from_start: Dict[int, float] = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.8,
            weight="length",
        )
    except Exception as e:
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
            message=f"그래프 최단거리 탐색 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    rod_target = target_m / 2.0
    rod_min = rod_target * 0.6
    rod_max = rod_target * 1.4

    candidate_nodes = [
        n for n, d in dist_from_start.items()
        if rod_min <= d <= rod_max and n != start_node
    ]

    if len(candidate_nodes) < 5:
        lo = target_m * 0.25
        hi = target_m * 0.75
        candidate_nodes = [
            n for n, d in dist_from_start.items()
            if lo <= d <= hi and n != start_node
        ]

    if not candidate_nodes:
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
            message="적절한 rod endpoint 후보를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:120]

    best_score = -1e18
    best_poly: Optional[Polyline] = None
    best_stats: Dict[str, Any] = {}

    # 4) forward + poisoned backward 루프 생성
    for endpoint in candidate_nodes:
        try:
            forward_nodes = nx.shortest_path(
                undirected,
                start_node,
                endpoint,
                weight="length",
            )
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len <= 0.0:
            continue

        if forward_len < target_m * 0.25 or forward_len > target_m * 0.8:
            continue

        poisoned = _apply_route_poison(undirected, forward_nodes, factor=8.0)

        try:
            back_nodes = nx.shortest_path(
                poisoned,
                endpoint,
                start_node,
                weight="length",
            )
        except Exception:
            continue

        back_len = _path_length_on_graph(undirected, back_nodes)
        if back_len <= 0.0:
            continue

        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        poly = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(poly)
        if length_m <= 0.0:
            continue

        if polyline_hits_redzone(poly):
            continue

        err = abs(length_m - target_m)
        if err > target_m * HARD_ERR_FRAC:
            continue

        r = polygon_roundness(poly)
        ov = _edge_overlap_fraction(full_nodes)
        cp = _curve_penalty(full_nodes, undirected)

        length_pen = err / (max(1.0, target_m * LENGTH_TOL_FRAC))

        score = (
            ROUNDNESS_WEIGHT * r
            - OVERLAP_PENALTY * ov
            - CURVE_PENALTY_WEIGHT * cp
            - LENGTH_PENALTY_WEIGHT * length_pen
        )

        length_ok = err <= target_m * LENGTH_TOL_FRAC
        if length_ok:
            meta["routes_validated"] += 1

        if score > best_score:
            best_score = score
            best_poly = poly
            best_stats = {
                "len": length_m,
                "err": err,
                "roundness": r,
                "overlap": ov,
                "curve_penalty": cp,
                "score": score,
                "length_ok": length_ok,
            }

    if best_poly is None:
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
            message="논문 기반 OSM 루프 생성에 실패하여 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # 6) 시작 좌표 앵커링
    if best_poly:
        first_lat, first_lng = best_poly[0]
        if haversine(lat, lng, first_lat, first_lng) > 1.0:
            best_poly.insert(0, (lat, lng))

        last_lat, last_lng = best_poly[-1]
        if haversine(lat, lng, last_lat, last_lng) > 1.0:
            best_poly.append((lat, lng))

        length2 = polyline_length_m(best_poly)
        err2 = abs(length2 - target_m)
        length_ok2 = err2 <= target_m * LENGTH_TOL_FRAC

        best_stats["len"] = length2
        best_stats["err"] = err2
        best_stats["length_ok"] = length_ok2

    success = bool(best_stats.get("length_ok"))

    meta.update(best_stats)
    meta.update(
        success=success,
        used_fallback=False,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else f"요청 오차(±{int(target_m * LENGTH_TOL_FRAC)}m)를 초과하지만, 가장 인접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    return best_poly, safe_dict(meta)