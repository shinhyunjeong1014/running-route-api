from __future__ import annotations

import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx

try:
    import osmnx as ox
    # [설정] 캐싱 활성화 및 타임아웃 30초
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 30
except Exception:
    ox = None

LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ==========================
# JSON-safe 변환 유틸
# ==========================
def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
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
        theta = math.acos(dot)

        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    if not nodes or len(nodes) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            return 0.0
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total


# [In-place 연산] 그래프 복사 없이 가중치 수정 (속도 향상의 핵심)
def _poison_edges_inplace(G: nx.MultiGraph, path_nodes: List[int], factor: float = 8.0) -> List[Tuple[int, int, Any, float]]:
    history = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not G.has_edge(u, v):
            continue
        for key, data in G[u][v].items():
            if "length" in data:
                old_len = float(data["length"])
                history.append((u, v, key, old_len))
                data["length"] = old_len * factor
    return history


def _restore_edges_inplace(G: nx.MultiGraph, history: List[Tuple[int, int, Any, float]]):
    for u, v, key, old_len in history:
        if G.has_edge(u, v) and key in G[u][v]:
            G[u][v][key]["length"] = old_len


# ==========================
# OSM 보행자 그래프 구축
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    # [품질 보장 1] 반경을 너무 줄이지 않고 적절히 유지 (안전성 확보)
    radius_m = max(350.0, km * 250.0 + 350.0)

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


def _nodes_to_polyline(G: nx.MultiGraph, nodes: List[int]) -> Polyline:
    """
    노드 리스트를 위경도 좌표 리스트(Polyline)로 변환합니다.
    단순히 노드만 연결하는 것이 아니라, 엣지(Edge)의 형상정보(geometry)를 포함하여
    도로의 곡선이나 실제 경로를 부드럽게 표현합니다.
    """
    if not nodes:
        return []

    poly: Polyline = []
    
    # 첫 번째 노드 추가
    first_node = G.nodes[nodes[0]]
    poly.append((float(first_node['y']), float(first_node['x'])))

    for u, v in zip(nodes[:-1], nodes[1:]):
        # u -> v 엣지 데이터 가져오기
        if not G.has_edge(u, v):
            # 연결 끊김 등 예외 상황: v 노드 좌표만 직선 연결
            node_v = G.nodes[v]
            poly.append((float(node_v['y']), float(node_v['x'])))
            continue

        # MultiGraph이므로 여러 엣지 중 최적(최단 거리) 엣지 선택
        edges = G[u][v]
        # length가 가장 짧은 키를 선택 (Dijkstra가 선택했을 가능성 높음)
        best_key = min(edges, key=lambda k: edges[k].get('length', float('inf')))
        data = edges[best_key]

        if 'geometry' in data:
            # OSMnx는 geometry를 shapely.geometry.LineString으로 저장함 ((lng, lat) 순서)
            g = data['geometry']
            coords = list(g.coords)
            
            # [중요] 엣지 지오메트리 방향 보정
            # Undirected 그래프에서는 엣지가 (u, v)로 저장되어 있어도
            # 지오메트리는 v -> u 방향일 수 있음.
            # 따라서 지오메트리의 시작점이 u와 가까운지 확인해야 함.
            node_u = G.nodes[u]
            u_lng, u_lat = float(node_u['x']), float(node_u['y'])
            
            # 첫 점과 u 사이의 거리 제곱
            d_start = (coords[0][0] - u_lng)**2 + (coords[0][1] - u_lat)**2
            # 마지막 점과 u 사이의 거리 제곱
            d_end = (coords[-1][0] - u_lng)**2 + (coords[-1][1] - u_lat)**2
            
            if d_end < d_start:
                # 지오메트리가 역방향(v -> u)으로 정의된 경우 뒤집음
                coords = coords[::-1]

            # coords[0]은 u와 같으므로(혹은 매우 근접) 제외하고 추가
            for lng, lat in coords[1:]:
                poly.append((lat, lng))
        else:
            # geometry가 없으면 v 노드 좌표만 추가 (직선)
            node_v = G.nodes[v]
            poly.append((float(node_v['y']), float(node_v['x'])))

    return poly


def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
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


# ==========================
# 메인: 러닝 루프 생성기 (통합 버전)
# ==========================
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

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

    try:
        G = _build_pedestrian_graph(lat, lng, km)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC), used_fallback=True,
            message=f"OSM 그래프 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    try:
        undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat) if ox is not None else None
        if start_node is None:
            raise RuntimeError("nearest_nodes 실패")
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC), used_fallback=True,
            message=f"시작점 스냅 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    try:
        cutoff_dist = target_m * 0.8  # 0.75 -> 0.8로 약간 여유 둠
        dist_from_start: Dict[int, float] = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=cutoff_dist,
            weight="length",
        )
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC), used_fallback=True,
            message=f"탐색 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

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
            len=length, err=err, roundness=r, success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC), used_fallback=True,
            message="후보 노드 없음",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # [품질 보장 2] 후보군 5개 -> 15개로 복구
    # In-place 최적화 덕분에 15개를 검사해도 매우 빠름 (약 1~2초 소요 예상)
    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:15]

    best_score = -1e18
    best_poly: Optional[Polyline] = None
    best_stats: Dict[str, Any] = {}

    for endpoint in candidate_nodes:
        # 1. Forward 경로 탐색
        try:
            forward_nodes = nx.shortest_path(
                undirected, start_node, endpoint, weight="length"
            )
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len <= 0.0:
            continue
        if forward_len < target_m * 0.25 or forward_len > target_m * 0.8:
            continue

        # 2. In-place Poisoning (복사 비용 0)
        history = _poison_edges_inplace(undirected, forward_nodes, factor=8.0)

        # 3. Backward 경로 탐색
        try:
            back_nodes = nx.shortest_path(
                undirected, endpoint, start_node, weight="length"
            )
        except Exception:
            _restore_edges_inplace(undirected, history)
            continue
        
        # 4. 즉시 복구
        _restore_edges_inplace(undirected, history)

        back_len = _path_length_on_graph(undirected, back_nodes)
        if back_len <= 0.0:
            continue

        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        poly = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(poly)
        if length_m <= 0.0:
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
            len=length, err=err, roundness=r, success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC), used_fallback=True,
            message="경로 생성 실패 (In-place)",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    used_fallback = False

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
        used_fallback=used_fallback,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message="최적 경로 도출" if success else "인접 경로 반환",
    )
    meta["time_s"] = time.time() - start_time

    return safe_list(best_poly), safe_dict(meta)
