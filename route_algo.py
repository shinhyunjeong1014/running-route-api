import math
import random
import time
from typing import List, Tuple, Dict, Any

import networkx as nx
import osmnx as ox


# ==========================
# JSON-safe 변환 유틸 (유지)
# ==========================
def safe_float(x: Any):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def safe_list(lst):
    out = []
    for v in lst:
        if isinstance(v, float):
            out.append(safe_float(v))
        elif isinstance(v, dict):
            out.append(safe_dict(v))
        elif isinstance(v, list):
            out.append(safe_list(v))
        else:
            out.append(v)
    return out


def safe_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = safe_float(v)
        elif isinstance(v, dict):
            out[k] = safe_dict(v)
        elif isinstance(v, list):
            out[k] = safe_list(v)
        else:
            out[k] = v
    return out


# ==========================
# 거리 계산 및 형태 지표 (유지)
# ==========================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline: List[Tuple[float, float]]) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lon1, lat2, lon2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


def _to_local_xy(polyline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not polyline:
        return []
    ref_lat = polyline[0][0]
    ref_lon = polyline[0][1]
    R = 6371000.0
    cos_ref = math.cos(math.radians(ref_lat))
    xy = []
    for lat, lon in polyline:
        x = (lon - ref_lon) * cos_ref * R
        y = (lat - ref_lat) * R
        xy.append((x, y))
    return xy


def polygon_roundness(polyline: List[Tuple[float, float]]) -> float:
    if len(polyline) < 4:
        return 0.0
    xy = _to_local_xy(polyline)
    area = 0.0
    perimeter = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        area += x1 * y2 - x2 * y1
        perimeter += math.hypot(x2 - x1, y2 - y1)
    area = abs(area) * 0.5
    if area == 0 or perimeter == 0:
        return 0.0
    r = 4 * math.pi * area / (perimeter ** 2)
    if math.isinf(r) or math.isnan(r):
        return 0.0
    return r


def _edge_overlap_fraction(node_path: List[int]) -> float:
    if len(node_path) < 2:
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
    penalty = 0.0
    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        
        if b not in G.nodes or a not in G.nodes or c not in G.nodes:
            continue
            
        ya, xa = G.nodes[a]["y"], G.nodes[a]["x"]
        yb, xb = G.nodes[b]["y"], G.nodes[b]["x"]
        yc, xc = G.nodes[c]["y"], G.nodes[c]["x"]

        v1x = xb - xa
        v1y = yb - ya
        v2x = xc - xb
        v2y = yc - yb
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


def _path_length_on_graph(G: nx.Graph, nodes: List[int], weight: str = "length") -> float:
    """그래프 상에서 node 경로의 총 비용 (weight 합)."""
    if len(nodes) < 2:
        return 0.0
    total_cost = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            continue
        data = min(G[u][v].values(), key=lambda d: d.get(weight, 1.0))
        total_cost += float(data.get(weight, 0.0))
    return total_cost


def _apply_route_poison(G: nx.Graph, path_nodes: List[int], factor: float = 10.0) -> nx.Graph:
    """
    RUNAMIC 스타일 route poisoning: rod에 해당하는 간선 비용을 factor만큼 늘림.
    """
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G2.has_edge(u, v):
            for k in list(G2[u][v].keys()):
                data = G2[u][v][k]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
            if G2.has_edge(v, u):
                for k in list(G2[v][u].keys()):
                    data = G2[v][u][k]
                    if "length" in data:
                        data["length"] = float(data["length"]) * factor
    return G2


# ==========================
# OSM 보행자 그래프 구축 (V7 강력 필터링)
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    [V7 개선] 러닝에 부적합한 경로(residential, service, steps)를 제외하는 강력한 필터 적용.
    """
    radius_m = max(700.0, km * 500.0 + 700.0)
    
    # [V7 필터링] OSMnx의 기본 'walk' 필터에서 residential, service, steps를 제외
    # 'walk' 필터는 ['footway', 'path', 'pedestrian', 'steps', 'living_street', 'track', 'service', 'residential', 'tertiary'] 등을 포함합니다.
    
    # 러닝에 적합한 경로만 남기도록 필터를 명시적으로 구성
    safe_highways = [
        'footway', 'path', 'pedestrian', 'track', 'cycleway', 
        'tertiary', 'secondary', 'primary', 'trunk'
    ]
    
    # 러닝에 부적합한 경로를 제외 (residential, service, steps는 단지 내부나 계단일 확률이 높음)
    exclusion_filter = [
        'residential', 'service', 'steps', 'living_street', 'tertiary_link', 'secondary_link'
    ]
    
    # 안전한 경로 목록 필터링
    # residential, service 등을 제외한 나머지 (러닝에 적합한 넓은 도로 옆 보도나 공원 길)
    # filter_list는 OSMnx 1.1.2 이후부터 지원하는 방식으로, 복잡한 문자열 쿼리보다 안정적입니다.
    # 단, 'steps'를 제외하고 'footway'를 포함하는 방식으로 처리합니다.
    
    # OSMnx 쿼리 안정성을 위해, 'all' 네트워크 타입에서 부적합한 것만 제외하는 필터를 구성
    #
    
    # 러닝에 적합한 경로만 남기기 (가장 강력한 필터링)
    custom_filter = '["highway"~"footway|path|pedestrian|track|cycleway|tertiary|secondary|primary"]' + \
                    '["area"!~"yes"]' + \
                    '["access"!~"private"]'
    
    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        custom_filter=custom_filter,
        network_type="all_private", # custom_filter가 적용되도록 all_private 사용
        simplify=True,
        retain_all=False
    )
    
    if not G.nodes:
         raise ValueError("Filtered graph has no nodes.")
         
    return G


def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> List[Tuple[float, float]]:
    """그래프 노드 시퀀스를 (lat, lng) polyline으로 변환."""
    poly = []
    for n in nodes:
        node = G.nodes[n]
        lat = float(node["y"])
        lon = float(node["x"])
        poly.append((lat, lon))
    return poly


# ==========================
# fallback: 기하학적 사각형 루프
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float):
    target_m = km * 1000.0
    side = target_m / 4.0
    delta_deg_lat = side / 111000.0
    cos_lat = math.cos(math.radians(lat))
    delta_deg_lng = side / (111000.0 * cos_lat if cos_lat != 0 else 111000.0)

    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]
    
    center_lat = (a[0] + c[0]) / 2
    center_lng = (b[1] + d[1]) / 2
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]
    
    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# 메인: 러닝 루프 생성기
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    PURE PEDESTRIAN 러닝 루프 생성기 (V7 강력 필터링 버전)
    """
    start_time = time.time()
    target_m = km * 1000.0
    
    # [V7 유지] 스코어링 가중치 (강화된 값)
    ROUNDNESS_WEIGHT = 3.0
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3
    LENGTH_PENALTY_WEIGHT = 10.0

    meta: Dict[str, Any] = {
        "len": None,
        "err": None,
        "roundness": None,
        "overlap": None,
        "curve_penalty": None,
        "score": None,
        "success": False,
        "length_ok": False,
        "used_fallback": False,
        "valhalla_calls": 0,
        "kakao_calls": 0,
        "routes_checked": 0,
        "routes_validated": 0,
        "km_requested": km,
        "target_m": target_m,
        "time_s": None,
        "message": ""
    }

    # --------------------------
    # 1) OSM 보행자 그래프 구축 (V7 강력 필터링 적용)
    # --------------------------
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=0,
            routes_validated=0,
            message=f"OSM 보행자 그래프 생성/필터링 실패로 기하학적 사각형 루프를 사용했습니다: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=0,
            routes_validated=0,
            message=f"시작 노드 매칭 실패로 기하학적 사각형 루프를 사용했습니다: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)

    # --------------------------
    # 2) start에서의 단일-출발 최단거리 (rod 후보 탐색)
    # --------------------------
    # weight를 'length'로 복원하여 길이 기반의 안정적인 탐색 수행
    try:
        dist = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.8,
            weight="length"
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=0,
            routes_validated=0,
            message=f"그래프 최단거리 탐색 실패로 기하학적 사각형 루프를 사용했습니다: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # rod 길이 후보: 0.35 ~ 0.6 * target_m 사이 (max_leg 축소)
    min_leg = target_m * 0.35
    max_leg = target_m * 0.60
    candidate_nodes = [n for n, c in dist.items() if min_leg <= c <= max_leg and c >= 0.0 and n != start_node]

    if not candidate_nodes:
        candidate_nodes = [n for n, c in dist.items() if c >= target_m * 0.25 and c >= 0.0]

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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=0,
            routes_validated=0,
            message="적절한 rod endpoint 후보를 찾지 못해 기하학적 사각형 루프를 사용했습니다."
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:40]

    best_score = -1e18
    best_poly = None
    best_meta_stats = {}

    # --------------------------
    # 3) 각 endpoint에 대해 rod + detour 루프 생성 (V7 안정화 적용)
    # --------------------------
    for endpoint in candidate_nodes:
        
        # 3-1) start -> endpoint rod (길이 기반 최단 경로)
        try:
            forward_nodes = nx.shortest_path(
                undirected,
                start_node,
                endpoint,
                weight="length"
            )
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes, weight="length")
        if forward_len <= 0:
            continue

        # 3-2) rod 간선에 penalty를 줘서 detour 경로 유도 (Poisoning)
        poisoned = _apply_route_poison(undirected, forward_nodes, factor=10.0)

        # 3-3) endpoint -> start detour (Poisoned Length 기반 최단 경로)
        try:
            back_nodes = nx.shortest_path(
                poisoned,
                endpoint,
                start_node,
                weight="length"
            )
        except Exception:
            continue

        back_len = _path_length_on_graph(undirected, back_nodes, weight="length")
        if back_len <= 0:
            continue

        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        polyline = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(polyline)
        if length_m <= 0:
            continue

        err = abs(length_m - target_m)
        roundness = polygon_roundness(polyline)
        overlap = _edge_overlap_fraction(full_nodes)
        curve_penalty = _curve_penalty(full_nodes, undirected)

        length_ok = err <= 99.0
        if length_ok:
            meta["routes_validated"] += 1
            
        # [V7 유지] 스코어링 가중치 적용 (길이 오차 페널티 10.0)
        length_pen = err / target_m 
        score = (
            roundness * ROUNDNESS_WEIGHT
            - overlap * OVERLAP_PENALTY
            - curve_penalty * CURVE_PENALTY_WEIGHT
            - length_pen * LENGTH_PENALTY_WEIGHT
        )
        
        # 단일 경쟁 로직: 가장 높은 Score를 선택
        if score > best_score:
            best_score = score
            best_poly = polyline
            best_meta_stats = {
                "len": length_m,
                "err": err,
                "roundness": roundness,
                "overlap": overlap,
                "curve_penalty": curve_penalty,
                "score": score,
                "length_ok": length_ok,
            }

    # --------------------------
    # 4) 후보 루프가 하나도 없을 때
    # --------------------------
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=meta["routes_checked"],
            routes_validated=meta["routes_validated"],
            message="논문 기반 OSM 루프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다."
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 5) 최종 meta 구성
    # --------------------------
    used_fallback = False
    success = best_meta_stats["length_ok"]

    meta.update(best_meta_stats)
    meta.update(
        success=success,
        used_fallback=used_fallback,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else "요청 오차(±99m)를 초과하지만, 가장 인접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
