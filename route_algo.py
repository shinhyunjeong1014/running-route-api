import math
import random
import time
from typing import List, Tuple, Dict, Any

import networkx as nx
import osmnx as ox


# ==========================
# JSON-safe 변환 유틸
# ==========================
def safe_float(x: Any):
    """NaN / Inf 를 JSON에서 허용 가능한 값(None)으로 변환."""
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
# 거리 계산 (Haversine)
# ==========================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 점 사이의 대원거리 (미터)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline: List[Tuple[float, float]]) -> float:
    """경로(위도,경도 리스트)의 총 길이 (미터)."""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lon1, lat2, lon2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ==========================
# roundness 계산용 로컬 좌표 변환
# ==========================
def _to_local_xy(polyline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """작은 영역에서는 위경도를 간단한 평면 좌표(미터)로 근사."""
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
    """
    4πA / P^2 로 정의되는 roundness.
    1에 가까울수록 원에 가까운 형태, 0에 가까울수록 길쭉/찌그러진 형태.
    """
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


# ==========================
# overlap / 커브 페널티
# ==========================
def _edge_overlap_fraction(node_path: List[int]) -> float:
    """
    노드 시퀀스에서 같은 간선을 여러 번 쓰는 비율.
    (rod + detour 구조 등에서 중복을 줄이기 위해 사용)
    """
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
    """
    경로에서 너무 급격한 커브(예: 60도 이하)를 얼마나 많이 만드는지 측정.
    작을수록 부드러운 경로.
    """
    if len(node_path) < 3:
        return 0.0
    penalty = 0.0
    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]

        if a not in G.nodes or b not in G.nodes or c not in G.nodes:
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
        theta = math.acos(dot)  # 0 ~ pi

        # 너무 급한 커브(60도 미만)에 페널티
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    """그래프 상에서 node 경로의 길이 (edge의 length 합)."""
    if len(nodes) < 2:
        return 0.0
    length = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            continue
        data = min(G[u][v].values(), key=lambda d: d.get("length", 1.0))
        length += float(data.get("length", 0.0))
    return length


def _apply_route_poison(G: nx.Graph, path_nodes: List[int], factor: float = 5.0) -> nx.Graph:
    """
    RUNAMIC 스타일 route poisoning (v2에서는 사용하지 않지만 남겨둠).
    """
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G2.has_edge(u, v):
            for k in G2[u][v]:
                data = G2[u][v][k]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
    return G2


# ==========================
# OSM 보행자 그래프 구축
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    시작점 주변 dist(미터) 반경의 보행자용 그래프를 가져온다.
    (Overpass API 사용, network_type='walk')
    """
    radius_m = max(800.0, km * 700.0 + 500.0)
    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True
    )
    if not G.nodes:
        raise ValueError("Empty pedestrian graph from OSM.")
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
    """
    모든 고급 알고리즘 실패 시 사용되는 마지막 안전 장치.
    요청 거리 km를 대략 만족하는 사각형 루프 생성.
    """
    target_m = km * 1000.0
    side = target_m / 4.0  # 4변 합이 target_m
    delta_deg_lat = side / 111000.0
    cos_lat = math.cos(math.radians(lat))
    delta_deg_lng = side / (111000.0 * cos_lat if cos_lat != 0 else 111000.0)

    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]
    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# 메인: 러닝 루프 생성기 (PSP2 스타일)
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    PURE PEDESTRIAN 러닝 루프 생성기 (v2: PSP2 스타일)

    1) OSM 보행자 그래프 구축 (network_type='walk')
    2) start에서 단일-출발 최단거리 계산
    3) u, v 후보를 거리 기반으로 샘플링
    4) s->u, u->v, v->s 세 구간 모두 최단경로로 연결 (삼각 루프)
    5) 길이 오차 / roundness / overlap / curve_penalty 종합 스코어로 최적 루프 선택
    6) 실패 시 fallback 사각형 루프 사용
    """
    start_time = time.time()
    target_m = km * 1000.0

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
        "message": "",
    }

    # --------------------------
    # 1) OSM 보행자 그래프 구축
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
            message=f"OSM 보행자 그래프 생성 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 시작 노드 매칭
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
            message=f"시작 노드 매칭 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)

    # --------------------------
    # 2) start에서의 단일-출발 최단거리
    # --------------------------
    try:
        dist_s = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.9,
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=0,
            routes_validated=0,
            message=f"그래프 최단거리 탐색 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    if len(dist_s) <= 1:
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
            message="유효한 보행자 노드가 부족하여 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 3) u, v 후보 선정 (거리 기반 ring)
    # --------------------------
    # u: 0.2L ~ 0.6L, v: 0.4L ~ 0.9L
    u_min = target_m * 0.2
    u_max = target_m * 0.6
    v_min = target_m * 0.4
    v_max = target_m * 0.9

    u_nodes = [n for n, d in dist_s.items() if n != start_node and u_min <= d <= u_max]
    v_nodes = [n for n, d in dist_s.items() if n != start_node and v_min <= d <= v_max]

    # 후보가 없으면 조건 완화
    if not u_nodes:
        u_nodes = [n for n, d in dist_s.items() if n != start_node and d >= target_m * 0.15]
    if not v_nodes:
        v_nodes = [n for n, d in dist_s.items() if n != start_node and d >= target_m * 0.3]

    if not u_nodes or not v_nodes:
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
            message="PSP2용 u,v 후보를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    random.shuffle(u_nodes)
    random.shuffle(v_nodes)
    u_nodes = u_nodes[:30]
    v_nodes = v_nodes[:30]

    best_score = -1e18
    best_poly = None
    best_round = 0.0
    best_overlap = 0.0
    best_curve_penalty = 0.0
    best_err = 1e18
    routes_checked = 0
    routes_validated = 0

    MAX_TRIALS = 80
    trials = 0

    # --------------------------
    # 4) PSP2 스타일: s→u→v→s 루프 후보 탐색
    # --------------------------
    while trials < MAX_TRIALS:
        trials += 1
        u = random.choice(u_nodes)
        v = random.choice(v_nodes)
        if u == v or u == start_node or v == start_node:
            continue

        try:
            path_su = nx.shortest_path(undirected, start_node, u, weight="length")
            path_uv = nx.shortest_path(undirected, u, v, weight="length")
            path_vs = nx.shortest_path(undirected, v, start_node, weight="length")
        except Exception:
            continue

        full_nodes = list(path_su) + list(path_uv)[1:] + list(path_vs)[1:]
        if len(full_nodes) < 4:
            continue

        routes_checked += 1

        loop_len = _path_length_on_graph(undirected, full_nodes)
        if loop_len <= 0:
            continue

        polyline = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(polyline)
        if length_m <= 0:
            continue

        err = abs(length_m - target_m)
        roundness = polygon_roundness(polyline)
        overlap = _edge_overlap_fraction(full_nodes)
        curve_penalty = _curve_penalty(full_nodes, undirected)

        length_pen = err / max(target_m, 1.0)
        score = (
            roundness * 3.0
            - overlap * 2.0
            - curve_penalty * 0.3
            - length_pen * 6.0
        )

        if err <= 99.0:
            routes_validated += 1

        if score > best_score:
            best_score = score
            best_poly = polyline
            best_round = roundness
            best_overlap = overlap
            best_curve_penalty = curve_penalty
            best_err = err

    # --------------------------
    # 5) 후보 루프가 하나도 없을 때
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
            routes_checked=routes_checked,
            routes_validated=routes_validated,
            message="PSP2 기반 루프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 6) 최종 meta 구성
    # --------------------------
    used_fallback = False
    length_ok = best_err <= 99.0
    success = length_ok

    meta.update(
        len=polyline_length_m(best_poly),
        err=best_err,
        roundness=best_round,
        overlap=best_overlap,
        curve_penalty=best_curve_penalty,
        score=best_score,
        success=success,
        length_ok=length_ok,
        used_fallback=used_fallback,
        routes_checked=routes_checked,
        routes_validated=routes_validated,
        message=(
            "PSP2 스타일 OSM 보행자 그래프에서 러닝 루프를 생성했습니다."
            if success
            else "요청 오차(±99m)를 초과하지만, 가장 인접한 PSP2 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
