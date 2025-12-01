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
    d = R * c
    if math.isinf(d) or math.isnan(d):
        return 0.0
    return d


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

        if b not in G.nodes or a not in G.nodes or c not in G.nodes:
            continue

        y_a, x_a = G.nodes[a].get("y"), G.nodes[a].get("x")
        y_b, x_b = G.nodes[b].get("y"), G.nodes[b].get("x")
        y_c, x_c = G.nodes[c].get("y"), G.nodes[c].get("x")

        if y_a is None or y_b is None or y_c is None:
            continue
        if x_a is None or x_b is None or x_c is None:
            continue

        v1x = x_b - x_a
        v1y = y_b - y_a
        v2x = x_c - x_b
        v2y = y_c - y_b
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0 or n2 == 0:
            continue

        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)  # 0 ~ pi (180도)

        # 너무 급한 커브(60도 미만, pi/3)에 페널티
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


# ==========================
# 그래프 / Polyline 유틸
# ==========================
def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> List[Tuple[float, float]]:
    """그래프 노드 시퀀스를 (lat, lng) polyline으로 변환."""
    poly: List[Tuple[float, float]] = []
    for n in nodes:
        data = G.nodes.get(n)
        if not data:
            continue
        lat = data.get("y")
        lon = data.get("x")
        if lat is None or lon is None:
            continue
        poly.append((float(lat), float(lon)))
    return poly


def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    러닝/보행에 적합한 OSM 보행자 네트워크 생성.

    - OSMnx network_type="walk" 사용
    - 너무 작은 네트워크가 되지 않도록 km 에 비례해서 반경 확장
    """
    # km 가 커질수록 탐색 반경을 넉넉히 잡되, 과도한 확장은 방지
    radius_m = max(800.0, km * 700.0 + 500.0)

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )

    # edge length 보정 (없을 경우 계산)
    G = ox.add_edge_lengths(G)

    # 고립 노드 제거
    G = ox.utils_graph.remove_isolated_nodes(G)

    return G


def _to_simple_undirected(G: nx.MultiDiGraph) -> nx.Graph:
    """
    MultiDiGraph -> 단순 무방향 그래프 변환.
    - 가장 짧은 length edge 하나만 유지
    - node 의 x,y 좌표 유지 (cycle 기반 알고리즘에서 필수)
    """
    UG: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    SG = nx.Graph()

    # 노드 복사 (x, y 보존)
    for n, data in UG.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            continue
        SG.add_node(n, x=float(x), y=float(y))

    # 가장 짧은 edge 만 선택
    for u, v, key, data in UG.edges(keys=True, data=True):
        if u not in SG.nodes or v not in SG.nodes:
            continue
        length = float(data.get("length", 0.0))
        if length <= 0:
            continue
        if SG.has_edge(u, v):
            # 기존 것보다 짧으면 교체
            if length < SG[u][v].get("length", 1e18):
                SG[u][v]["length"] = length
        else:
            SG.add_edge(u, v, length=length)

    # 연결성 확보를 위해 (드물지만) 한 개 이하의 컴포넌트만 유지
    if not nx.is_connected(SG):
        comps = list(nx.connected_components(SG))
        if comps:
            largest = max(comps, key=len)
            SG = SG.subgraph(largest).copy()

    return SG


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

    # 원점 기준 대략적인 사각형
    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]

    # 시작점을 중심으로 재배치 (더 안정적인 형태)
    center_lat = (a[0] + c[0]) / 2
    center_lng = (b[1] + d[1]) / 2
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]

    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# CYCLE 기반 v3.1: 시작점 포함 단순 사이클 탐색
# ==========================
def _search_cycles_from_start(
    G: nx.Graph,
    start_node: int,
    target_m: float,
    tol_m: float,
    max_routes: int = 3000,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    단순 DFS 로 start_node 를 포함하는 사이클을 찾는다.

    - 길이 범위: [target_m - tol_m, target_m + tol_m]
    - 허용 오차 내 사이클이 없으면 가장 스코어가 좋은 사이클을 반환
    """
    min_len = max(0.0, target_m - tol_m)
    max_len = target_m + tol_m
    max_path_len = target_m * 1.6  # 약간의 여유 허용

    if start_node not in G.nodes:
        return [], {}

    # DFS 스택: (현재노드, path_nodes, length_so_far)
    stack: List[Tuple[int, List[int], float]] = []

    for v in G.neighbors(start_node):
        edge_len = float(G[start_node][v].get("length", 0.0))
        if edge_len <= 0:
            continue
        stack.append((v, [start_node, v], edge_len))

    routes_checked = 0
    routes_validated = 0

    best_cycle_len_ok: List[int] = []
    best_meta_len_ok: Dict[str, Any] = {}
    best_score_len_ok = -1e18

    best_cycle_any: List[int] = []
    best_meta_any: Dict[str, Any] = {}
    best_score_any = -1e18

    while stack and routes_checked < max_routes:
        current, path, dist_so_far = stack.pop()

        # 확장 길이가 너무 크면 중단
        if dist_so_far > max_path_len:
            continue

        for nxt in G.neighbors(current):
            edge_len = float(G[current][nxt].get("length", 0.0))
            if edge_len <= 0:
                continue

            new_dist = dist_so_far + edge_len

            # start 로 돌아와서 사이클이 완성되는 경우
            if nxt == start_node:
                if len(path) < 3:
                    # 2-노드 왕복은 러닝 루프로 부적절
                    continue

                routes_checked += 1
                full_nodes = path + [start_node]

                polyline = _nodes_to_polyline(G, full_nodes)
                if len(polyline) < 4:
                    continue

                length_m = polyline_length_m(polyline)
                if length_m <= 0:
                    continue

                err = abs(length_m - target_m)
                roundness = polygon_roundness(polyline)
                overlap = _edge_overlap_fraction(full_nodes)
                curve_penalty = _curve_penalty(full_nodes, G)

                length_ok = (min_len <= length_m <= max_len)
                if length_ok:
                    routes_validated += 1

                # 스코어 계산 (길이 오차 / 모양 / 중복 / 급커브 반영)
                length_pen = err / target_m if target_m > 0 else 0.0
                score = (
                    roundness * 3.0
                    - overlap * 2.0
                    - curve_penalty * 0.3
                    - length_pen * 5.0
                )

                # 1) 허용오차 내 route 중 최적
                if length_ok and score > best_score_len_ok:
                    best_score_len_ok = score
                    best_cycle_len_ok = full_nodes
                    best_meta_len_ok = {
                        "len": length_m,
                        "err": err,
                        "roundness": roundness,
                        "overlap": overlap,
                        "curve_penalty": curve_penalty,
                        "score": score,
                        "length_ok": True,
                    }

                # 2) 전체 route 중 최적 (fallback 용)
                if score > best_score_any:
                    best_score_any = score
                    best_cycle_any = full_nodes
                    best_meta_any = {
                        "len": length_m,
                        "err": err,
                        "roundness": roundness,
                        "overlap": overlap,
                        "curve_penalty": curve_penalty,
                        "score": score,
                        "length_ok": length_ok,
                    }

                continue

            # 아직 start 가 아니면, 단순 경로 유지 (cycle 단순성 보장)
            if nxt in path:
                continue

            # 길이 상한 초과는 확장 X
            if new_dist > max_path_len:
                continue

            stack.append((nxt, path + [nxt], new_dist))

    # 허용 오차 내 후보가 있으면 우선적으로 사용
    if best_cycle_len_ok:
        meta = dict(best_meta_len_ok)
        meta["routes_checked"] = routes_checked
        meta["routes_validated"] = routes_validated
        return best_cycle_len_ok, meta

    # 그 외에는 가장 스코어 좋은 사이클을 fallback 으로 사용
    if best_cycle_any:
        meta = dict(best_meta_any)
        meta["routes_checked"] = routes_checked
        meta["routes_validated"] = routes_validated
        return best_cycle_any, meta

    # 사이클을 하나도 못 찾은 경우
    return [], {"routes_checked": routes_checked, "routes_validated": routes_validated}


# ==========================
# 메인: 러닝 루프 생성기 (Cycle 기반 v3.2 안정화)
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    CYCLE 기반 v3.2 안정화 버전 러닝 루프 생성기.

    - 시작 노드(보행 네트워크 상)와 실제 출발점 사이 거리를 계산
    - 너무 멀면(예: 250m 이상, 또는 target_m * 0.4 이상) 네트워크 기반 경로를 포기하고
      출발좌표 중심의 사각형 fallback 루프를 사용
    - 그렇지 않으면 시작 노드를 포함하는 단순 사이클을 탐색
    - 길이 오차(±99m) 내 사이클이 있으면 최우선 선택
    - 없으면 가장 형태가 좋은 사이클을 선택
    - 그래프/사이클 탐색 실패 시, 기하학적 사각형 루프 fallback
    """
    start_time = time.time()
    target_m = km * 1000.0
    tol_m = 99.0  # 길이 허용 오차

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
        "snap_dist_m": None,   # 출발 좌표와 네트워크 최근접 노드 간 거리
        "time_s": None,
        "message": "",
    }

    # --------------------------
    # 1) OSM 보행자 그래프 구축
    # --------------------------
    try:
        G_raw = _build_pedestrian_graph(lat, lng, km)
        if not G_raw.nodes:
            raise ValueError("Empty pedestrian graph from OSM.")
    except Exception as e:
        # 그래프 생성에 실패하면 바로 fallback
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
            length_ok=(err <= tol_m),
            used_fallback=True,
            message=f"OSM 보행자 그래프 생성 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 2) 단순 무방향 그래프 변환 (Cycle 탐색용)
    # --------------------------
    try:
        G = _to_simple_undirected(G_raw)
        if not G.nodes or not G.edges:
            raise ValueError("Simple undirected graph is empty.")
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
            length_ok=(err <= tol_m),
            used_fallback=True,
            message=f"보행자 그래프 단순화 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 3) 시작 노드 찾기 + 출발점과 거리 체크
    # --------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
        if start_node not in G.nodes:
            raise ValueError("Start node not in simplified graph.")

        node_data = G.nodes[start_node]
        n_lat = float(node_data.get("y", lat))
        n_lng = float(node_data.get("x", lng))
        snap_dist = haversine(lat, lng, n_lat, n_lng)  # m
        meta["snap_dist_m"] = snap_dist

        # 출발점과 네트워크가 너무 멀리 떨어진 경우 -> 네트워크 기반 경로 폐기
        max_snap_dist = min(250.0, target_m * 0.4)  # 예: 2km -> 250m, 1km -> 250m, 5km -> 250m
        if snap_dist > max_snap_dist:
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
                length_ok=(err <= tol_m),
                used_fallback=True,
                message=(
                    "출발 지점에서 보행 네트워크까지 거리가 너무 멀어 "
                    "네트워크 기반 루프를 사용하지 않고, 출발 좌표 중심의 기하학적 사각형 루프를 사용했습니다."
                ),
            )
            meta["time_s"] = time.time() - start_time
            return safe_list(poly), safe_dict(meta)

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
            length_ok=(err <= tol_m),
            used_fallback=True,
            message=f"시작 노드 매칭 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 4) 시작점 포함 단순 사이클 탐색
    # --------------------------
    try:
        cycle_nodes, cycle_meta = _search_cycles_from_start(
            G,
            start_node,
            target_m=target_m,
            tol_m=tol_m,
            max_routes=4000,  # 탐색 상한 (너무 많은 사이클 탐색 방지)
        )
    except Exception as e:
        cycle_nodes = []
        cycle_meta = {"error": str(e)}

    # --------------------------
    # 5) 사이클이 없는 경우 -> fallback
    # --------------------------
    if not cycle_nodes:
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
            length_ok=(err <= tol_m),
            used_fallback=True,
            routes_checked=cycle_meta.get("routes_checked", 0),
            routes_validated=cycle_meta.get("routes_validated", 0),
            message="그래프 내에서 시작점을 포함하는 적절한 사이클을 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 6) 최종 polyline 및 meta 구성
    # --------------------------
    polyline = _nodes_to_polyline(G, cycle_nodes)
    length_m = polyline_length_m(polyline)
    err = abs(length_m - target_m)
    roundness = polygon_roundness(polyline)
    overlap = _edge_overlap_fraction(cycle_nodes)
    curve_penalty = _curve_penalty(cycle_nodes, G)

    length_ok = (err <= tol_m)
    success = length_ok

    meta.update(
        len=length_m,
        err=err,
        roundness=roundness,
        overlap=overlap,
        curve_penalty=curve_penalty,
        score=cycle_meta.get("score"),
        success=success,
        length_ok=length_ok,
        used_fallback=False,
        routes_checked=cycle_meta.get("routes_checked", 0),
        routes_validated=cycle_meta.get("routes_validated", 0),
        message=(
            "Cycle 기반 보행자 네트워크에서 러닝 루프를 생성했습니다."
            if success
            else "요청 오차(±99m)를 초과하지만, Cycle 기반에서 가장 인접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(polyline)
    return safe_poly, safe_meta
