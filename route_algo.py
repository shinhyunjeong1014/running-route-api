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
        data = G[u][v]
        if isinstance(data, dict) and "length" in data:
            # simple Graph
            length += float(data.get("length", 0.0))
        else:
            # MultiGraph 등
            try:
                ed = min(G[u][v].values(), key=lambda d: d.get("length", 1.0))
                length += float(ed.get("length", 0.0))
            except Exception:
                continue
    return length


# ==========================
# OSM 보행자 그래프 구축
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    시작점 주변 dist(미터) 반경의 보행자용 그래프를 가져온다.
    (Overpass API 사용, network_type='walk')
    """
    # km가 커질수록 반경을 넉넉히 잡되, 너무 크게는 안 가게 제한
    radius_m = max(800.0, km * 700.0 + 500.0)
    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
    )
    if not G.nodes:
        raise ValueError("Empty pedestrian graph from OSM.")
    return G


def _to_simple_undirected(G: nx.MultiDiGraph) -> nx.Graph:
    """
    MultiDiGraph -> simple undirected Graph 로 변환.
    각 (u, v) 간선에 대해 length가 가장 짧은 간선만 사용.
    """
    UG_multi: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    H = nx.Graph()
    for u, v, data in UG_multi.edges(data=True):
        length = float(data.get("length", 1.0))
        if H.has_edge(u, v):
            if length < H[u][v].get("length", 1e18):
                H[u][v]["length"] = length
        else:
            H.add_edge(u, v, length=length)
    return H


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
# 메인: Cycle 기반 러닝 루프 생성기 (v3)
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    CYCLE-BASED v3 러닝 루프 생성기.

    1) OSM 보행자 그래프 구축 (network_type='walk')
    2) simple undirected 그래프 변환
    3) cycle_basis 로 단순 사이클 후보들 추출
    4) 각 사이클을 루프 후보로 평가 (길이, roundness, overlap, curve_penalty)
    5) target_m 과 가장 잘 맞는 사이클을 선택
    6) 실패 시 기하학적 사각형 루프 사용
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
    # 1) 보행자 그래프 구축
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
            message=f"OSM 보행자 그래프 생성 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # start_node: 루프와의 거리 평가용 (polyline에는 직접 안 들어갈 수 있음)
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
    except Exception:
        start_node = None

    # simple undirected graph
    try:
        UG = _to_simple_undirected(G)
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
            message=f"보행자 그래프 변환 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 2) cycle_basis 를 사용한 단순 사이클 추출
    # --------------------------
    try:
        if start_node is not None and start_node in UG:
            cycles = nx.cycle_basis(UG, root=start_node)
        else:
            cycles = nx.cycle_basis(UG)
    except Exception:
        cycles = []

    # root 기준 cycle_basis 가 빈 경우 전체 기준으로 한 번 더 시도
    if not cycles:
        try:
            cycles = nx.cycle_basis(UG)
        except Exception:
            cycles = []

    best_poly = None
    best_score = -1e18
    best_round = 0.0
    best_overlap = 0.0
    best_curve_penalty = 0.0
    best_err = 1e18
    routes_checked = 0
    routes_validated = 0

    # 사이클이 너무 많을 경우를 대비해 샘플링
    random.shuffle(cycles)
    MAX_CYCLES = 200
    cycles = cycles[:MAX_CYCLES]

    # --------------------------
    # 3) 각 사이클을 후보 러닝 루프로 평가
    # --------------------------
    for cyc in cycles:
        if len(cyc) < 4:
            continue  # 너무 작은 사이클은 제외 (삼각형 포함)
        # 닫힌 루프로 만들기 위해 처음 노드를 끝에 한 번 더 추가
        cyc_nodes = list(cyc) + [cyc[0]]

        # 루프 길이 계산
        loop_len = _path_length_on_graph(UG, cyc_nodes)
        if loop_len <= 0:
            continue

        # 요청 거리 대비 너무 짧거나 너무 긴 경우는 제외
        if loop_len < target_m * 0.4 or loop_len > target_m * 1.8:
            continue

        polyline = _nodes_to_polyline(UG, cyc_nodes)
        length_m = polyline_length_m(polyline)
        if length_m <= 0:
            continue

        err = abs(length_m - target_m)
        roundness = polygon_roundness(polyline)
        overlap = _edge_overlap_fraction(cyc_nodes)
        curve_penalty = _curve_penalty(cyc_nodes, UG)

        # 시작점과 루프 간의 거리 (루프 상에서 가장 가까운 점 기준)
        if start_node is not None and start_node in UG:
            min_d = 1e18
            for n in cyc_nodes:
                nx_lat = UG.nodes[n]["y"]
                nx_lng = UG.nodes[n]["x"]
                d = haversine(lat, lng, nx_lat, nx_lng)
                if d < min_d:
                    min_d = d
        else:
            min_d = 0.0

        # 시작점과 너무 멀리 떨어진 사이클은 제외
        max_center_dist = max(300.0, target_m * 0.5)
        if min_d > max_center_dist:
            continue

        length_pen = err / max(target_m, 1.0)
        start_dist_pen = min_d / max(target_m, 1.0)

        # 스코어: roundness 높을수록, overlap/curve_penalty/length_pen/start_dist_pen 낮을수록 좋음
        score = (
            roundness * 3.0
            - overlap * 2.0
            - curve_penalty * 0.3
            - length_pen * 5.0
            - start_dist_pen * 2.0
        )

        routes_checked += 1
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
    # 4) 사이클 기반 후보가 없을 때
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
            message="CYCLE 기반 루프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 5) 최종 meta 구성
    # --------------------------
    length_ok = best_err <= 99.0
    success = length_ok

    final_len = polyline_length_m(best_poly)

    meta.update(
        len=final_len,
        err=best_err,
        roundness=best_round,
        overlap=best_overlap,
        curve_penalty=best_curve_penalty,
        score=best_score,
        success=success,
        length_ok=length_ok,
        used_fallback=False,
        routes_checked=routes_checked,
        routes_validated=routes_validated,
        message=(
            "CYCLE 기반 OSM 보행자 그래프에서 러닝 루프를 생성했습니다."
            if success
            else "요청 오차(±99m)를 초과하지만, 가장 인접한 CYCLE 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
