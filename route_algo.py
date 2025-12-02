import math
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
    rod 형태의 왕복을 줄이기 위해 사용.
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
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)
    return penalty


def graph_path_length(G: nx.Graph, nodes: List[int]) -> float:
    """그래프 상에서 node 경로의 길이 (edge length 합)."""
    if len(nodes) < 2:
        return 0.0
    length = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            continue
        data_dict = G[u][v]
        if isinstance(data_dict, dict):
            # MultiGraph: edge-key -> attr dict
            if len(data_dict) == 0:
                continue
            best = min(data_dict.values(), key=lambda d: d.get("length", 1.0))
            length += float(best.get("length", 0.0))
        else:
            # 방어적 코드 (예상치 못한 구조)
            try:
                length += float(data_dict.get("length", 0.0))
            except Exception:
                continue
    return length


# ==========================
# OSM 보행자 그래프 구축
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    OSMnx를 이용해 보행 가능한 네트워크만 추출.
    - network_type='walk'
    - motorway, trunk 등 차량 전용 도로는 기본적으로 제외
    - footway, path, sidewalk, cycleway, steps, pedestrian, track, service, residential 등을 포함
    """
    # 요청 거리에 따라 검색 반경 조절
    radius_m = max(700.0, km * 800.0 + 500.0)

    # 보행/자전거 친화 도로 중심 필터
    custom_filter = (
        '["highway"~"footway|path|pedestrian|sidewalk|cycleway|track|steps|living_street|'
        'residential|service|unclassified|tertiary|secondary|primary|tertiary_link|secondary_link|primary_link"]'
        '["area"!~"yes"]'
    )

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        custom_filter=custom_filter,
        simplify=True,
        retain_all=False,
    )

    if not G.nodes:
        raise ValueError("Filtered pedestrian graph has no nodes.")
    return G


def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> List[Tuple[float, float]]:
    """그래프 노드 시퀀스를 (lat, lng) polyline으로 변환."""
    poly = []
    for n in nodes:
        node = G.nodes[n]
        poly.append((float(node["y"]), float(node["x"])))
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
    side = target_m / 4.0  # 네 변의 합이 target_m
    delta_deg_lat = side / 111000.0
    cos_lat = math.cos(math.radians(lat))
    delta_deg_lng = side / (111000.0 * cos_lat if cos_lat != 0 else 111000.0)

    a = (lat + delta_deg_lat, lng)
    b = (lat + delta_deg_lng, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng + delta_deg_lng)
    d = (lat - delta_deg_lat, lng)
    poly = [a, b, c, d, a]

    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# CYCLE 기반 루트 탐색 (옵션 A)
# ==========================
def _find_cycles_around_start(
    G: nx.MultiGraph,
    start_node: int,
    target_m: float,
    meta: Dict[str, Any],
    max_pairs: int = 40,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    start_node 주변에서 simple cycle을 탐색.
    - pivot = start_node
    - pivot의 이웃 노드 쌍 (n1, n2)에 대해, pivot을 제거한 그래프에서 n1->n2 최단 경로를 찾아 사이클 구성.
    - best_strict: 길이 오차가 ±5% 이내인 후보 중 최고 점수
    - best_any: 모든 후보 중 최고 점수 (디버그/로그용)
    """
    neighbors = list(G.neighbors(start_node))
    if len(neighbors) < 2:
        return None, None

    # pivot 제거한 그래프 (사이클이 pivot을 관통하도록)
    H = G.copy()
    if H.has_node(start_node):
        H.remove_node(start_node)

    best_strict = None
    best_any = None

    # 이웃 쌍 조합을 제한하여 탐색 비용 제어
    pair_count = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            n1 = neighbors[i]
            n2 = neighbors[j]
            pair_count += 1
            if pair_count > max_pairs:
                break

            try:
                path = nx.shortest_path(H, n1, n2, weight="length")
            except Exception:
                continue

            # 사이클 노드 시퀀스: start -> ... -> start
            cycle_nodes = [start_node] + path + [start_node]

            # polyline/길이/스코어 계산
            poly_base = _nodes_to_polyline(G, cycle_nodes)
            length_m = polyline_length_m(poly_base)
            if length_m <= 0:
                continue

            err = abs(length_m - target_m)
            roundness = polygon_roundness(poly_base)
            overlap = _edge_overlap_fraction(cycle_nodes)
            curve_pen = _curve_penalty(cycle_nodes, G)

            meta["routes_checked"] += 1

            # 길이 오차 비율
            length_pen = err / target_m
            # 길이 페널티를 가장 크게, 그 다음 roundness/overlap/curve_penalty로 보정
            score = (
                roundness * 3.0
                - overlap * 2.0
                - curve_pen * 0.3
                - length_pen * 10.0
            )

            candidate = {
                "nodes": cycle_nodes,
                "len": length_m,
                "err": err,
                "roundness": roundness,
                "overlap": overlap,
                "curve_penalty": curve_pen,
                "score": score,
            }

            # 모든 후보 중 최고 점수
            if (best_any is None) or (score > best_any["score"]):
                best_any = candidate

            # ±5% 이내만 strict 후보로
            if err <= target_m * 0.05:
                meta["routes_validated"] += 1
                if (best_strict is None) or (score > best_strict["score"]):
                    best_strict = candidate

        if pair_count > max_pairs:
            break

    return best_strict, best_any


# ==========================
# 메인: 러닝 루프 생성기 (옵션 A)
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    CYCLE 기반 러닝 루프 생성기 (옵션 A).
    - 보행자 네트워크('walk') + custom_filter 기반.
    - start 지점을 피벗으로 하는 simple cycle을 탐색.
    - 생성된 루트 길이가 요청 거리의 ±5% 이내일 때만 유효 루트로 인정.
      그 외에는 기하학적 사각형 루프 fallback 사용.
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

    # 1) OSM 보행자 그래프 구축
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
            length_ok=False,
            used_fallback=True,
            message=f"보행자 그래프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # undirected 그래프로 변환
    UG: nx.MultiGraph = ox.utils_graph.get_undirected(G)

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
            length_ok=False,
            used_fallback=True,
            message=f"시작 노드 매칭 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 2) start_node를 중심으로 cycle 탐색
    best_strict, best_any = _find_cycles_around_start(UG, start_node, target_m, meta)

    # 3) ±5% 이내 유효 루트를 찾지 못한 경우 -> fallback
    if best_strict is None:
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
            length_ok=False,
            used_fallback=True,
            message="요청 거리의 ±5% 이내에 해당하는 보행 루프를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 4) 최종 루트 구성 (polyline의 시작/끝은 요청 좌표로 고정)
    nodes = best_strict["nodes"]
    poly_base = _nodes_to_polyline(UG, nodes)

    # 요청 시작 좌표를 첫/마지막 좌표로 설정 (JSON 일관성 유지)
    if poly_base:
        polyline = [(float(lat), float(lng))] + poly_base[1:-1] + [(float(lat), float(lng))]
    else:
        polyline = [(float(lat), float(lng)), (float(lat), float(lng))]

    length = polyline_length_m(polyline)
    err = abs(length - target_m)
    roundness = polygon_roundness(polyline)
    overlap = _edge_overlap_fraction(nodes)
    curve_pen = _curve_penalty(nodes, UG)
    length_ok = err <= target_m * 0.05
    length_pen = err / target_m if target_m > 0 else 0.0
    score = (
        roundness * 3.0
        - overlap * 2.0
        - curve_pen * 0.3
        - length_pen * 10.0
    )

    meta.update(
        len=length,
        err=err,
        roundness=roundness,
        overlap=overlap,
        curve_penalty=curve_pen,
        score=score,
        success=length_ok,
        length_ok=length_ok,
        used_fallback=False,
        message="최적의 CYCLE 기반 러닝 루프가 도출되었습니다."
        if length_ok
        else "길이 오차가 약간 존재하지만, 가장 인접한 CYCLE 기반 러닝 루프를 반환합니다.",
    )
    meta["time_s"] = time.time() - start_time

    return safe_list(polyline), safe_dict(meta)
