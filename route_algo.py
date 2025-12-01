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


def safe_list(arr: Any):
    if not isinstance(arr, list):
        return arr
    out = []
    for v in arr:
        if isinstance(v, float):
            out.append(safe_float(v))
        elif isinstance(v, dict):
            out.append(safe_dict(v))
        elif isinstance(v, list):
            out.append(safe_list(v))
        else:
            out.append(v)
    return out


def safe_dict(d: Any):
    if not isinstance(d, dict):
        return d
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
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
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
    (rod + detour 구조에서 중복을 줄이기 위해 사용)
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
        # MultiGraph 이므로 여러 edge 중 length가 최소인 것 사용
        data = min(G[u][v].values(), key=lambda d: d.get("length", 1.0))
        length += float(data.get("length", 0.0))
    return length


def _apply_route_poison(G: nx.Graph, path_nodes: List[int], factor: float = 5.0) -> nx.Graph:
    """
    RUNAMIC 스타일 route poisoning:
    rod 에 해당하는 간선의 비용(length)을 factor만큼 증가시켜서
    detour 경로를 유도하는 용도였으나,
    PSP3 구현에서는 사용하지 않지만 남겨둔다(추후 hybrid 용도).
    """
    H = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if H.has_edge(u, v):
            for key in H[u][v]:
                length = H[u][v][key].get("length", 1.0)
                H[u][v][key]["length"] = length * factor
        if H.has_edge(v, u):
            for key in H[v][u]:
                length = H[v][u][key].get("length", 1.0)
                H[v][u][key]["length"] = length * factor
    return H


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
# 메인: PSP3 스타일 러닝 루프 생성기
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    PURE PEDESTRIAN 러닝 루프 생성기 (PSP3 스타일)

    1) OSM 보행자 그래프 구축
    2) start_node 에서 single-source Dijkstra 로 거리 계산
    3) L/4, L/2, 3L/4 ring 에서 u, m, v 후보 노드 선정
    4) s → u → m → v → s 4개 leg 를 모두 최단 경로로 연결
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
        "valhalla_calls": 0,      # 이 버전에서는 사용하지 않음
        "kakao_calls": 0,         # 이 버전에서는 사용하지 않음
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
        # 그래프 자체를 못 가져오면 바로 fallback
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

    # 시작 노드 찾기
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
            message=f"시작 노드 탐색 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)

    # --------------------------
    # 2) PSP3 스타일 3-via 러닝 루프 생성
    #    - s(시작) → u → m → v → s 의 4개 leg 로 구성
    #    - 각 leg 길이는 대략 L/4 근처가 되도록 설계
    #    - 모든 경로는 OSM 보행자 네트워크 최단경로를 사용
    # --------------------------
    try:
        # start_node 로부터의 단일 출발 최단거리
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

    # 길이 타겟과 ring 설정
    quarter = target_m / 4.0
    eps = 0.25  # 각 leg 길이에 허용할 상대 오차 (25%)

    # s 에서의 거리 기반 ring
    via1_min = quarter * (1.0 - eps)
    via1_max = quarter * (1.0 + eps)
    mid_min = 2.0 * quarter * (1.0 - eps)
    mid_max = 2.0 * quarter * (1.0 + eps)
    via2_min = 3.0 * quarter * (1.0 - eps)
    via2_max = 3.0 * quarter * (1.0 + eps)

    via1_nodes = [
        n for n, d in dist_s.items()
        if n != start_node and via1_min <= d <= via1_max
    ]
    mid_nodes = [
        n for n, d in dist_s.items()
        if n != start_node and mid_min <= d <= mid_max
    ]
    via2_nodes = [
        n for n, d in dist_s.items()
        if n != start_node and via2_min <= d <= via2_max
    ]

    # ring 이 비면 조건을 조금 완화
    if not via1_nodes:
        via1_nodes = [
            n for n, d in dist_s.items()
            if n != start_node and quarter * 0.5 <= d <= quarter * 1.5
        ]
    if not mid_nodes:
        mid_nodes = [
            n for n, d in dist_s.items()
            if n != start_node and 2.0 * quarter * 0.5 <= d <= 2.0 * quarter * 1.5
        ]
    if not via2_nodes:
        via2_nodes = [
            n for n, d in dist_s.items()
            if n != start_node and 3.0 * quarter * 0.5 <= d <= 3.0 * quarter * 1.5
        ]

    if not via1_nodes or not mid_nodes or not via2_nodes:
        # PSP3 후보를 전혀 만들 수 없으면 바로 fallback
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
            message="PSP3 ring 후보를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 너무 많은 노드를 사용하면 조합 폭발이 발생하므로 샘플링
    random.shuffle(via1_nodes)
    random.shuffle(mid_nodes)
    random.shuffle(via2_nodes)
    via1_nodes = via1_nodes[:30]
    mid_nodes = mid_nodes[:30]
    via2_nodes = via2_nodes[:30]

    best_score = -1e18
    best_poly = None
    best_round = 0.0
    best_overlap = 0.0
    best_curve_penalty = 0.0
    best_err = 1e18
    routes_checked = 0
    routes_validated = 0

    # --------------------------
    # 3) PSP3 스타일: s → u → m → v → s 루프 후보 탐색
    #    - 모든 leg 는 최단경로
    #    - 전체 길이가 target_m 에 가깝고 roundness 가 높은 것 우선 선택
    # --------------------------
    MAX_TRIALS = 80
    for _ in range(MAX_TRIALS):
        u = random.choice(via1_nodes)
        m = random.choice(mid_nodes)
        v = random.choice(via2_nodes)

        # 서로 너무 가까운 노드는 제거 (쓸모없는 작은 루프 방지)
        if u == m or m == v or u == v:
            continue

        try:
            path_su = nx.shortest_path(undirected, start_node, u, weight="length")
            path_um = nx.shortest_path(undirected, u, m, weight="length")
            path_mv = nx.shortest_path(undirected, m, v, weight="length")
            path_vs = nx.shortest_path(undirected, v, start_node, weight="length")
        except Exception:
            # 한 구간이라도 실패하면 이 조합은 건너뜀
            continue

        # 노드 연결 시 중간 중복 제거
        full_nodes = (
            list(path_su)
            + list(path_um)[1:]
            + list(path_mv)[1:]
            + list(path_vs)[1:]
        )

        # 너무 짧은 루프는 제외
        full_len = _path_length_on_graph(undirected, full_nodes)
        if full_len <= 0:
            continue

        routes_checked += 1

        polyline = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(polyline)
        if length_m <= 0:
            continue

        err = abs(length_m - target_m)
        roundness = polygon_roundness(polyline)
        overlap = _edge_overlap_fraction(full_nodes)
        curve_penalty = _curve_penalty(full_nodes, undirected)

        # 길이 오차 / roundness / overlap / 커브 페널티를 합쳐 스코어 계산
        length_pen = err / max(target_m, 1.0)
        score = (
            roundness * 1.5
            - overlap * 2.0
            - curve_penalty * 0.3
            - length_pen * 3.0
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
            routes_checked=routes_checked,
            routes_validated=routes_validated,
            message="PSP3 기반 루프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 5) 최종 meta 구성
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
            "PSP3 스타일 OSM 보행자 그래프에서 러닝 루프를 생성했습니다."
            if success
            else "요청 오차(±99m)를 초과하지만, 가장 인접한 PSP3 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
