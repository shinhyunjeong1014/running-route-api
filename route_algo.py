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
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
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
    if area == 0.0 or perimeter == 0.0:
        return 0.0
    r = 4.0 * math.pi * area / (perimeter ** 2)
    if math.isinf(r) or math.isnan(r):
        return 0.0
    return r


# ==========================
# overlap / 커브 페널티
# ==========================
def _edge_overlap_fraction(node_path: List[int]) -> float:
    """
    노드 시퀀스에서 같은 간선을 여러 번 쓰는 비율.
    rod + detour 구조에서 중복을 줄이기 위해 사용.
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

        ya, xa = G.nodes[a].get("y"), G.nodes[a].get("x")
        yb, xb = G.nodes[b].get("y"), G.nodes[b].get("x")
        yc, xc = G.nodes[c].get("y"), G.nodes[c].get("x")

        if ya is None or yb is None or yc is None:
            continue
        if xa is None or xb is None or xc is None:
            continue

        v1x = xb - xa
        v1y = yb - ya
        v2x = xc - xb
        v2y = yc - yb
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0.0 or n2 == 0.0:
            continue

        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)  # 0 ~ pi (180도)

        # 너무 급한 커브(60도 미만, pi/3)에 페널티
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    """그래프 상에서 node 경로의 길이 (edge 의 length 합)."""
    if len(nodes) < 2:
        return 0.0
    length = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            continue
        data = G[u][v]
        # Graph 로 만들 것이므로 단일 edge dict
        edge_len = float(data.get("length", 0.0))
        if edge_len <= 0.0:
            continue
        length += edge_len
    return length


# ==========================
# OSM 보행자 그래프 구축 (필터 개선)
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    논문 기반 + RUNAMIC 아이디어를 구현하기 위한 보행자 그래프.

    - OSMnx의 network_type="walk" 그래프를 사용
    - 아파트 단지 내부 도로 / 주차장 통로 / 사유지 진입도로( driveways )를
      최대한 제외해서 "러닝에 적합한" 네트워크만 남긴다.
    """
    # km 가 커질수록 반경을 넉넉히 잡되, 너무 크게는 안 가게 제한
    radius_m = max(800.0, km * 700.0 + 500.0)

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )

    # edge length 보정 (없으면 계산)
    G = ox.add_edge_lengths(G)

    # 러닝에 비적합한 edge 제거
    remove_edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        hwy = data.get("highway")
        if isinstance(hwy, list):
            hwy = hwy[0]
        service = data.get("service")
        if isinstance(service, list):
            service = service[0]
        access = data.get("access")
        if isinstance(access, list):
            access = access[0]
        area = data.get("area")

        bad = False

        # 아파트 단지/주차장 내부 통로
        if service in {"driveway", "parking_aisle"}:
            bad = True

        # 사유지 접근용 service/residential + private
        if access in {"private", "no"} and hwy in {"service", "residential", "living_street"}:
            bad = True

        # 계단 구간은 러닝에서 비선호 (원하면 제거)
        if hwy == "steps":
            bad = True

        # area=yes 인 면적형 도로(광장 등)는 보수적으로 제거
        if area == "yes":
            bad = True

        if bad:
            remove_edges.append((u, v, k))

    if remove_edges:
        G.remove_edges_from(remove_edges)

    # 고립 노드 제거
    G = ox.utils_graph.remove_isolated_nodes(G)

    return G


# ==========================
# 노드 → polyline 변환
# ==========================
def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> List[Tuple[float, float]]:
    """그래프 노드 시퀀스를 (lat, lng) polyline으로 변환."""
    poly: List[Tuple[float, float]] = []
    for n in nodes:
        if n not in G.nodes:
            continue
        data = G.nodes[n]
        lat = data.get("y")
        lon = data.get("x")
        if lat is None or lon is None:
            continue
        poly.append((float(lat), float(lon)))
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
    delta_deg_lng = side / (111000.0 * cos_lat if cos_lat != 0.0 else 111000.0)

    # 원점 기준 대략적인 사각형
    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]

    # 시작점을 중심으로 재배치 (더 안정적인 형태)
    center_lat = (a[0] + c[0]) / 2.0
    center_lng = (b[1] + d[1]) / 2.0
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]

    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# route poisoning (RUNAMIC 스타일)
# ==========================
def _apply_route_poison(G: nx.Graph, path_nodes: List[int], factor: float = 5.0) -> nx.Graph:
    """
    rod 에 해당하는 간선의 비용(length)을 factor만큼 늘려
    되돌아올 때 같은 길을 반복해서 타는 것을 억제.
    """
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G2.has_edge(u, v):
            # 무방향 Graph 기준 single edge
            data = G2[u][v]
            if "length" in data:
                data["length"] = float(data["length"]) * factor
        if G2.has_edge(v, u):
            data = G2[v][u]
            if "length" in data:
                data["length"] = float(data["length"]) * factor
    return G2


# ==========================
# 메인: 러닝 루프 생성기 (rod + detour 하이브리드 v1.5)
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    RUNAMIC 스타일 rod + detour 기반 러닝 루프 생성기 (개선 버전).

    - 시작점 기준 OSM walk 그래프 구축 (아파트 단지/주차장/사유지 도로 필터링)
    - start -> endpoint rod (최단 경로) 여러 개 생성
    - rod 간선에 penalty 를 주고 endpoint -> start detour 경로 탐색
    - rod + detour 로 폐곡선 루프 구성
    - 길이 오차 / roundness / overlap / curve_penalty / 중심 거리 등을 종합해서
      가장 "러닝 친화적인" 루프 선택
    - 그래프/경로 탐색 실패 시, 출발 좌표 중심 사각형 루프로 fallback
    """
    start_time = time.time()
    target_m = km * 1000.0
    tol_m = 99.0

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
        "snap_dist_m": None,
        "time_s": None,
        "message": "",
    }

    # --------------------------
    # 1) OSM 보행자 그래프 구축
    # --------------------------
    try:
        G_raw = _build_pedestrian_graph(lat, lng, km)
        if not G_raw.nodes or not G_raw.edges:
            raise ValueError("Empty pedestrian graph.")
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
            message=f"OSM 보행자 그래프 생성 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # undirected 단순 그래프로 변환
    UG_multi: nx.MultiGraph = ox.utils_graph.get_undirected(G_raw)
    G = nx.Graph()
    # 노드 복사
    for n, data in UG_multi.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            continue
        G.add_node(n, x=float(x), y=float(y))
    # 가장 짧은 edge 하나만 사용
    for u, v, k, data in UG_multi.edges(keys=True, data=True):
        if u not in G.nodes or v not in G.nodes:
            continue
        length = float(data.get("length", 0.0))
        if length <= 0.0:
            continue
        if G.has_edge(u, v):
            if length < G[u][v].get("length", 1e18):
                G[u][v]["length"] = length
        else:
            G.add_edge(u, v, length=length)

    if not G.nodes or not G.edges:
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
            message="보행자 그래프 단순화 실패로 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 2) start 노드 찾기 + snap 거리 기록
    # --------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
        n_data = G.nodes[start_node]
        n_lat = float(n_data.get("y", lat))
        n_lng = float(n_data.get("x", lng))
        snap_dist = haversine(lat, lng, n_lat, n_lng)
        meta["snap_dist_m"] = snap_dist
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
    # 3) start 에서의 단일-출발 최단거리 (rod 후보 탐색)
    # --------------------------
    try:
        dist = nx.single_source_dijkstra_path_length(
            G,
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
            length_ok=(err <= tol_m),
            used_fallback=True,
            message=f"그래프 최단거리 탐색 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # rod 길이 후보 범위 조정
    min_leg = target_m * 0.35
    if target_m <= 2500.0:
        max_leg = target_m * 0.55  # 짧은 루프일수록 rod 를 줄여서 spiky 방지
    else:
        max_leg = target_m * 0.65

    candidate_nodes = [n for n, d in dist.items() if min_leg <= d <= max_leg and n != start_node]

    # 후보가 너무 적으면 조건 완화
    if not candidate_nodes:
        candidate_nodes = [n for n, d in dist.items() if d >= target_m * 0.25]

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
            length_ok=(err <= tol_m),
            used_fallback=True,
            message="적절한 rod endpoint 후보를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:60]

    best_poly_len_ok: List[Tuple[float, float]] = []
    best_meta_len_ok: Dict[str, Any] = {}
    best_score_len_ok = -1e18

    best_poly_any: List[Tuple[float, float]] = []
    best_meta_any: Dict[str, Any] = {}
    best_score_any = -1e18

    routes_checked = 0
    routes_validated = 0

    # --------------------------
    # 4) 각 endpoint 에 대해 rod + detour 루프 생성
    # --------------------------
    for endpoint in candidate_nodes:
        # 4-1) start -> endpoint rod (최단 경로)
        try:
            forward_nodes = nx.shortest_path(
                G,
                start_node,
                endpoint,
                weight="length",
            )
        except Exception:
            continue

        forward_len = _path_length_on_graph(G, forward_nodes)
        if forward_len <= 0.0:
            continue

        # 4-2) rod 간선에 penalty 를 줘서 detour 경로 유도 (RUNAMIC 아이디어)
        poisoned = _apply_route_poison(G, forward_nodes, factor=5.0)

        # endpoint -> start detour
        try:
            back_nodes = nx.shortest_path(
                poisoned,
                endpoint,
                start_node,
                weight="length",
            )
        except Exception:
            continue

        back_len = _path_length_on_graph(G, back_nodes)
        if back_len <= 0.0:
            continue

        # full loop = rod + detour (노드 중복 방지 위해 back_nodes[1:]부터 이어붙임)
        full_nodes = forward_nodes + back_nodes[1:]
        routes_checked += 1

        polyline = _nodes_to_polyline(G, full_nodes)
        if len(polyline) < 4:
            continue

        length_m = polyline_length_m(polyline)
        if length_m <= 0.0:
            continue

        err = abs(length_m - target_m)
        roundness = polygon_roundness(polyline)
        overlap = _edge_overlap_fraction(full_nodes)
        curve_pen = _curve_penalty(full_nodes, G)

        # 루프 중심 및 반경 기반 페널티 (spiky / 지나치게 멀리 나가는 루프 억제)
        avg_lat = sum(p[0] for p in polyline) / len(polyline)
        avg_lng = sum(p[1] for p in polyline) / len(polyline)
        center_dist = haversine(lat, lng, avg_lat, avg_lng)

        max_radial = max(haversine(lat, lng, p[0], p[1]) for p in polyline)

        # 길이 오차 / roundness / overlap / 커브 / 중심/반경 페널티를 합쳐 스코어 계산
        length_pen = err / target_m if target_m > 0.0 else 0.0
        center_pen = center_dist / max(target_m, 1.0)
        # 짧은 루프일수록 너무 멀리 튀어나가는 형태를 더 강하게 억제
        radial_pen = (max_radial / max(target_m, 1.0)) * (1.5 if km <= 3.0 else 0.7)

        score = (
            roundness * 2.5
            - overlap * 2.0
            - curve_pen * 0.25
            - length_pen * 4.0
            - center_pen * 1.0
            - radial_pen * 1.0
        )

        length_ok = (err <= tol_m)
        if length_ok:
            routes_validated += 1

        # 1) 허용오차 내 route 중 최적
        if length_ok and score > best_score_len_ok:
            best_score_len_ok = score
            best_poly_len_ok = polyline
            best_meta_len_ok = {
                "len": length_m,
                "err": err,
                "roundness": roundness,
                "overlap": overlap,
                "curve_penalty": curve_pen,
                "score": score,
                "length_ok": True,
            }

        # 2) 전체 route 중 최적 (fallback 용)
        if score > best_score_any:
            best_score_any = score
            best_poly_any = polyline
            best_meta_any = {
                "len": length_m,
                "err": err,
                "roundness": roundness,
                "overlap": overlap,
                "curve_penalty": curve_pen,
                "score": score,
                "length_ok": length_ok,
            }

    # --------------------------
    # 5) 후보 루프 선택 (길이 충족 우선)
    # --------------------------
    if best_poly_len_ok:
        best_poly = best_poly_len_ok
        best_meta = best_meta_len_ok
    elif best_poly_any:
        best_poly = best_poly_any
        best_meta = best_meta_any
    else:
        # 루프를 하나도 못 만들면 fallback
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
            routes_checked=routes_checked,
            routes_validated=routes_validated,
            message="rod + detour 기반 러닝 루프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 6) 최종 meta 구성
    # --------------------------
    length_ok = best_meta.get("length_ok", False)
    success = length_ok

    meta.update(best_meta)
    meta.update(
        success=success,
        used_fallback=False,
        routes_checked=routes_checked,
        routes_validated=routes_validated,
        message=(
            "논문 기반 OSM 보행자 그래프에서 러닝 루프를 생성했습니다."
            if success
            else "요청 오차(±99m)를 초과하지만, 가장 인접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
