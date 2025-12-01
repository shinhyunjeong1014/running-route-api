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
    (반복 루프 생성 시 중복을 줄이기 위해 사용)
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
        theta = math.acos(dot)  # 0 ~ pi (180도)

        if theta < math.pi / 3.0:  # 60도 미만
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    """그래프 상에서 node 경로의 길이 (edge length 합)."""
    if len(nodes) < 2:
        return 0.0
    length = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            continue
        data = min(G[u][v].values(), key=lambda d: d.get("length", 1.0))
        length += float(data.get("length", 0.0))
    return length


# ==========================
# polyline smoothing (local search)
# ==========================
def _simplify_polyline_angle(
    polyline: List[Tuple[float, float]],
    angle_threshold_deg: float = 8.0,
) -> List[Tuple[float, float]]:
    """
    거의 일직선(180도±angle_threshold_deg) 구간의 중간점을 제거해서
    지그재그를 줄이는 단순 각도 기반 polyline simplification.

    - OSM 상 노드가 촘촘하게 찍힌 구간에서만 의미 있음
    - 실제 경로는 동일 도로 위에 있고, 시각적으로만 부드럽게 보이도록 함
    """
    if len(polyline) < 3:
        return polyline[:]

    kept = [polyline[0]]
    for i in range(1, len(polyline) - 1):
        p_prev = kept[-1]
        p = polyline[i]
        p_next = polyline[i + 1]

        v1x = p[1] - p_prev[1]
        v1y = p[0] - p_prev[0]
        v2x = p_next[1] - p[1]
        v2y = p_next[0] - p[0]

        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0 or n2 == 0:
            kept.append(p)
            continue

        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.degrees(math.acos(dot))  # 0~180

        # 거의 일직선(180도 근처)이면 중간점을 생략
        if abs(180.0 - theta) <= angle_threshold_deg:
            continue
        else:
            kept.append(p)

    kept.append(polyline[-1])
    return kept


# ==========================
# OSM 보행자 그래프 구축 (개선: walk + custom_filter)
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    OSMnx 'walk' 네트워크 기반 + custom_filter로 보행/생활도로 위주 필터링.

    포함:
      - footway, path, sidewalk, cycleway, steps, pedestrian,
        track, service, residential, living_street, alley
    (motorway/trunk 등 자동차 전용도로는 highway 값 자체가 위 리스트에 없으므로 자연스럽게 제외)
    """
    # km가 커질수록 반경 확대 (상한 4000m 정도로 제한)
    radius_m = min(max(800.0, km * 800.0 + 500.0), 4000.0)

    custom_filter = (
        '["highway"~"footway|path|sidewalk|cycleway|steps|pedestrian|track|'
        'service|residential|living_street|alley"]'
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

    # 고립 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))

    if not G.nodes:
        raise ValueError("Graph became empty after removing isolated nodes.")

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
    (실제 도로를 쓰지 않지만, 완전 실패 대비용이므로 예외적으로 허용)
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

    center_lat = (a[0] + c[0]) / 2
    center_lng = (b[1] + d[1]) / 2
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]

    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# 메인: CYCLE-HUNT + LOCAL SEARCH 러닝 루프 생성기
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    CYCLE-HUNT + LOCAL-SEARCH (보행자 전용 + 길이 ±5% 제약 강화 버전)

    1) 보행자 그래프 생성 (OSM walk + custom_filter)
    2) 시작 노드 주변 subgraph 추출
    3) cycle_basis로 사이클 후보 추출
    4) start → cycle 접근 경로 + cycle 반복(1~5회) + start 복귀 루프 구성
    5) 길이, roundness, overlap, curve_penalty로 스코어링
    6) 최종적으로 "요청 거리의 ±5% 이내"인 루트가 하나라도 있으면 그 중 최적을 반환,
       아니면 정사각형 fallback 루프 사용
    """
    start_time = time.time()
    target_m = km * 1000.0

    # 길이 허용 오차: 요청 거리의 ±5%
    tolerance_m = target_m * 0.05

    # 스코어링 가중치 (길이 페널티 강하게)
    ROUNDNESS_WEIGHT = 2.8
    OVERLAP_PENALTY = 1.5
    CURVE_PENALTY_WEIGHT = 0.22
    LENGTH_PENALTY_WEIGHT = 8.0  # 길이 오차에 훨씬 민감하게

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
            length_ok=(err <= tolerance_m),
            used_fallback=True,
            message=f"OSM 보행자 그래프 생성 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
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
            length_ok=(err <= tolerance_m),
            used_fallback=True,
            message=f"시작 노드 매칭 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # undirected MultiGraph
    UG: nx.MultiGraph = ox.utils_graph.get_undirected(G)

    # --------------------------
    # 2) start에서의 단일-출발 최단거리 (탐색 영역 제한)
    # --------------------------
    try:
        cutoff = min(target_m * 0.9, 3000.0)
        dist_map = nx.single_source_dijkstra_path_length(
            UG,
            start_node,
            cutoff=cutoff,
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
            length_ok=(err <= tolerance_m),
            used_fallback=True,
            message=f"그래프 최단거리 탐색 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    if not dist_map:
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
            length_ok=(err <= tolerance_m),
            used_fallback=True,
            message="최단거리 결과가 비어 있어 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    local_nodes = list(dist_map.keys())
    H_multi = UG.subgraph(local_nodes).copy()
    if H_multi.number_of_nodes() < 4:
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
            length_ok=(err <= tolerance_m),
            used_fallback=True,
            message="주변 그래프 노드가 너무 적어 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # cycle_basis 는 simple Graph 기준이므로 MultiGraph → Graph 변환
    H_simple = nx.Graph(H_multi)

    # --------------------------
    # 3) 사이클 탐색 (cycle_basis)
    # --------------------------
    try:
        cycles = nx.cycle_basis(H_simple, root=start_node)
    except Exception:
        cycles = nx.cycle_basis(H_simple)

    if not cycles:
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
            length_ok=(err <= tolerance_m),
            used_fallback=True,
            message="주변 보행자 그래프에서 유의미한 사이클을 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 사이클 개수가 너무 많으면 랜덤 샘플링
    random.shuffle(cycles)
    cycles = cycles[:120]

    # 사이클 길이 범위 (조금 넓게)
    min_cycle_len = target_m * 0.2
    max_cycle_len = target_m * 2.0

    # found_length_ok = 길이±5%를 만족하는 후보가 존재하는지
    found_length_ok = False
    best_score_any = -1e18
    best_poly_any: List[Tuple[float, float]] = []
    best_meta_any: Dict[str, Any] = {}

    best_score_len_ok = -1e18
    best_poly_len_ok: List[Tuple[float, float]] = []
    best_meta_len_ok: Dict[str, Any] = {}

    # --------------------------
    # 4) 각 사이클 + 접근 노드 + 반복 횟수(1~5회) 조합 탐색
    # --------------------------
    for cyc in cycles:
        if len(cyc) < 3:
            continue

        cyc_closed = cyc + [cyc[0]]
        cycle_len = _path_length_on_graph(UG, cyc_closed)
        if cycle_len <= 0:
            continue

        if cycle_len < min_cycle_len or cycle_len > max_cycle_len:
            continue

        access_candidates = [n for n in cyc if n in dist_map]
        if not access_candidates:
            continue

        access_candidates.sort(key=lambda n: dist_map[n])
        access_candidates = access_candidates[:4]

        for a in access_candidates:
            dist_a = dist_map[a]
            if dist_a > target_m * 0.7:
                continue

            try:
                path_to = nx.shortest_path(
                    UG,
                    start_node,
                    a,
                    weight="length",
                )
            except Exception:
                continue

            if len(path_to) < 2:
                continue

            if a in cyc:
                idx = cyc.index(a)
                cyc_rot = cyc[idx:] + cyc[:idx]
            else:
                cyc_rot = cyc[:]

            cyc_closed_rot = cyc_rot + [cyc_rot[0]]

            for rep in range(1, 6):  # 최대 5회 반복
                full_nodes: List[int] = []
                full_nodes.extend(path_to)

                for _ in range(rep):
                    full_nodes.extend(cyc_closed_rot[1:])

                back_path = list(reversed(path_to))
                if len(back_path) > 1:
                    full_nodes.extend(back_path[1:])

                meta["routes_checked"] += 1

                polyline = _nodes_to_polyline(UG, full_nodes)
                length_m = polyline_length_m(polyline)
                if length_m <= 0:
                    continue

                err = abs(length_m - target_m)
                roundness = polygon_roundness(polyline)
                overlap = _edge_overlap_fraction(full_nodes)
                curve_penalty = _curve_penalty(full_nodes, UG)

                length_ok = err <= tolerance_m
                if length_ok:
                    found_length_ok = True
                    meta["routes_validated"] += 1

                length_pen = err / max(target_m, 1.0)
                score = (
                    roundness * ROUNDNESS_WEIGHT
                    - overlap * OVERLAP_PENALTY
                    - curve_penalty * CURVE_PENALTY_WEIGHT
                    - length_pen * LENGTH_PENALTY_WEIGHT
                )

                # 1) 전체 후보 중 최고 (fallback 대비 정보용)
                if score > best_score_any:
                    best_score_any = score
                    best_poly_any = polyline
                    best_meta_any = {
                        "len": length_m,
                        "err": err,
                        "roundness": roundness,
                        "overlap": overlap,
                        "curve_penalty": curve_penalty,
                        "score": score,
                        "length_ok": length_ok,
                    }

                # 2) 길이±5% 만족 후보 중 최고
                if length_ok and score > best_score_len_ok:
                    best_score_len_ok = score
                    best_poly_len_ok = polyline
                    best_meta_len_ok = {
                        "len": length_m,
                        "err": err,
                        "roundness": roundness,
                        "overlap": overlap,
                        "curve_penalty": curve_penalty,
                        "score": score,
                        "length_ok": length_ok,
                    }

    # --------------------------
    # 5) 길이±5% 만족 후보가 있을 때 vs 없을 때
    # --------------------------
    if found_length_ok and best_poly_len_ok:
        # 길이 조건 만족 루트들 중 최적 후보 사용
        base_poly = best_poly_len_ok
        base_meta_local = best_meta_len_ok
        used_fallback = False
        success = True
        msg = "최적의 보행자용 CYCLE-HUNT + LOCAL-SEARCH 루트가 도출되었습니다."
    else:
        # 길이±5% 만족 루트가 하나도 없으면, 정사각형 fallback 사용
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
            length_ok=(err <= tolerance_m),
            used_fallback=True,
            message="요청 거리의 ±5% 이내 보행자용 루프를 찾지 못해 정사각형 fallback 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 6) polyline smoothing (LOCAL SEARCH) + 최종 메타
    # --------------------------
    smoothed_poly = _simplify_polyline_angle(base_poly, angle_threshold_deg=6.0)
    smoothed_len = polyline_length_m(smoothed_poly)
    smoothed_roundness = polygon_roundness(smoothed_poly)

    err = abs(smoothed_len - target_m)
    length_ok_final = err <= tolerance_m

    final_meta_local = dict(base_meta_local)
    final_meta_local.update(
        len=smoothed_len,
        err=err,
        roundness=smoothed_roundness,
        length_ok=length_ok_final,
    )

    meta.update(final_meta_local)
    meta.update(
        success=bool(length_ok_final),
        used_fallback=False,
        message=(
            "최적의 보행자용 CYCLE-HUNT + LOCAL-SEARCH 루트가 도출되었습니다."
            if length_ok_final
            else "길이 스무딩 과정에서 오차가 약간 늘었지만, 가장 인접한 CYCLE-HUNT + LOCAL-SEARCH 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(smoothed_poly)
    return safe_poly, safe_meta
