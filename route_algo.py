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
# OSM 보행자 그래프 구축 (Cycle-Hunt용)
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    OSMnx 'walk' 네트워크를 기반으로 보행 가능 그래프 생성.
    - 너무 공격적인 필터는 넣지 않고,
    - 고속도로 계열만 한 번 더 제거.
    """
    # km가 커질수록 반경 확대 (상한 4000m)
    radius_m = min(max(800.0, km * 800.0 + 500.0), 4000.0)

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )

    if not G.nodes:
        raise ValueError("OSM walk graph has no nodes in this area.")

    # motorway / trunk 계열 한 번 더 제거
    remove_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        hwy = data.get("highway")
        if hwy in ("motorway", "motorway_link", "trunk", "trunk_link"):
            remove_edges.append((u, v, key))
    for u, v, key in remove_edges:
        G.remove_edge(u, v, key)

    # 고립 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))

    if not G.nodes:
        raise ValueError("Graph became empty after filtering.")

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

    center_lat = (a[0] + c[0]) / 2
    center_lng = (b[1] + d[1]) / 2
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]

    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# 메인: Cycle-Hunt 러닝 루프 생성기
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    CYCLE-HUNT v1: 사이클 기반 러닝 루프 생성기.

    - Rod / Poison 없이 순수 Cycle 기반
    - 시작 노드에서 접근 가능한 서브그래프에서 사이클 탐색
    - 한 사이클을 1~3회 반복하여 목표 거리 근처까지 확장
    - 시작점에서 출발/복귀하는 완전 루프
    """
    start_time = time.time()
    target_m = km * 1000.0
    tolerance_m = 99.0  # 길이 허용 오차

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
        # 사이클 탐색용 cutoff: target_m * 0.6 ~ 0.8 정도가 적당
        cutoff = min(target_m * 0.8, 2500.0)
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
    # root를 start_node로 주면 start 주변 사이클을 우선 탐색
    try:
        cycles = nx.cycle_basis(H_simple, root=start_node)
    except Exception:
        # root가 서브그래프에 없는 경우 등 → 그냥 전체 cycle_basis
        cycles = nx.cycle_basis(H_simple)

    if not cycles:
        # 사이클이 전혀 없는 tree 구조면 fallback
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
    cycles = cycles[:80]

    min_cycle_len = target_m * 0.3
    max_cycle_len = target_m * 1.8

    best_score = -1e18
    best_poly: List[Tuple[float, float]] = []
    best_meta_local: Dict[str, Any] = {}
    found_length_ok = False

    # --------------------------
    # 4) 각 사이클 + 접근 노드 + 반복 횟수(1~3회) 조합 탐색
    # --------------------------
    for cyc in cycles:
        if len(cyc) < 3:
            continue

        # 먼저 사이클 자체의 길이를 계산 (닫힌 형태)
        cyc_closed = cyc + [cyc[0]]
        cycle_len = _path_length_on_graph(UG, cyc_closed)
        if cycle_len <= 0:
            continue

        if cycle_len < min_cycle_len or cycle_len > max_cycle_len:
            # 너무 짧거나 / 너무 긴 사이클은 제외
            continue

        # 이 사이클 안에서 start로부터 접근 가능한 노드 후보 찾기
        access_candidates = [n for n in cyc if n in dist_map]
        if not access_candidates:
            continue

        # start에서 가까운 순으로 정렬 후 상위 몇 개만 사용
        access_candidates.sort(key=lambda n: dist_map[n])
        access_candidates = access_candidates[:3]

        for a in access_candidates:
            dist_a = dist_map[a]
            # 접근 거리가 타깃의 70%를 넘으면 비효율적이므로 제외
            if dist_a > target_m * 0.7:
                continue

            # start → a 최단 경로
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

            # 사이클을 a에서 시작하도록 회전
            if a in cyc:
                idx = cyc.index(a)
                cyc_rot = cyc[idx:] + cyc[:idx]
            else:
                cyc_rot = cyc[:]  # 혹시 모를 방어

            cyc_closed_rot = cyc_rot + [cyc_rot[0]]  # 닫힌 사이클

            # 사이클을 1~3번 반복하며 길이 맞추기
            for rep in range(1, 4):
                full_nodes: List[int] = []

                # start → a
                full_nodes.extend(path_to)

                # a 기준 사이클 rep회 반복
                for _ in range(rep):
                    # 이미 a에서 끝나 있으므로, 다음 노드부터 추가
                    full_nodes.extend(cyc_closed_rot[1:])

                # 다시 a로 돌아온 상태이므로, start로 복귀 (역방향)
                back_path = list(reversed(path_to))
                # 첫 노드는 a(이미 있기 때문에 제외), 나머지 추가
                if len(back_path) > 1:
                    full_nodes.extend(back_path[1:])

                meta["routes_checked"] += 1

                # 길이 및 품질 평가
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

                # 길이 오차/roundness/중복/급커브를 모두 반영한 스코어
                length_pen = err / max(target_m, 1.0)
                score = (
                    roundness * 2.5
                    - overlap * 1.5
                    - curve_penalty * 0.25
                    - length_pen * 4.0
                )

                # 길이 만족 경로를 아직 못 찾았으면 length_ok 여부와 상관없이 best 갱신
                # 길이 만족 경로를 이미 찾은 후에는 length_ok인 후보만 비교
                if (not found_length_ok) or length_ok:
                    if score > best_score:
                        best_score = score
                        best_poly = polyline
                        best_meta_local = {
                            "len": length_m,
                            "err": err,
                            "roundness": roundness,
                            "overlap": overlap,
                            "curve_penalty": curve_penalty,
                            "score": score,
                            "length_ok": length_ok,
                        }

    # --------------------------
    # 5) 후보 루프가 하나도 없을 때
    # --------------------------
    if not best_poly:
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
            message="주변 보행자 그래프에서 유의미한 사이클 기반 루프를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 6) 최종 meta 구성
    # --------------------------
    success = bool(best_meta_local.get("length_ok", False))
    meta.update(best_meta_local)
    meta.update(
        success=success,
        used_fallback=False,
        message=(
            "최적의 정밀 CYCLE-HUNT 루프가 도출되었습니다."
            if success
            else "요청 오차(±99m)를 약간 초과하지만, 가장 인접한 CYCLE-HUNT 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
