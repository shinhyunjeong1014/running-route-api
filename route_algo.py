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
    """4πA / P^2 로 정의되는 roundness."""
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
    """노드 시퀀스에서 같은 간선을 여러 번 쓰는 비율."""
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
    """경로에서 너무 급격한 커브(예: 60도 이하)를 얼마나 많이 만드는지 측정."""
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
        theta = math.acos(dot)
        if theta < math.pi / 3.0:
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


def _apply_route_poison(G: nx.Graph, path_nodes: List[int], factor: float = 6.0) -> nx.Graph:
    """rod 간선의 length를 factor배로 늘려 detour를 유도."""
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G2.has_edge(u, v):
            for key in list(G2[u][v].keys()):
                data = G2[u][v][key]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
        if G2.has_edge(v, u):
            for key in list(G2[v][u].keys()):
                data = G2[v][u][key]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
    return G2


# ==========================
# OSM 보행자 그래프 구축 (공용)
# ==========================

def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """OSMnx walk 네트워크 기반 보행 가능 그래프."""
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
    # 고속도로 계열 제거 (혹시 남아있는 경우)
    remove_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        hwy = data.get("highway")
        if hwy in ("motorway", "motorway_link", "trunk", "trunk_link"):
            remove_edges.append((u, v, key))
    for u, v, key in remove_edges:
        G.remove_edge(u, v, key)
    G.remove_nodes_from(list(nx.isolates(G)))
    if not G.nodes:
        raise ValueError("Graph became empty after filtering.")
    return G


def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> List[Tuple[float, float]]:
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
# CYCLE-HUNT 후보 생성
# ==========================

def _generate_cycle_candidates(
    UG: nx.MultiGraph,
    start_node: int,
    target_m: float,
    tolerance_m: float,
    meta: Dict[str, Any],
) -> list:
    candidates = []
    # 1) start 기준 탐색 반경
    cutoff = min(target_m * 0.8, 2500.0)
    try:
        dist_map = nx.single_source_dijkstra_path_length(
            UG, start_node, cutoff=cutoff, weight="length"
        )
    except Exception:
        return candidates
    if not dist_map:
        return candidates
    local_nodes = list(dist_map.keys())
    H_multi = UG.subgraph(local_nodes).copy()
    if H_multi.number_of_nodes() < 4:
        return candidates
    H_simple = nx.Graph(H_multi)
    try:
        cycles = nx.cycle_basis(H_simple, root=start_node)
    except Exception:
        cycles = nx.cycle_basis(H_simple)
    if not cycles:
        return candidates
    random.shuffle(cycles)
    cycles = cycles[:80]
    min_cycle_len = target_m * 0.3
    max_cycle_len = target_m * 1.8
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
        access_candidates = access_candidates[:3]
        for a in access_candidates:
            dist_a = dist_map[a]
            if dist_a > target_m * 0.7:
                continue
            try:
                path_to = nx.shortest_path(UG, start_node, a, weight="length")
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
            for rep in range(1, 4):
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
                    meta["routes_validated"] += 1
                candidates.append(
                    {
                        "polyline": polyline,
                        "nodes": full_nodes,
                        "len": length_m,
                        "err": err,
                        "roundness": roundness,
                        "overlap": overlap,
                        "curve_penalty": curve_penalty,
                        "length_ok": length_ok,
                        "source": "cycle",
                    }
                )
    return candidates


# ==========================
# ROD + DETOUR 후보 생성 (RUNAMIC 스타일)
# ==========================

def _generate_rod_candidates(
    UG: nx.MultiGraph,
    start_node: int,
    target_m: float,
    tolerance_m: float,
    meta: Dict[str, Any],
) -> list:
    candidates = []
    try:
        dist = nx.single_source_dijkstra_path_length(
            UG, start_node, cutoff=target_m * 0.8, weight="length"
        )
    except Exception:
        return candidates
    if not dist:
        return candidates
    min_leg = target_m * 0.35
    max_leg = target_m * 0.60
    candidate_nodes = [n for n, d in dist.items() if min_leg <= d <= max_leg and n != start_node]
    if not candidate_nodes:
        candidate_nodes = [n for n, d in dist.items() if d >= target_m * 0.25]
    if not candidate_nodes:
        return candidates
    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:40]
    for endpoint in candidate_nodes:
        try:
            forward_nodes = nx.shortest_path(UG, start_node, endpoint, weight="length")
        except Exception:
            continue
        forward_len = _path_length_on_graph(UG, forward_nodes)
        if forward_len <= 0:
            continue
        poisoned = _apply_route_poison(UG, forward_nodes, factor=6.0)
        try:
            back_nodes = nx.shortest_path(poisoned, endpoint, start_node, weight="length")
        except Exception:
            continue
        back_len = _path_length_on_graph(UG, back_nodes)
        if back_len <= 0:
            continue
        full_nodes = forward_nodes + back_nodes[1:]
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
            meta["routes_validated"] += 1
        candidates.append(
            {
                "polyline": polyline,
                "nodes": full_nodes,
                "len": length_m,
                "err": err,
                "roundness": roundness,
                "overlap": overlap,
                "curve_penalty": curve_penalty,
                "length_ok": length_ok,
                "source": "rod",
            }
        )
    return candidates


# ==========================
# 메인: HYBRID 러닝 루프 생성기
# ==========================

def generate_area_loop(lat: float, lng: float, km: float):
    """CYCLE + ROD 하이브리드 러닝 루프 생성기."""
    start_time = time.time()
    target_m = km * 1000.0
    tolerance_m = 99.0

    # 스코어링 가중치 (두 방식 공통)
    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 1.5
    CURVE_PENALTY_WEIGHT = 0.25
    LENGTH_PENALTY_WEIGHT = 4.0

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

    # 1) 그래프 구축 및 시작 노드 매칭
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

    UG: nx.MultiGraph = ox.utils_graph.get_undirected(G)

    # 2) 두 방식 모두에서 후보 생성
    cycle_candidates = _generate_cycle_candidates(UG, start_node, target_m, tolerance_m, meta)
    rod_candidates = _generate_rod_candidates(UG, start_node, target_m, tolerance_m, meta)
    all_candidates = cycle_candidates + rod_candidates

    if not all_candidates:
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
            message="CYCLE/ROD 기반 루프 생성에 모두 실패하여 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 3) 스코어 계산 및 best 선택
    best_score = -1e18
    best = None
    found_length_ok = any(c["length_ok"] for c in all_candidates)

    for c in all_candidates:
        err = c["err"]
        length_pen = err / max(target_m, 1.0)
        score = (
            c["roundness"] * ROUNDNESS_WEIGHT
            - c["overlap"] * OVERLAP_PENALTY
            - c["curve_penalty"] * CURVE_PENALTY_WEIGHT
            - length_pen * LENGTH_PENALTY_WEIGHT
        )
        c["score"] = score
        if (not found_length_ok) or c["length_ok"]:
            if score > best_score:
                best_score = score
                best = c

    if best is None:
        c = max(all_candidates, key=lambda x: x.get("score", -1e18))
        best = c
        best_score = best.get("score", -1e18)

    success = best["length_ok"]

    meta.update(
        len=best["len"],
        err=best["err"],
        roundness=best["roundness"],
        overlap=best["overlap"],
        curve_penalty=best["curve_penalty"],
        score=best_score,
        success=success,
        length_ok=best["length_ok"],
        used_fallback=False,
        message=(
            f"HYBRID-{best['source'].upper()} 기반 최적 루프가 도출되었습니다."
            if success
            else f"요청 오차(±{tolerance_m}m)를 초과하지만, 가장 인접한 HYBRID-{best['source'].upper()} 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best["polyline"])
    return safe_poly, safe_meta
