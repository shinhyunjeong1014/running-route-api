from __future__ import annotations

import heapq
import math
import random
import time
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx

try:
    import osmnx as ox
except Exception:  # 배포 환경에서 import 실패 대비
    ox = None

LatLng = Tuple[float, float]
Polyline = List[LatLng]

# -----------------------------
# 기본 상수
# -----------------------------
MIN_LOOP_M = 200.0          # 최소 루프 길이 (m)
MAX_OSMNX_RADIUS_M = 2500.0 # OSM 그래프 조회 최대 반경
MIN_OSMNX_RADIUS_M = 600.0  # OSM 그래프 조회 최소 반경

LENGTH_TOL_FRAC = 0.05      # 목표 거리 허용 오차 비율 (±5%)
SECTOR_COUNT = 12           # via-node를 뽑을 각도 섹터 개수
MAX_VIA_TOTAL = 6           # 전체 via-node 상한 (double-loop 대비)
MAX_VIA_PAIRS = 15          # 평가할 via 쌍 최대 개수

POISON_FACTOR = 3.0         # 왕복 경로 억제를 위한 가중치 배수
MAX_MICRO_LOOPS = 3         # 거리 보정 micro-loop 최대 삽입 횟수


# ==========================
# JSON-safe 유틸
# ==========================

def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return default
    return x


def safe_list(lst: Any) -> list:
    if not isinstance(lst, (list, tuple)):
        return []
    out = []
    for v in lst:
        if isinstance(v, (list, tuple)):
            out.append([safe_float(x, None) for x in v])
        else:
            out.append(v)
    return out


def safe_dict(d: Any) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = safe_float(v, None)
        elif isinstance(v, (list, tuple)):
            out[k] = safe_list(v)
        elif isinstance(v, dict):
            out[k] = safe_dict(v)
        else:
            out[k] = v
    return out


# ==========================
# 거리 계산 유틸
# ==========================

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 좌표 사이의 대략적인 거리(m)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(poly: Polyline) -> float:
    """단순 위경도 polyline 길이(m). (fallback 경로 등에서 사용)"""
    if not poly or len(poly) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(poly[:-1], poly[1:]):
        total += haversine_m(la1, lo1, la2, lo2)
    return total


# ==========================
# fallback: 기하학적 사각형 루프
# ==========================

def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """OSM/그래프를 전혀 쓰지 못할 때 사용하는 단순 정사각형 루프."""
    target_m = max(MIN_LOOP_M, km * 1000.0)
    side = target_m / 4.0

    d_lat = side / 111111.0
    d_lng = side / (111111.0 * max(math.cos(math.radians(lat)), 1e-6))

    p1 = (lat, lng)
    p2 = (lat, lng + d_lng)
    p3 = (lat + d_lat, lng + d_lng)
    p4 = (lat + d_lat, lng)
    p5 = (lat, lng)

    poly = [p1, p2, p3, p4, p5]
    length_m = polyline_length_m(poly)
    err = abs(length_m - target_m)
    return poly, length_m, err


# ==========================
# OSM 그래프 생성 / 전처리
# ==========================

def _build_osm_graph(lat: float, lng: float, target_m: float) -> Tuple[nx.Graph, int]:
    """OSMnx 보행자 그래프를 만들고 단순 무방향 그래프로 변환."""
    if ox is None:
        raise RuntimeError("osmnx 가 설치되어 있지 않습니다.")

    # 요청 거리와 좌표가 항상 다르다는 점을 고려해,
    # 목표 거리 기반으로 "적당한" 반경을 설정 (지형에 따라 조금 작은 그래프일 수는 있음)
    radius = max(MIN_OSMNX_RADIUS_M, min(MAX_OSMNX_RADIUS_M, target_m * 0.7))

    G_raw = ox.graph_from_point(
        (lat, lng),
        dist=radius,
        network_type="walk",
        simplify=True,
    )

    G_undirected = ox.utils_graph.get_undirected(G_raw)
    G_undirected = ox.distance.add_edge_lengths(G_undirected)

    # MultiGraph -> Graph (가장 짧은 edge 선택)
    G = nx.Graph()
    for u, v, data in G_undirected.edges(data=True):
        length = data.get("length", 1.0)
        if G.has_edge(u, v):
            if length < G[u][v]["length"]:
                G[u][v]["length"] = length
        else:
            G.add_edge(u, v, length=length)

    for n, data in G_undirected.nodes(data=True):
        if n not in G:
            G.add_node(n)
        G.nodes[n]["y"] = data.get("y")
        G.nodes[n]["x"] = data.get("x")

    start_node = ox.distance.nearest_nodes(G_undirected, lng, lat)
    if start_node not in G:
        raise RuntimeError("시작 노드를 그래프에서 찾을 수 없습니다.")

    return G, start_node


# ==========================
# via-node 후보 선택 (PCD-lite, sector 기반)
# ==========================

def _select_via_candidates(
    G: nx.Graph,
    start_node: int,
    target_m: float,
    sectors: int = SECTOR_COUNT,
    max_total: int = MAX_VIA_TOTAL,
) -> List[int]:
    """시작 노드 기준 각도 섹터별 대표 노드를 via-node 후보로 선택."""
    sy = G.nodes[start_node]["y"]
    sx = G.nodes[start_node]["x"]

    # 원형 루프의 반지름 근사: L ≈ 2πR → R ≈ L / (2π)
    rough_radius = target_m / (2 * math.pi)

    # 요청 거리와 좌표가 매번 다르므로, 지형에 따라 조금 유연하게 링 폭을 설정
    min_r = max(rough_radius * 0.8, 200.0)
    max_r = rough_radius * 1.3  # 약간 더 여유를 줌

    sector_best: Dict[int, Tuple[int, float]] = {}

    for n, data in G.nodes(data=True):
        if n == start_node:
            continue
        lat = data.get("y")
        lng = data.get("x")
        if lat is None or lng is None:
            continue
        d = haversine_m(sy, sx, lat, lng)
        if d < min_r or d > max_r:
            continue

        dy = lat - sy
        dx = lng - sx
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi

        sector = int(sectors * angle / (2 * math.pi))
        degree = G.degree[n]

        score = (d / max_r) + 0.3 * (degree / max(1, G.degree[start_node]))

        prev = sector_best.get(sector)
        if prev is None or score > prev[1]:
            sector_best[sector] = (n, score)

    candidates = [v for (v, _) in sector_best.values()]
    random.shuffle(candidates)

    if len(candidates) > max_total:
        candidates = candidates[:max_total]

    return candidates


# ==========================
# A* 기반 최단 경로 (실거리 + poison 지원)
# ==========================

def _astar_path(
    G: nx.Graph,
    src: int,
    dst: int,
    poison_edges: Optional[set] = None,
) -> Tuple[float, List[int]]:
    """직선거리 heuristic + poison edge 가중치가 들어간 A*.
    반환 거리는 '실제 거리(미터)' 기준.
    """
    if src == dst:
        return 0.0, [src]

    poison_edges = poison_edges or set()

    def heuristic(n: int) -> float:
        ny = G.nodes[n]["y"]
        nx_ = G.nodes[n]["x"]
        dy = G.nodes[dst]["y"]
        dx_ = G.nodes[dst]["x"]
        return haversine_m(ny, nx_, dy, dx_)

    open_heap: List[Tuple[float, int]] = []
    heapq.heappush(open_heap, (heuristic(src), src))

    # g_cost: 탐색 비용(포이즌 가중치 포함), g_dist: 실제 거리
    g_cost: Dict[int, float] = {src: 0.0}
    g_dist: Dict[int, float] = {src: 0.0}
    came_from: Dict[int, int] = {}

    visited = set()

    while open_heap:
        f, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == dst:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return g_dist[dst], path

        for neighbor in G.neighbors(current):
            base_len = G[current][neighbor]["length"]
            e = (min(current, neighbor), max(current, neighbor))
            w = base_len * (POISON_FACTOR if e in poison_edges else 1.0)

            tentative_cost = g_cost[current] + w
            if tentative_cost >= g_cost.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_cost[neighbor] = tentative_cost
            g_dist[neighbor] = g_dist[current] + base_len
            f_score = tentative_cost + heuristic(neighbor)
            heapq.heappush(open_heap, (f_score, neighbor))

    raise nx.NetworkXNoPath(f"A* 경로를 찾지 못했습니다: {src} -> {dst}")


# ==========================
# 루프 경로 품질 평가
# ==========================

def _compute_roundness(poly: Polyline) -> float:
    """루프의 '둥근 정도'를 0~1로 평가."""
    if len(poly) < 3:
        return 0.0
    ys = [p[0] for p in poly]
    xs = [p[1] for p in poly]
    cy = sum(ys) / len(ys)
    cx = sum(xs) / len(xs)

    dists = [haversine_m(cy, cx, y, x) for y, x in poly]
    mean = sum(dists) / len(dists)
    if mean <= 0:
        return 0.0
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    std = math.sqrt(var)
    score = max(0.0, 1.0 - std / (mean + 1e-6))
    return min(score, 1.0)


def _compute_overlap_ratio(node_path: List[int]) -> float:
    """같은 간선을 여러 번 쓰는 비율 (0에 가까울수록 좋음)."""
    if not node_path or len(node_path) < 2:
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


def _compute_curve_penalty(poly: Polyline) -> float:
    """급커브가 많을수록 penalty 증가."""
    if len(poly) < 3:
        return 0.0

    def bearing(p1: LatLng, p2: LatLng) -> float:
        lat1, lon1 = map(math.radians, p1)
        lat2, lon2 = map(math.radians, p2)
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        brng = math.degrees(math.atan2(y, x))
        return (brng + 360.0) % 360.0

    turns = 0
    sharp = 0
    for a, b, c in zip(poly[:-2], poly[1:-1], poly[2:]):
        br1 = bearing(a, b)
        br2 = bearing(b, c)
        diff = abs(br2 - br1)
        if diff > 180.0:
            diff = 360.0 - diff
        if diff < 10.0:
            continue
        turns += 1
        if diff >= 60.0:
            sharp += 1
    if turns == 0:
        return 0.0
    return sharp / turns


# ==========================
# 거리 보정용 micro-loop 삽입
# ==========================

def _extend_with_micro_loops(
    G: nx.Graph,
    node_path: List[int],
    current_len: float,
    target_m: float,
    max_loops: int = MAX_MICRO_LOOPS,
) -> Tuple[List[int], float]:
    """
    루프 길이가 목표보다 짧을 때,
    경로 상의 교차로에서 옆 골목으로 들어갔다 나오는 micro-loop(u->w->u)를
    최대 max_loops번까지 삽입해서 거리 보정.
    - 실제 도로 그래프 위에서만 움직임.
    """
    if not node_path or len(node_path) < 2:
        return node_path, current_len

    new_node_path = list(node_path)
    total_len = current_len

    for _ in range(max_loops):
        missing = target_m - total_len
        if missing <= target_m * LENGTH_TOL_FRAC:
            break  # 이미 허용 오차 이내

        best_idx = None
        best_neighbor = None
        best_new_len = None
        best_err = float("inf")

        # 경로 중간 노드들에서 micro-loop 후보 탐색
        for i in range(1, len(new_node_path) - 1):
            u = new_node_path[i]
            # 교차로(연결이 여러 개) 위주로 시도
            if G.degree[u] < 3:
                continue

            for w in G.neighbors(u):
                # 기존 경로의 직전/직후 노드는 제외 (이미 사용 중인 방향)
                if w == new_node_path[i - 1] or w == new_node_path[i + 1]:
                    continue
                base_len = G[u][w]["length"]
                extra = 2.0 * base_len  # u->w->u
                candidate_len = total_len + extra
                # 너무 많이 overshoot 하면 skip
                if candidate_len > target_m * (1.0 + LENGTH_TOL_FRAC):
                    continue
                err = abs(candidate_len - target_m)
                if err < best_err:
                    best_err = err
                    best_idx = i
                    best_neighbor = w
                    best_new_len = candidate_len

        if best_idx is None or best_neighbor is None or best_new_len is None:
            # 더 이상 쓸만한 micro-loop 없음
            break

        # 실제로 node_path에 micro-loop 삽입: ... u, w, u, next ...
        insert_pos = best_idx + 1
        u = new_node_path[best_idx]
        new_node_path = (
            new_node_path[:insert_pos] + [best_neighbor, u] + new_node_path[insert_pos:]
        )
        total_len = best_new_len

    return new_node_path, total_len


# ==========================
# 메인: 러닝 루프 생성 (double via + 거리보정)
# ==========================

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """double-via 기반 러닝 루프 + 거리보정 micro-loop.

    - via-node 2개(A,B)를 조합해 start→A→B→start 루프를 만든다.
    - PCD/goal-directed A*를 이용해 탐색을 줄이되,
      거리 정밀도와 루프 모양을 동시에 고려한다.
    - 루프가 목표 거리보다 짧으면 실제 도로 위에서 micro-loop를 삽입해 거리 보정.
    """
    start_time = time.time()
    target_m = max(MIN_LOOP_M, km * 1000.0)

    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.4
    ERROR_WEIGHT = 4.0  # 길이 오차 비중을 조금 더 강화

    meta: Dict[str, Any] = {
        "status": "init",
        "len": 0.0,
        "err": None,
        "roundness": None,
        "overlap": None,
        "curve_penalty": None,
        "score": None,
        "success": False,
        "length_ok": False,
        "used_fallback": False,
        "routes_checked": 0,
        "routes_validated": 0,
        "via_candidates": [],
        "via_pairs": [],
        "message": "",
    }

    # 1) 그래프 생성
    try:
        G, start_node = _build_osm_graph(lat, lng, target_m)
    except Exception as e:
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=False,
            length_ok=bool(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message=f"OSM 그래프 생성 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = float(time.time() - start_time)
        return safe_list(poly), safe_dict(meta)

    # 2) via-node 후보 선택
    via_nodes = _select_via_candidates(G, start_node, target_m)
    meta["via_candidates"] = via_nodes

    if len(via_nodes) == 0:
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="no_via_fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=False,
            length_ok=bool(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="via-node 후보를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = float(time.time() - start_time)
        return safe_list(poly), safe_dict(meta)

    # via-node가 1개뿐이면 기존 single-loop 방식으로라도 사용
    if len(via_nodes) == 1:
        only_v = via_nodes[0]
        try:
            out_len, out_nodes = _astar_path(G, start_node, only_v)
            poison_edges = set()
            for u, v in zip(out_nodes[:-1], out_nodes[1:]):
                if u == v:
                    continue
                e = (min(u, v), max(u, v))
                poison_edges.add(e)
            back_len, back_nodes = _astar_path(G, only_v, start_node, poison_edges=poison_edges)
            node_path = out_nodes + back_nodes[1:]
            loop_len = out_len + back_len

            # 거리 보정 micro-loop 시도
            node_path, loop_len = _extend_with_micro_loops(G, node_path, loop_len, target_m)

            poly: Polyline = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_path]
            err = abs(loop_len - target_m)

            r = _compute_roundness(poly)
            overlap = _compute_overlap_ratio(node_path)
            curve_pen = _compute_curve_penalty(poly)

            meta.update(
                status="approx",
                len=float(loop_len),
                err=float(err),
                roundness=float(r),
                overlap=float(overlap),
                curve_penalty=float(curve_pen),
                score=None,
                success=bool(err <= target_m * LENGTH_TOL_FRAC),
                length_ok=bool(err <= target_m * LENGTH_TOL_FRAC),
                used_fallback=False,
                message="via-node가 1개뿐이라 single-loop + 거리보정으로 생성했습니다.",
            )
            meta["time_s"] = float(time.time() - start_time)
            return safe_list(poly), safe_dict(meta)
        except Exception as e:
            poly, length_m, err = _fallback_square_loop(lat, lng, km)
            meta.update(
                status="single_via_failed",
                len=float(length_m),
                err=float(err),
                roundness=float(_compute_roundness(poly)),
                overlap=0.0,
                curve_penalty=float(_compute_curve_penalty(poly)),
                score=None,
                success=False,
                length_ok=bool(err <= target_m * LENGTH_TOL_FRAC),
                used_fallback=True,
                message=f"single-loop 생성 실패로 사각형 루프를 사용했습니다: {e}",
            )
            meta["time_s"] = float(time.time() - start_time)
            return safe_list(poly), safe_dict(meta)

    # 3) start→via 경로를 모두 캐싱 (속도 최적화)
    start_to_via_paths: Dict[int, Tuple[float, List[int]]] = {}
    for v in via_nodes:
        try:
            length, nodes = _astar_path(G, start_node, v)
            start_to_via_paths[v] = (length, nodes)
        except Exception:
            continue

    if not start_to_via_paths:
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="no_start_via_fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=False,
            length_ok=bool(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="start→via 경로를 하나도 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = float(time.time() - start_time)
        return safe_list(poly), safe_dict(meta)

    usable_vias = list(start_to_via_paths.keys())

    # 4) via 쌍 조합 생성 (A,B), 개수 제한
    all_pairs = list(combinations(usable_vias, 2))
    random.shuffle(all_pairs)
    if len(all_pairs) > MAX_VIA_PAIRS:
        all_pairs = all_pairs[:MAX_VIA_PAIRS]
    meta["via_pairs"] = all_pairs

    best_score = -float("inf")
    best_node_path: List[int] = []
    best_len = 0.0
    best_err = float("inf")
    best_roundness = 0.0
    best_overlap = 1.0
    best_curve_penalty = 1.0

    # 5) 각 via 쌍(A,B)에 대해 start→A→B→start 루프 생성
    for a, b in all_pairs:
        meta["routes_checked"] += 1
        try:
            # start → A
            len1, path1 = start_to_via_paths[a]
            poison_edges = set()
            for u, v in zip(path1[:-1], path1[1:]):
                if u == v:
                    continue
                e = (min(u, v), max(u, v))
                poison_edges.add(e)

            # A → B
            len2, path2 = _astar_path(G, a, b, poison_edges=poison_edges)
            for u, v in zip(path2[:-1], path2[1:]):
                if u == v:
                    continue
                e = (min(u, v), max(u, v))
                poison_edges.add(e)

            # B → start
            len3, path3 = _astar_path(G, b, start_node, poison_edges=poison_edges)

            node_path = path1 + path2[1:] + path3[1:]
            loop_len = len1 + len2 + len3
            err = abs(loop_len - target_m)

            poly: Polyline = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_path]
            r = _compute_roundness(poly)
            overlap = _compute_overlap_ratio(node_path)
            curve_pen = _compute_curve_penalty(poly)

            meta["routes_validated"] += 1

            length_term = -ERROR_WEIGHT * (err / target_m)
            score = (
                length_term
                + ROUNDNESS_WEIGHT * r
                - OVERLAP_PENALTY * overlap
                - CURVE_PENALTY_WEIGHT * curve_pen
            )

            if score > best_score:
                best_score = score
                best_node_path = node_path
                best_len = loop_len
                best_err = err
                best_roundness = r
                best_overlap = overlap
                best_curve_penalty = curve_pen

        except Exception:
            continue

    if not best_node_path:
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="no_loop_fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=False,
            length_ok=bool(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="via 쌍들로 유효한 루프를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = float(time.time() - start_time)
        return safe_list(poly), safe_dict(meta)

    # 6) 거리 보정 micro-loop 삽입 (부족한 경우에만)
    adjusted_node_path = list(best_node_path)
    adjusted_len = best_len
    if adjusted_len < target_m:
        adjusted_node_path, adjusted_len = _extend_with_micro_loops(
            G, adjusted_node_path, adjusted_len, target_m
        )

    adjusted_err = abs(adjusted_len - target_m)
    adjusted_poly: Polyline = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in adjusted_node_path]
    adjusted_roundness = _compute_roundness(adjusted_poly)
    adjusted_overlap = _compute_overlap_ratio(adjusted_node_path)
    adjusted_curve_penalty = _compute_curve_penalty(adjusted_poly)

    length_term = -ERROR_WEIGHT * (adjusted_err / target_m)
    adjusted_score = (
        length_term
        + ROUNDNESS_WEIGHT * adjusted_roundness
        - OVERLAP_PENALTY * adjusted_overlap
        - CURVE_PENALTY_WEIGHT * adjusted_curve_penalty
    )

    success = bool(adjusted_err <= target_m * LENGTH_TOL_FRAC)
    used_fallback = False

    meta.update(
        status="ok" if success else "approx",
        len=float(adjusted_len),
        err=float(adjusted_err),
        roundness=float(adjusted_roundness),
        overlap=float(adjusted_overlap),
        curve_penalty=float(adjusted_curve_penalty),
        score=float(adjusted_score),
        success=success,
        length_ok=success,
        used_fallback=bool(used_fallback),
        message=(
            "요청 거리와 모양을 모두 만족하는 double-loop 러닝 코스를 생성했습니다."
            if success
            else f"요청 오차(±{int(target_m * LENGTH_TOL_FRAC)}m)를 일부 초과했지만, 거리 보정을 포함한 가장 근접한 루프를 반환합니다."
        ),
    )
    meta["time_s"] = float(time.time() - start_time)

    return safe_list(adjusted_poly), safe_dict(meta)