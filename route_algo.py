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

# 기존 비율 기반 오차 (지금은 사용하지 않지만 남겨둠)
LENGTH_TOL_FRAC = 0.05      # 목표 거리 허용 오차 비율 (±5%)

# 새 절대 오차 기준 (핵심)
MAX_ABS_ERR_M = 45.0        # 목표 거리 허용 오차 (±45m)

# 옵션 A 튜닝값
SECTOR_COUNT = 12           # (이제는 사용 X, 남겨둠)
MAX_VIA_TOTAL = 4           # via-node 상한 (속도/품질 균형)
MAX_VIA_PAIRS = 6           # 평가할 via 쌍 최대 개수

POISON_FACTOR = 3.0         # 왕복 경로 억제를 위한 가중치 배수

# micro-loop (옵션 B: 40~80m × 2~3회)
MAX_MICRO_LOOPS = 3
MICRO_MIN_EDGE = 20.0       # edge 길이 하한 (20m → loop 40m)
MICRO_MAX_EDGE = 40.0       # edge 길이 상한 (40m → loop 80m)
MAX_MICRO_CANDIDATES = 15   # micro-loop 후보 노드 최대 개수
MAX_PCD_RAW_CANDIDATES = 80 # PCD용 raw 후보 최대 개수 (점수 상위 k개만 사용)


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

def _fallback_square_loop(
    lat: float,
    lng: float,
    km: float,
    target_m: Optional[float] = None,
) -> Tuple[Polyline, float, float]:
    """
    OSM/그래프를 전혀 쓰지 못할 때 사용하는 단순 정사각형 루프.
    - target_m이 주어지면 그 값을 기준으로 길이를 맞추고
    - 없으면 km * 1000.0을 기준으로 맞춘다.
    """
    if target_m is None:
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

    # 좌표와 요청거리가 매번 달라도 적절한 그래프가 만들어지도록
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
# via-node 후보 선택 (PCD-full)
# ==========================

def _select_via_candidates(
    G: nx.Graph,
    start_node: int,
    target_m: float,
    sectors: int = SECTOR_COUNT,   # (unused, 유지만)
    max_total: int = MAX_VIA_TOTAL,
) -> List[int]:
    """
    PCD-full 스타일 via-node 선택:
    1) start에서 거리, degree 기반으로 raw 후보를 만든 뒤
    2) 점수 상위 k개에서
    3) '품질(score) + 다양성(closest-distance)'를 동시에 극대화하는 greedy selection.

    - 품질(score): 목표 반경에 얼마나 잘 맞는지 + 교차로(degree) 선호
    - 다양성(diversity): 이미 선택된 via들과의 최소거리(maximize)
    """

    sy = G.nodes[start_node]["y"]
    sx = G.nodes[start_node]["x"]

    # 원형 루프 반지름 근사: L ≈ 2πR → R ≈ L / (2π)
    rough_radius = target_m / (2 * math.pi)

    # 루프 반경 범위
    min_r = max(rough_radius * 0.8, 200.0)
    max_r = rough_radius * 1.6  # 넉넉하게

    raw_candidates: List[Dict[str, Any]] = []

    # 1) raw 후보 수집
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

        degree = G.degree[n]

        # (1) 반경 품질: rough_radius 근처일수록 점수 ↑
        #   d == rough_radius → 1 근처, 멀어질수록 감소
        radius_dev = abs(d - rough_radius)
        radius_score = max(0.0, 1.0 - radius_dev / max_r)

        # (2) 교차로 품질: degree가 높을수록 선호 (최대 6 이상은 동일 취급)
        deg_norm = min(float(degree), 6.0) / 6.0  # 0~1

        # 최종 품질 점수 (반경:degree = 0.7:0.3)
        base_score = 0.7 * radius_score + 0.3 * deg_norm

        raw_candidates.append(
            {
                "id": n,
                "lat": lat,
                "lng": lng,
                "d": d,
                "degree": degree,
                "score": base_score,
            }
        )

    if not raw_candidates:
        return []

    # 2) 품질 점수 상위 일부만 사용 (속도/안정성용)
    raw_candidates.sort(key=lambda c: c["score"], reverse=True)
    k = max(max_total, min(MAX_PCD_RAW_CANDIDATES, len(raw_candidates)))
    raw_candidates = raw_candidates[:k]

    # 3) PCD-style greedy selection
    chosen: List[Dict[str, Any]] = []
    chosen_ids: List[int] = []

    # 첫 번째는 품질이 가장 높은 노드 선택
    first = raw_candidates[0]
    chosen.append(first)
    chosen_ids.append(first["id"])

    if len(raw_candidates) == 1 or max_total == 1:
        return [first["id"]]

    # 품질-다양성 trade-off 계수 (λ)
    LAMBDA_QUALITY = 0.5  # 품질 비중
    # 1 - LAMBDA_QUALITY = 0.5 → 다양성 비중

    while len(chosen_ids) < max_total and len(chosen_ids) < len(raw_candidates):
        best_cand = None
        best_value = -float("inf")

        for cand in raw_candidates:
            if cand["id"] in chosen_ids:
                continue

            # 다양성: 이미 선택된 via들과의 최소 거리
            min_dist = float("inf")
            for c in chosen:
                d = haversine_m(cand["lat"], cand["lng"], c["lat"], c["lng"])
                if d < min_dist:
                    min_dist = d

            # 거리를 max_r로 정규화 (0~1 클램핑)
            diversity = min(1.0, max(0.0, min_dist / max_r))

            quality = cand["score"]  # 이미 0~1 근사

            value = LAMBDA_QUALITY * quality + (1.0 - LAMBDA_QUALITY) * diversity

            if value > best_value:
                best_value = value
                best_cand = cand

        if best_cand is None:
            break

        chosen.append(best_cand)
        chosen_ids.append(best_cand["id"])

    return chosen_ids


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
# micro-loop 후보 노드 선택
# ==========================

def _select_microloop_indices(node_path: List[int]) -> List[int]:
    """
    전체 경로 중에서 micro-loop를 시도해 볼 대표 노드 인덱스 선택.
    - 최대 MAX_MICRO_CANDIDATES 개
    - 시작/중간/끝 쪽으로 고르게 분산
    """
    n = len(node_path)
    if n < 4:
        return []

    indices = []

    # 내부 노드 구간만 사용 (0, n-1 제외)
    inner = list(range(1, n - 1))
    if len(inner) <= MAX_MICRO_CANDIDATES:
        return inner

    step = max(1, len(inner) // MAX_MICRO_CANDIDATES)
    for i in range(0, len(inner), step):
        indices.append(inner[i])
        if len(indices) >= MAX_MICRO_CANDIDATES:
            break

    return indices


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
    경로 상의 대표 교차로들에서 옆 골목으로 들어갔다 나오는
    micro-loop(u->w->u)를 최대 max_loops번까지 삽입해서 거리 보정.
    - edge 길이 20~40m인 경우만 사용 (loop 40~80m)
    - 실제 도로 그래프 위에서만 움직임.
    - overshoot: target_m + MAX_ABS_ERR_M 를 넘는 보정은 금지.
    """
    if not node_path or len(node_path) < 2:
        return node_path, current_len

    new_node_path = list(node_path)
    total_len = current_len

    candidate_indices = _select_microloop_indices(new_node_path)

    for _ in range(max_loops):
        # 이미 절대 오차 45m 이내면 더 이상 보정하지 않음
        missing = target_m - total_len
        if abs(missing) <= MAX_ABS_ERR_M:
            break
        # 이미 target_m 이상인데, 부족한 정도가 크지 않으면(<=45m)도 위에서 break됨.
        # missing <= 0 이면 더 이상 길이를 늘릴 필요가 없으니 종료
        if missing <= 0:
            break

        best_idx = None
        best_neighbor = None
        best_new_len = None
        best_err = float("inf")

        for i in candidate_indices:
            u = new_node_path[i]
            # 교차로(연결이 여러 개) 위주로 시도
            if G.degree[u] < 3:
                continue

            for w in G.neighbors(u):
                # 기존 경로의 직전/직후 노드는 제외 (이미 사용 중인 방향)
                if w == new_node_path[i - 1] or w == new_node_path[i + 1]:
                    continue

                base_len = G[u][w]["length"]
                # edge 20~40m만 사용 (loop 40~80m)
                if base_len < MICRO_MIN_EDGE or base_len > MICRO_MAX_EDGE:
                    continue

                extra = 2.0 * base_len  # u->w->u
                candidate_len = total_len + extra

                # overshoot: target_m + MAX_ABS_ERR_M 를 넘으면 금지
                if candidate_len - target_m > MAX_ABS_ERR_M:
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
# 메인: 러닝 루프 생성 (double via + 거리보정, PCD-full)
# ==========================

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """double-via 기반 러닝 루프 + micro-loop 거리보정 (PCD-full via 선택).

    - via-node 2개(A,B)를 조합해 start→A→B→start 루프를 만든다.
    - PCD-style via 선택으로 '품질 + 다양성'을 극대화한다.
    - 루프가 목표 거리보다 짧으면 실제 도로 위에서 micro-loop를 삽입해 거리 보정.
    - 거리 허용 오차는 '절대 45m 이내' 이며,
      target_m = 요청거리 + 45m 이므로
      실제 길이는 [요청, 요청+90]m 범위 안을 노리게 된다.
    """
    start_time = time.time()

    # 핵심: target_m 을 요청거리 + 45m 로 설정
    # 실제 성공 범위는 [target_m - 45, target_m + 45] = [km*1000, km*1000 + 90]
    target_m = max(MIN_LOOP_M, km * 1000.0 + MAX_ABS_ERR_M)

    # scoring 가중치 (shape/겹침/커브/길이 오차)
    ROUNDNESS_WEIGHT = 3.0
    OVERLAP_PENALTY = 3.0
    CURVE_PENALTY_WEIGHT = 0.6
    ERROR_WEIGHT = 2.5

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
        poly, length_m, err = _fallback_square_loop(lat, lng, km, target_m=target_m)
        meta.update(
            status="fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=bool(err <= MAX_ABS_ERR_M),
            length_ok=bool(err <= MAX_ABS_ERR_M),
            used_fallback=True,
            message=f"OSM 그래프 생성 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = float(time.time() - start_time)
        return safe_list(poly), safe_dict(meta)

    # 2) via-node 후보 선택 (PCD-full)
    via_nodes = _select_via_candidates(G, start_node, target_m)
    meta["via_candidates"] = via_nodes

    if len(via_nodes) == 0:
        poly, length_m, err = _fallback_square_loop(lat, lng, km, target_m=target_m)
        meta.update(
            status="no_via_fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=bool(err <= MAX_ABS_ERR_M),
            length_ok=bool(err <= MAX_ABS_ERR_M),
            used_fallback=True,
            message="via-node 후보를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = float(time.time() - start_time)
        return safe_list(poly), safe_dict(meta)

    # via-node가 1개뿐이면 single-loop + 거리보정
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

            success = err <= MAX_ABS_ERR_M

            meta.update(
                status="ok" if success else "approx",
                len=float(loop_len),
                err=float(err),
                roundness=float(r),
                overlap=float(overlap),
                curve_penalty=float(curve_pen),
                score=None,
                success=success,
                length_ok=success,
                used_fallback=False,
                message="via-node가 1개뿐이라 single-loop + 거리보정으로 생성했습니다."
                        if success
                        else f"요청 오차(±{int(MAX_ABS_ERR_M)}m)를 일부 초과했지만, "
                             f"single-loop 거리 보정을 포함한 가장 근접한 루프를 반환합니다.",
            )
            meta["time_s"] = float(time.time() - start_time)
            return safe_list(poly), safe_dict(meta)
        except Exception as e:
            poly, length_m, err = _fallback_square_loop(lat, lng, km, target_m=target_m)
            meta.update(
                status="single_via_failed",
                len=float(length_m),
                err=float(err),
                roundness=float(_compute_roundness(poly)),
                overlap=0.0,
                curve_penalty=float(_compute_curve_penalty(poly)),
                score=None,
                success=bool(err <= MAX_ABS_ERR_M),
                length_ok=bool(err <= MAX_ABS_ERR_M),
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
        poly, length_m, err = _fallback_square_loop(lat, lng, km, target_m=target_m)
        meta.update(
            status="no_start_via_fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=bool(err <= MAX_ABS_ERR_M),
            length_ok=bool(err <= MAX_ABS_ERR_M),
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
        poly, length_m, err = _fallback_square_loop(lat, lng, km, target_m=target_m)
        meta.update(
            status="no_loop_fallback",
            len=float(length_m),
            err=float(err),
            roundness=float(_compute_roundness(poly)),
            overlap=0.0,
            curve_penalty=float(_compute_curve_penalty(poly)),
            score=None,
            success=bool(err <= MAX_ABS_ERR_M),
            length_ok=bool(err <= MAX_ABS_ERR_M),
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

    success = adjusted_err <= MAX_ABS_ERR_M
    used_fallback = False

    meta.update(
        status="ok" if success else "approx",
        len=float(adjusted_len),
        err=float(adjusted_err),
        roundness=float(adjusted_roundness),
        overlap=float(adjusted_overlap),
        curve_penalty=float(adjusted_curve_penalty),
        score=float(adjusted_score),
        success=bool(success),
        length_ok=bool(success),
        used_fallback=bool(used_fallback),
        message=(
            "요청 거리와 모양을 모두 만족하는 double-loop 러닝 코스를 생성했습니다."
            if success
            else f"요청 오차(±{int(MAX_ABS_ERR_M)}m)를 일부 초과했지만, "
                 f"거리 보정을 포함한 가장 근접한 루프를 반환합니다."
        ),
    )
    meta["time_s"] = float(time.time() - start_time)

    return safe_list(adjusted_poly), safe_dict(meta)