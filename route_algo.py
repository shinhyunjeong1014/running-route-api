from __future__ import annotations

import heapq
import math
import random
import time
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
MAX_VIA_PER_SECTOR = 1      # 섹터당 후보 수 (현재 1로 고정)
MAX_VIA_TOTAL = 14          # 전체 via-node 상한

POISON_FACTOR = 3.0         # 왕복 경로 억제를 위한 가중치 배수


# ==========================
# 유틸 함수
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
    """
    OSM/그래프를 전혀 쓰지 못할 때 사용하는 매우 단순한 정사각형 루프.
    - 실제 도로망과 맞지 않을 수 있지만, API가 완전히 죽었을 때의 최후 수단.
    """
    target_m = max(MIN_LOOP_M, km * 1000.0)
    side = target_m / 4.0

    # 위도 1m ≈ 1/111111 deg
    d_lat = side / 111111.0
    # 경도 1m ≈ 1/(111111 * cos(lat))
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
    """
    OSMnx를 사용해 보행자용 그래프를 생성하고,
    무방향 단순 그래프로 바꾼 뒤 시작 노드를 반환한다.
    """
    if ox is None:
        raise RuntimeError("osmnx 가 설치되어 있지 않습니다.")

    # 목표 거리를 기준으로 탐색 반경 설정
    radius = max(MIN_OSMNX_RADIUS_M, min(MAX_OSMNX_RADIUS_M, target_m * 0.7))

    # 보행자용 그래프 로딩
    G_raw = ox.graph_from_point(
        (lat, lng),
        dist=radius,
        network_type="walk",
        simplify=True,
    )

    # undirected + edge length 보장
    G_undirected = ox.utils_graph.get_undirected(G_raw)
    G_undirected = ox.distance.add_edge_lengths(G_undirected)

    # MultiGraph -> 단순 Graph 로 변환 (가장 짧은 edge 사용)
    G = nx.Graph()
    for u, v, data in G_undirected.edges(data=True):
        length = data.get("length", 1.0)
        if G.has_edge(u, v):
            if length < G[u][v]["length"]:
                G[u][v]["length"] = length
        else:
            G.add_edge(u, v, length=length)

    # node 좌표 복사
    for n, data in G_undirected.nodes(data=True):
        if n not in G:
            G.add_node(n)
        G.nodes[n]["y"] = data.get("y")
        G.nodes[n]["x"] = data.get("x")

    # 시작 노드: 가장 가까운 노드
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
    """
    시작 노드 기준으로 일정 거리 범위 안의 노드들 중,
    방향(각도)별로 대표 노드를 뽑아 via-node 후보를 구성.
    - PCD 논문의 cluster 개념을 각도 섹터로 단순화한 버전.
    """
    sy = G.nodes[start_node]["y"]
    sx = G.nodes[start_node]["x"]

    # 루프의 "반지름" 근사: 원형 루프라고 가정
    # 둘레 L ≈ 2πR → R ≈ L / (2π)
    rough_radius = target_m / (2 * math.pi)
    min_r = max(rough_radius * 0.7, 150.0)
    max_r = rough_radius * 1.4

    sector_best: Dict[int, Tuple[int, float]] = {}  # sector -> (node, score)

    # 전체 노드 중 거리 조건에 맞는 노드만 후보로 사용
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

        # 각도 계산 (라디안, -π ~ π)
        dy = lat - sy
        dx = lng - sx
        angle = math.atan2(dy, dx)
        # 0 ~ 2π 로 변환
        if angle < 0:
            angle += 2 * math.pi

        sector = int(sectors * angle / (2 * math.pi))
        degree = G.degree[n]

        # 점수: 거리가 far할수록, degree가 클수록 우선
        score = (d / max_r) + 0.3 * (degree / max(1, G.degree[start_node]))

        prev = sector_best.get(sector)
        if prev is None or score > prev[1]:
            sector_best[sector] = (n, score)

    # 섹터별로 고른 노드들을 모아서 상위 max_total개만 사용
    candidates = [v for (v, _) in sector_best.values()]
    random.shuffle(candidates)  # 섹터 순서 편향 방지

    if len(candidates) > max_total:
        candidates = candidates[:max_total]

    return candidates


# ==========================
# A* 기반 최단 경로 (왕복 경로 억제용 poison 지원)
# ==========================

def _astar_path(
    G: nx.Graph,
    src: int,
    dst: int,
    poison_edges: Optional[set] = None,
) -> Tuple[float, List[int]]:
    """
    간단한 A* 구현.
    - heuristic: 직선 거리(haversine)
    - poison_edges: 포함된 edge는 length * POISON_FACTOR 로 가중
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

    g_score: Dict[int, float] = {src: 0.0}
    came_from: Dict[int, int] = {}

    visited = set()

    while open_heap:
        f, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == dst:
            # 경로 복원
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return g_score[dst], path

        for neighbor in G.neighbors(current):
            base_len = G[current][neighbor]["length"]
            e = (min(current, neighbor), max(current, neighbor))
            w = base_len * (POISON_FACTOR if e in poison_edges else 1.0)

            tentative_g = g_score[current] + w
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score = tentative_g + heuristic(neighbor)
            heapq.heappush(open_heap, (f_score, neighbor))

    raise nx.NetworkXNoPath(f"A* 경로를 찾지 못했습니다: {src} -> {dst}")


# ==========================
# 루프 경로 품질 평가
# ==========================

def _compute_roundness(poly: Polyline) -> float:
    """
    루프가 얼마나 '둥근지'를 0~1 사이로 평가.
    - 중심 기준 거리의 표준편차가 작을수록 1에 가깝게.
    """
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
    # 표준편차가 작을수록 값이 커지도록
    score = max(0.0, 1.0 - std / (mean + 1e-6))
    return min(score, 1.0)


def _compute_overlap_ratio(node_path: List[int]) -> float:
    """
    노드 시퀀스에서 같은 간선을 여러 번 쓰는 비율.
    (0에 가까울수록 더 '한 번씩만' 지나는 좋은 루프)
    """
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
    """
    급커브가 많을수록 penalty 증가.
    - 60도 이하의 뾰족한 방향 전환이 많으면 불리.
    """
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
# 메인: 러닝 루프 생성
# ==========================

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    요청 좌표(lat, lng)와 목표 거리(km)를 기반으로
    '요청거리 정확도'와 '루프 모양'을 동시에 고려한 러닝 루프를 생성한다.

    - PCD / goal-directed A* 개념을 적용해
      불필요한 탐색을 줄이고, sector 기반 via-node로 모양을 안정화한다.
    """
    start_time = time.time()
    target_m = max(MIN_LOOP_M, km * 1000.0)

    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.4
    ERROR_WEIGHT = 3.5  # 길이 오차 비중

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
        "message": "",
    }

    # 1) OSM 그래프 구성 시도
    try:
        G, start_node = _build_osm_graph(lat, lng, target_m)
    except Exception as e:
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="fallback",
            len=length_m,
            err=err,
            roundness=_compute_roundness(poly),
            overlap=0.0,
            curve_penalty=_compute_curve_penalty(poly),
            score=None,
            success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message=f"OSM 그래프 생성 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 2) via-node 후보 선택 (sector 기반)
    via_nodes = _select_via_candidates(G, start_node, target_m)
    meta["via_candidates"] = via_nodes

    if not via_nodes:
        # 그래프는 있으나, 적절한 거리 범위의 via 노드를 찾지 못한 경우
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="no_via_fallback",
            len=length_m,
            err=err,
            roundness=_compute_roundness(poly),
            overlap=0.0,
            curve_penalty=_compute_curve_penalty(poly),
            score=None,
            success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="그래프 상에서 적절한 via-node 후보를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 3) 각 via-node에 대해 왕복 루프 후보 생성
    best_score = -float("inf")
    best_poly: Polyline = []
    best_len = 0.0
    best_err = float("inf")
    best_roundness = 0.0
    best_overlap = 1.0
    best_curve_penalty = 1.0

    for via in via_nodes:
        meta["routes_checked"] += 1
        try:
            # 3-1) out path (start -> via)
            out_len, out_nodes = _astar_path(G, start_node, via)

            # 3-2) edge poisoning: out path에 사용된 edge들은 왕복에서 덜 사용되도록 가중치 증가
            poison_edges = set()
            for u, v in zip(out_nodes[:-1], out_nodes[1:]):
                if u == v:
                    continue
                e = (min(u, v), max(u, v))
                poison_edges.add(e)

            # 3-3) back path (via -> start)
            back_len, back_nodes = _astar_path(G, via, start_node, poison_edges=poison_edges)

            # 노드 경로 및 polyline 생성
            node_path = out_nodes + back_nodes[1:]
            poly: Polyline = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_path]
            loop_len = out_len + back_len
            err = abs(loop_len - target_m)

            # 품질 평가
            r = _compute_roundness(poly)
            overlap = _compute_overlap_ratio(node_path)
            curve_pen = _compute_curve_penalty(poly)

            meta["routes_validated"] += 1

            # score: 길이 오차가 작고, 둥글고, 중복이 적고, 급커브가 적을수록 좋음
            length_term = -ERROR_WEIGHT * (err / target_m)
            score = (
                length_term
                + ROUNDNESS_WEIGHT * r
                - OVERLAP_PENALTY * overlap
                - CURVE_PENALTY_WEIGHT * curve_pen
            )

            if score > best_score:
                best_score = score
                best_poly = poly
                best_len = loop_len
                best_err = err
                best_roundness = r
                best_overlap = overlap
                best_curve_penalty = curve_pen

        except Exception:
            # 해당 via-node 경로 생성 실패는 무시하고 다음 후보로 진행
            continue

    if not best_poly:
        # 그래프는 있으나 어떤 via-node로도 루프를 만들지 못했을 때
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="no_loop_fallback",
            len=length_m,
            err=err,
            roundness=_compute_roundness(poly),
            overlap=0.0,
            curve_penalty=_compute_curve_penalty(poly),
            score=None,
            success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="그래프 상에서 유효한 루프 경로를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 4) 최종 메타 정보 구성
    success = bool(best_err <= target_m * LENGTH_TOL_FRAC)
    used_fallback = False

    meta.update(
        status="ok" if success else "approx",
        len=float(best_len),
        err=float(best_err),
        roundness=float(best_roundness),
        overlap=float(best_overlap),
        curve_penalty=float(best_curve_penalty),
        score=float(best_score),
        success=success,
        length_ok=success,
        used_fallback=bool(used_fallback),
        message=(
            "요청 거리와 모양을 모두 만족하는 러닝 루프를 생성했습니다."
            if success
            else f"요청 오차(±{int(target_m * LENGTH_TOL_FRAC)}m)를 일부 초과했지만, 가장 근접한 러닝 루프를 반환합니다."
        ),
    )

    meta["time_s"] = float(time.time() - start_time)

    return safe_list(best_poly), safe_dict(meta)