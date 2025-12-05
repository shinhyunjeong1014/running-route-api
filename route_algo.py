from __future__ import annotations

import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx

try:
    import osmnx as ox
except Exception:
    ox = None

LatLng = Tuple[float, float]
Polyline = List[LatLng]

# -----------------------------
# 기본 상수
# -----------------------------
MIN_LOOP_M = 200.0

# 그래프 반경 (속도 튜닝)
GRAPH_RADIUS_FACTOR = 0.40      # 요청거리 * 0.4
MIN_OSMNX_RADIUS_M = 600.0
MAX_OSMNX_RADIUS_M = 2000.0

# via-node 관련
SECTOR_COUNT = 8                # 각도 섹터 개수 (PCD-lite)
MAX_VIA_TOTAL = 3               # via-node 최대 개수
MAX_VIA_PAIRS = 4               # 평가할 via 쌍 최대 개수

# 거리 오차 허용
LENGTH_TOLERANCE_M = 40.0       # 최종 오차 목표 ±40m

# micro-loop 보정 (짧을 때만)
MAX_MICRO_LOOPS = 2
MICRO_MIN_EDGE = 20.0           # edge 길이 20~40m → loop 40~80m
MICRO_MAX_EDGE = 40.0
MAX_MICRO_NODES = 10            # micro-loop 시도할 노드 수


# ==========================
# 유틸
# ==========================

def haversine_m(a: LatLng, b: LatLng) -> float:
    R = 6371000.0
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    s = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    d = 2 * R * math.asin(math.sqrt(s))
    return d


def _edge_length(G: nx.Graph, u: int, v: int) -> float:
    """
    OSMnx MultiDiGraph에서도 잘 동작하도록 edge length 추출.
    """
    if not G.has_edge(u, v):
        return 0.0
    data = G.get_edge_data(u, v)
    if not isinstance(data, dict):
        return 0.0
    # MultiGraph: {key: {attr}}
    if 0 in data:
        return float(data[0].get("length", 0.0))
    # 키가 0이 아닐 수도 있으니 최소 length 사용
    best = float("inf")
    for d in data.values():
        length = float(d.get("length", 0.0))
        if length < best:
            best = length
    return best if best != float("inf") else 0.0


def path_length_m(G: nx.Graph, path: List[int]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += _edge_length(G, u, v)
    return total


def path_to_polyline(G: nx.Graph, path: List[int]) -> Polyline:
    poly: Polyline = []
    for nid in path:
        x = G.nodes[nid].get("x")
        y = G.nodes[nid].get("y")
        poly.append((float(y), float(x)))
    return poly


# ==========================
# fallback: 단순 사각형 루프
# ==========================

def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
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
    length_m = 4.0 * side
    meta = {
        "status": "fallback_square",
        "len": float(length_m),
        "err": float(abs(length_m - target_m)),
        "roundness": 0.0,
        "success": False,
        "length_ok": False,
        "used_fallback": True,
        "routes_checked": 0,
        "routes_validated": 0,
        "via_candidates": [],
        "via_pairs": [],
        "message": "OSM 그래프를 사용할 수 없어 기하학적 사각형 루프를 반환했습니다.",
    }
    return poly, meta


# ==========================
# 그래프 생성 (속도 최적화)
# ==========================

def _build_osm_graph_fast(lat: float, lng: float, target_m: float) -> Tuple[nx.MultiDiGraph, int]:
    if ox is None:
        raise RuntimeError("osmnx 가 설치되어 있지 않습니다.")

    radius = target_m * GRAPH_RADIUS_FACTOR
    radius = max(MIN_OSMNX_RADIUS_M, min(MAX_OSMNX_RADIUS_M, radius))

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius,
        network_type="walk",
        simplify=True,
    )
    start_node = ox.distance.nearest_nodes(G, lng, lat)
    return G, start_node


# ==========================
# via-node 후보 선택 (PCD-lite)
# ==========================

def _select_via_candidates(
    G: nx.MultiDiGraph,
    start_node: int,
    lat: float,
    lng: float,
    target_m: float,
) -> List[int]:
    """
    PCD-lite:
    - 루프 반지름 ≈ target / (2π)
    - [0.7R, 1.3R] 사이 노드만 후보
    - 각도 섹터(8분할)별 대표 노드를 1개씩 선택
    """
    center = (lat, lng)
    rough_r = max(150.0, target_m / (2.0 * math.pi))
    min_r = rough_r * 0.7
    max_r = rough_r * 1.3

    # start 기준 각도 계산을 위해 start 위경도
    sy = G.nodes[start_node]["y"]
    sx = G.nodes[start_node]["x"]

    sector_best: Dict[int, Tuple[int, float]] = {}
    for nid, data in G.nodes(data=True):
        if nid == start_node:
            continue
        y = data.get("y")
        x = data.get("x")
        if y is None or x is None:
            continue

        d = haversine_m(center, (y, x))
        if d < min_r or d > max_r:
            continue

        dy = y - sy
        dx = x - sx
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        sector = int(SECTOR_COUNT * angle / (2 * math.pi))

        # degree가 높을수록 교차로 가능성 ↑
        deg = G.degree[nid]
        score = d + 30.0 * deg  # 간단 가중치

        prev = sector_best.get(sector)
        if prev is None or score > prev[1]:
            sector_best[sector] = (nid, score)

    candidates = [v for (v, _) in sector_best.values()]
    random.shuffle(candidates)
    if len(candidates) > MAX_VIA_TOTAL:
        candidates = candidates[:MAX_VIA_TOTAL]
    return candidates


# ==========================
# 루프 roundness 계산 (품질 지표)
# ==========================

def _compute_roundness(poly: Polyline) -> float:
    if len(poly) < 3:
        return 0.0
    ys = [p[0] for p in poly]
    xs = [p[1] for p in poly]
    cy = sum(ys) / len(ys)
    cx = sum(xs) / len(xs)

    dists = [haversine_m((cy, cx), (y, x)) for y, x in poly]
    mean = sum(dists) / len(dists)
    if mean <= 0:
        return 0.0
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    std = math.sqrt(var)
    score = max(0.0, 1.0 - std / (mean + 1e-6))
    return min(score, 1.0)


# ==========================
# micro-loop 보정 (짧을 때만)
# ==========================

def _select_microloop_nodes(path: List[int]) -> List[int]:
    n = len(path)
    if n < 4:
        return []
    inner = list(range(1, n - 1))
    if len(inner) <= MAX_MICRO_NODES:
        return inner
    step = max(1, len(inner) // MAX_MICRO_NODES)
    result = []
    for i in range(0, len(inner), step):
        result.append(inner[i])
        if len(result) >= MAX_MICRO_NODES:
            break
    return result


def _extend_with_micro_loops(
    G: nx.MultiDiGraph,
    path: List[int],
    current_len: float,
    target_m: float,
) -> Tuple[List[int], float]:
    """
    - 경로가 target_m보다 짧고, 오차가 40m보다 클 때만 수행
    - u의 이웃 w로 갔다가 다시 u로 돌아오는 작은 loop(u->w->u)를 최대 2번 삽입
    - edge 길이 20~40m (loop 40~80m)만 사용
    - shortest_path 호출 없이 local 수정만 수행 → 속도 영향 낮음
    """
    new_path = list(path)
    total_len = current_len

    for _ in range(MAX_MICRO_LOOPS):
        missing = target_m - total_len
        if missing <= 0 or abs(missing) <= LENGTH_TOLERANCE_M:
            break

        best_idx = None
        best_neighbor = None
        best_new_len = None
        best_err = float("inf")

        for idx in _select_microloop_nodes(new_path):
            u = new_path[idx]
            # 교차로 느낌 나는 곳 위주
            if G.degree[u] < 3:
                continue

            for w in G.neighbors(u):
                if w == new_path[idx - 1] or w == new_path[idx + 1]:
                    continue
                base_len = _edge_length(G, u, w)
                if base_len < MICRO_MIN_EDGE or base_len > MICRO_MAX_EDGE:
                    continue

                extra = 2.0 * base_len
                cand_len = total_len + extra

                # 너무 과하게 overshoot 하면 무시
                if cand_len - target_m > LENGTH_TOLERANCE_M:
                    continue

                err = abs(cand_len - target_m)
                if err < best_err:
                    best_err = err
                    best_idx = idx
                    best_neighbor = w
                    best_new_len = cand_len

        if best_idx is None:
            break

        insert_pos = best_idx + 1
        u = new_path[best_idx]
        new_path = new_path[:insert_pos] + [best_neighbor, u] + new_path[insert_pos:]
        total_len = best_new_len  # type: ignore

    return new_path, total_len

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def polyline_length_m(polyline):
    """[[lat,lng],...] polyline의 총 길이(m)를 계산"""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lng1), (lat2, lng2) in zip(polyline[:-1], polyline[1:]):
        total += haversine_m(lat1, lng1, lat2, lng2)
    return total
    
# ==========================
# 메인: generate_area_loop
# ==========================

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    Ultra-fast + 품질 균형 버전 (C3)

    - time_s 목표: 4~6초 (3km 기준)
    - 거리 오차: ±40m 목표
    - 보행도로 기반 루프 (start→A→B→start)
    """
    start_time = time.time()
    target_m = max(MIN_LOOP_M, km * 1000.0)

    meta: Dict[str, Any] = {
        "status": "init",
        "len": 0.0,
        "err": None,
        "roundness": None,
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
        G, start_node = _build_osm_graph_fast(lat, lng, target_m)
    except Exception as e:
        poly, meta_fb = _fallback_square_loop(lat, lng, km)
        meta_fb["message"] = f"OSM 그래프 생성 실패: {e}"
        meta_fb["time_s"] = float(time.time() - start_time)
        return poly, meta_fb

    # 2) via-node 후보 선택
    via_nodes = _select_via_candidates(G, start_node, lat, lng, target_m)
    meta["via_candidates"] = [int(v) for v in via_nodes]

    if not via_nodes:
        # via가 하나도 없으면 start 기준 가장 먼 노드로 왕복
        nodes = list(G.nodes)
        center = (lat, lng)
        far_node = max(
            nodes,
            key=lambda n: haversine_m(center, (G.nodes[n]["y"], G.nodes[n]["x"]))
        )
        try:
            p1 = nx.shortest_path(G, start_node, far_node, weight="length")
            p2 = list(reversed(p1))
            full = p1 + p2[1:]
            L = path_length_m(G, full)
            poly = path_to_polyline(G, full)
            err = abs(L - target_m)
            r = _compute_roundness(poly)
            meta.update(
                status="approx",
                len=float(L),
                err=float(err),
                roundness=float(r),
                success=bool(err <= LENGTH_TOLERANCE_M),
                length_ok=bool(err <= LENGTH_TOLERANCE_M),
                used_fallback=True,
                message="via-node 후보가 없어 단순 왕복 루프를 사용했습니다.",
            )
            meta["time_s"] = float(time.time() - start_time)
            return poly, meta
        except Exception as e:
            poly, meta_fb = _fallback_square_loop(lat, lng, km)
            meta_fb["message"] = f"via-node 및 왕복 루프 생성 실패: {e}"
            meta_fb["time_s"] = float(time.time() - start_time)
            return poly, meta_fb

    # via가 1개일 때는 (A,A), 2개 이상이면 조합
    via_pairs: List[Tuple[int, int]] = []
    if len(via_nodes) == 1:
        via_pairs = [(via_nodes[0], via_nodes[0])]
    elif len(via_nodes) == 2:
        via_pairs = [(via_nodes[0], via_nodes[1])]
    else:
        # 3개 이상이면 랜덤 조합 최대 MAX_VIA_PAIRS 개
        all_pairs: List[Tuple[int, int]] = []
        for i in range(len(via_nodes)):
            for j in range(i + 1, len(via_nodes)):
                all_pairs.append((via_nodes[i], via_nodes[j]))
        random.shuffle(all_pairs)
        via_pairs = all_pairs[:MAX_VIA_PAIRS]

    meta["via_pairs"] = [(int(a), int(b)) for (a, b) in via_pairs]

    best_path: List[int] = []
    best_len = 0.0
    best_err = float("inf")
    best_roundness = 0.0
    best_score = -float("inf")

    # 3) 각 via 쌍에 대해 start→A→B→start 루프 생성
    for (a, b) in via_pairs:
        meta["routes_checked"] += 1
        try:
            p1 = nx.shortest_path(G, start_node, a, weight="length")
            p2 = nx.shortest_path(G, a, b, weight="length")
            p3 = nx.shortest_path(G, b, start_node, weight="length")
        except Exception:
            continue

        full = p1 + p2[1:] + p3[1:]
        L = path_length_m(G, full)
        err = abs(L - target_m)

        poly = path_to_polyline(G, full)
        r = _compute_roundness(poly)

        meta["routes_validated"] += 1

        # 간단한 스코어: 거리 오차 & 동그란 정도
        score = - (err / max(1.0, target_m)) * 3.0 + 2.0 * r

        if score > best_score:
            best_score = score
            best_path = full
            best_len = L
            best_err = err
            best_roundness = r

    if not best_path:
        poly, meta_fb = _fallback_square_loop(lat, lng, km)
        meta_fb["message"] = "via 쌍으로 유효한 루프를 찾지 못해 사각형 루프를 반환했습니다."
        meta_fb["time_s"] = float(time.time() - start_time)
        return poly, meta_fb

    # 4) 길이가 짧으면 micro-loop로 미세 보정
    adjusted_path = list(best_path)
    adjusted_len = best_len
    if adjusted_len + LENGTH_TOLERANCE_M < target_m:
        adjusted_path, adjusted_len = _extend_with_micro_loops(
            G, adjusted_path, adjusted_len, target_m
        )

    adjusted_err = abs(adjusted_len - target_m)
    adjusted_poly = path_to_polyline(G, adjusted_path)
    adjusted_roundness = _compute_roundness(adjusted_poly)

    success = bool(adjusted_err <= LENGTH_TOLERANCE_M)

    meta.update(
        status="ok" if success else "approx",
        len=float(adjusted_len),
        err=float(adjusted_err),
        roundness=float(adjusted_roundness),
        success=success,
        length_ok=success,
        used_fallback=False,
        message="요청 거리와 루프 모양을 모두 고려한 경로를 생성했습니다."
                if success
                else f"요청 거리와 약간의 오차(±{int(LENGTH_TOLERANCE_M)}m)를 가진 최적 루프를 반환합니다.",
    )
    meta["time_s"] = float(time.time() - start_time)

    return adjusted_poly, meta