from __future__ import annotations

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

# ==========================================================
# 그래프 캐시 (동일/근접 위치 + 거리별로 OSM 그래프 재사용)
# ==========================================================
_GRAPH_CACHE: Dict[Tuple[int, int, int], Tuple[nx.MultiDiGraph, nx.MultiGraph]] = {}
_GRAPH_CACHE_MAX = 8  # 캐시 항목 상한


def _graph_cache_key(lat: float, lng: float, km: float) -> Tuple[int, int, int]:
    """위도/경도/거리 km을 일정 버킷으로 묶어서 캐시 키로 사용."""
    lat_key = int(round(lat / 0.0025))   # ≈ 250m
    lng_key = int(round(lng / 0.0025))   # ≈ 250m
    km_key = int(round(km / 0.5))        # 0.5km 단위
    return (lat_key, lng_key, km_key)


def _get_graph_and_undirected(lat: float, lng: float, km: float) -> Tuple[nx.MultiDiGraph, nx.MultiGraph]:
    """_build_pedestrian_graph 결과를 캐시에 저장·재사용."""
    key = _graph_cache_key(lat, lng, km)

    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    G = _build_pedestrian_graph(lat, lng, km)

    try:
        undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    _GRAPH_CACHE[key] = (G, undirected)

    # 캐시 초과 시 가장 오래된 것 제거 (단순 FIFO)
    if len(_GRAPH_CACHE) > _GRAPH_CACHE_MAX:
        first_key = next(iter(_GRAPH_CACHE.keys()))
        if first_key != key:
            _GRAPH_CACHE.pop(first_key, None)

    return G, undirected


# ==========================================================
# JSON-safe 변환 유틸
# ==========================================================
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
            out.append(safe_list(v))
        elif isinstance(v, dict):
            out.append(safe_dict(v))
        else:
            out.append(safe_float(v, v))
    return out


def safe_dict(d: Any) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = safe_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = safe_list(v)
        else:
            out[k] = safe_float(v, v)
    return out


# ==========================================================
# 거리 / 길이 함수
# ==========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polyline_length_m(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lon1, lat2, lon2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ==========================================================
# roundness / overlap / 곡률 계산
# ==========================================================
def _to_local_xy(polyline: Polyline) -> List[Tuple[float, float]]:
    if not polyline:
        return []
    lats = [p[0] for p in polyline]
    lngs = [p[1] for p in polyline]
    lat0 = sum(lats) / len(lats)
    lng0 = sum(lngs) / len(lngs)
    R = 6371000.0
    xy = []
    for lat, lng in polyline:
        d_lat = math.radians(lat - lat0)
        d_lng = math.radians(lng - lng0)
        x = R * d_lng * math.cos(math.radians(lat0))
        y = R * d_lat
        xy.append((x, y))
    return xy


def polygon_roundness(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 3:
        return 0.0
    xy = _to_local_xy(polyline)
    if not xy:
        return 0.0
    if xy[0] != xy[-1]:
        xy = xy + [xy[0]]

    area = 0.0
    peri = 0.0
    for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
        area += x1 * y2 - x2 * y1
        peri += math.hypot(x2 - x1, y2 - y1)

    area = abs(area) * 0.5
    if area == 0.0 or peri == 0.0:
        return 0.0

    r = 4 * math.pi * area / (peri ** 2)
    if math.isinf(r) or math.isnan(r):
        return 0.0
    return float(r)


def _edge_overlap_fraction(node_path: List[int]) -> float:
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


def _curve_penalty(node_path: List[int], G: nx.Graph) -> float:
    if len(node_path) < 3:
        return 0.0

    coords: Dict[int, Tuple[float, float]] = {}
    for n in node_path:
        if n not in coords:
            node = G.nodes[n]
            coords[n] = (float(node.get("y")), float(node.get("x")))

    penalty = 0.0
    R = 6371000.0

    def to_xy(lat, lng, lat0, lng0):
        d_lat = math.radians(lat - lat0)
        d_lng = math.radians(lng - lng0)
        return (
            R * d_lng * math.cos(math.radians(lat0)),
            R * d_lat,
        )

    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        latA, lngA = coords[a]
        latB, lngB = coords[b]
        latC, lngC = coords[c]

        x1, y1 = to_xy(latA, lngA, latB, lngB)
        x2, y2 = to_xy(latC, lngC, latB, lngB)

        n1 = math.hypot(x1, y1)
        n2 = math.hypot(x2, y2)
        if n1 == 0 or n2 == 0:
            continue

        dot = (x1 * x2 + y1 * y2) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)

        if theta < math.pi / 3:
            penalty += (math.pi / 3 - theta)

    return penalty


# ==========================================================
# 그래프 관련 함수
# ==========================================================
def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    if len(nodes) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            return 0.0
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total


def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """OSMnx 'walk' 네트워크 타입 보행자 그래프 생성."""
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    radius_m = max(700.0, km * 500.0 + 700.0)

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )
    if not G.nodes:
        raise RuntimeError("OSM 보행자 네트워크를 생성하지 못했습니다.")
    return G


def _nodes_to_polyline(G: nx.MultiDiGraph, nodes: List[int]) -> Polyline:
    poly: Polyline = []
    for n in nodes:
        node = G.nodes[n]
        lat = float(node.get("y"))
        lng = float(node.get("x"))
        poly.append((lat, lng))
    return poly


# ==========================================================
# fallback: 사각형 루프
# ==========================================================
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0

    d_lat = side / 111111.0
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))

    a = (lat + d_lat, lng)
    b = (lat + d_lat, lng + d_lng)
    c = (lat,        lng + d_lng)
    d = (lat,        lng)

    poly: Polyline = [d, a, b, c, d]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================================================
# 메인: 러닝 루프 생성
# ==========================================================
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3
    LENGTH_TOL_FRAC = 0.05     # ±5%까지 "정상 범위"
    HARD_ERR_FRAC = 0.30       # ±30% 넘으면 후보에서 제외
    LENGTH_PENALTY_WEIGHT = 8.0

    meta: Dict[str, Any] = {
        "len": 0.0,
        "err": 0.0,
        "roundness": 0.0,
        "overlap": 0.0,
        "curve_penalty": 0.0,
        "score": -1e18,
        "success": False,
        "length_ok": False,
        "used_fallback": False,
        "routes_checked": 0,
        "routes_validated": 0,
        "km_requested": km,
        "target_m": target_m,
        "time_s": 0.0,
        "message": "",
    }

    # ------------------------------------------------------
    # 1) 그래프 + undirected (캐시 사용)
    # ------------------------------------------------------
    try:
        G, undirected = _get_graph_and_undirected(lat, lng, km)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length,
            err=err,
            roundness=r,
            success=False,
            used_fallback=True,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            message=f"그래프 생성 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # ------------------------------------------------------
    # 2) 시작 노드 스냅
    # ------------------------------------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat) if ox is not None else None
        if start_node is None:
            raise RuntimeError("nearest_nodes 실패")
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length,
            err=err,
            roundness=r,
            success=False,
            used_fallback=True,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            message=f"시작 노드 스냅 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # ------------------------------------------------------
    # 3) start에서 단일-source 최단거리 (rod 후보 탐색용)
    # ------------------------------------------------------
    try:
        dist_map: Dict[int, float] = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.8,
            weight="length",
        )
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length,
            err=err,
            roundness=r,
            success=False,
            used_fallback=True,
            message=f"rod 후보 탐색 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    rod_target = target_m / 2.0
    rod_min = rod_target * 0.6
    rod_max = rod_target * 1.4

    # 3-1) 거리 기반 1차 후보 (이전 버전과 동일한 조건)
    candidate_nodes = [
        n for n, d in dist_map.items()
        if rod_min <= d <= rod_max and n != start_node
    ]

    # 3-2) 후보가 너무 적으면 범위 완화
    if len(candidate_nodes) < 5:
        lo = target_m * 0.25
        hi = target_m * 0.75
        candidate_nodes = [
            n for n, d in dist_map.items()
            if lo <= d <= hi and n != start_node
        ]

    if not candidate_nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length,
            err=err,
            roundness=r,
            success=False,
            used_fallback=True,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            message="rod endpoint 후보를 찾지 못했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 3-3) 후보 수 상한 (너무 많으면 랜덤 샘플링)
    random.shuffle(candidate_nodes)
    max_candidates = min(len(candidate_nodes), 80)  # 이전(120)보다는 줄이되 충분히 유지
    candidate_nodes = candidate_nodes[:max_candidates]

    best_poly: Optional[Polyline] = None
    best_score = -1e18
    best_stats: Dict[str, Any] = {}

    # ------------------------------------------------------
    # 4) 각 endpoint에 대해 forward + "경량 poison" backward
    # ------------------------------------------------------
    for endpoint in candidate_nodes:
        # 4-1. forward path
        try:
            forward_nodes = nx.shortest_path(
                undirected,
                start_node,
                endpoint,
                weight="length",
            )
        except Exception:
            continue

        forward_len = dist_map.get(endpoint, _path_length_on_graph(undirected, forward_nodes))
        if forward_len <= 0.0:
            continue

        # rod 길이 제한
        if forward_len < target_m * 0.25 or forward_len > target_m * 0.8:
            continue

        # 4-2. rod edge 집합 구성 (양방향)
        rod_edges = set()
        for u, v in zip(forward_nodes[:-1], forward_nodes[1:]):
            if u == v:
                continue
            rod_edges.add((u, v))
            rod_edges.add((v, u))

        # 4-3. "경량 poison" weight 함수
        def poison_weight(u: int, v: int, data: Dict[str, Any]) -> float:
            base_len = float(data.get("length", 0.0)) or 0.0001
            # forward에서 쓴 간선은 길이를 키워서 다른 길을 선호하게 만듦
            if (u, v) in rod_edges:
                return base_len * 5.0  # 기존 8.0보다 완화해 너무 과한 우회를 방지
            return base_len

        # 4-4. poisoned weight 기반 backward path
        try:
            back_nodes = nx.shortest_path(
                undirected,
                endpoint,
                start_node,
                weight=poison_weight,
            )
        except Exception:
            continue

        if len(back_nodes) < 2:
            continue

        # 4-5. forward + backward를 붙여 하나의 루프
        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        poly = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(poly)
        if length_m <= 0.0:
            continue

        err = abs(length_m - target_m)

        # 길이 오차가 너무 큰 후보는 버림
        if err > target_m * HARD_ERR_FRAC:
            continue

        r = polygon_roundness(poly)
        ov = _edge_overlap_fraction(full_nodes)
        cp = _curve_penalty(full_nodes, undirected)

        length_pen = err / max(1.0, target_m * LENGTH_TOL_FRAC)

        score = (
            ROUNDNESS_WEIGHT * r
            - OVERLAP_PENALTY * ov
            - CURVE_PENALTY_WEIGHT * cp
            - LENGTH_PENALTY_WEIGHT * length_pen
        )

        length_ok = err <= target_m * LENGTH_TOL_FRAC
        if length_ok:
            meta["routes_validated"] += 1

        if score > best_score:
            best_score = score
            best_poly = poly
            best_stats = {
                "len": length_m,
                "err": err,
                "roundness": r,
                "overlap": ov,
                "curve_penalty": cp,
                "score": score,
                "length_ok": length_ok,
            }

    # ------------------------------------------------------
    # 5) 후보 루프가 없으면 fallback
    # ------------------------------------------------------
    if best_poly is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length,
            err=err,
            roundness=r,
            success=False,
            used_fallback=True,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            message="루프 생성 실패 (fallback 사용)",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # ------------------------------------------------------
    # 6) 시작 좌표 앵커링 + 길이 재계산
    # ------------------------------------------------------
    if best_poly:
        # polyline의 처음/끝이 실제 요청 좌표와 1m 이상 떨어져 있으면 앵커링
        first_lat, first_lng = best_poly[0]
        if haversine(lat, lng, first_lat, first_lng) > 1.0:
            best_poly.insert(0, (lat, lng))

        last_lat, last_lng = best_poly[-1]
        if haversine(lat, lng, last_lat, last_lng) > 1.0:
            best_poly.append((lat, lng))

        length2 = polyline_length_m(best_poly)
        err2 = abs(length2 - target_m)
        best_stats["len"] = length2
        best_stats["err"] = err2
        best_stats["length_ok"] = (err2 <= target_m * LENGTH_TOL_FRAC)

    success = bool(best_stats.get("length_ok"))

    meta.update(best_stats)
    meta.update(
        success=success,
        used_fallback=False,
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else "요청 거리와 약간 차이 있지만 가장 근접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    return safe_list(best_poly), safe_dict(meta)
