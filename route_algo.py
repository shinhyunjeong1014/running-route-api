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


# ==========================
# JSON-safe 변환 유틸
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


# ==========================
# 거리 / 길이 유틸
# ==========================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(la1, lo1, la2, lo2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ==========================
# roundness / overlap / 곡률 페널티
# ==========================
def _to_local_xy(polyline: Polyline) -> List[Tuple[float, float]]:
    if not polyline:
        return []
    lats = [p[0] for p in polyline]
    lngs = [p[1] for p in polyline]
    lat0 = sum(lats) / len(lats)
    lng0 = sum(lngs) / len(lngs)
    R = 6371000.0
    res = []
    for lat, lng in polyline:
        d_lat = math.radians(lat - lat0)
        d_lng = math.radians(lng - lng0)
        x = R * d_lng * math.cos(math.radians(lat0))
        y = R * d_lat
        res.append((x, y))
    return res


def polygon_roundness(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 3:
        return 0.0
    xy = _to_local_xy(polyline)
    if not xy:
        return 0.0
    if xy[0] != xy[-1]:
        xy = xy + [xy[0]]

    area = 0.0
    perimeter = 0.0
    for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
        area += x1 * y2 - x2 * y1
        perimeter += math.hypot(x2 - x1, y2 - y1)
    area = abs(area) * 0.5
    if area == 0.0 or perimeter == 0.0:
        return 0.0
    r = 4 * math.pi * area / (perimeter ** 2)
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
        if n in coords:
            continue
        node = G.nodes[n]
        coords[n] = (float(node.get("y")), float(node.get("x")))

    penalty = 0.0
    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        lat_a, lng_a = coords[a]
        lat_b, lng_b = coords[b]
        lat_c, lng_c = coords[c]

        R = 6371000.0
        def _to_xy(lat, lng, lat0, lng0):
            d_lat = math.radians(lat - lat0)
            d_lng = math.radians(lng - lng0)
            x = R * d_lng * math.cos(math.radians(lat0))
            y = R * d_lat
            return x, y

        x1, y1 = _to_xy(lat_a, lng_a, lat_b, lng_b)
        x2, y2 = _to_xy(lat_c, lng_c, lat_b, lng_b)

        v1x, v1y = x1, y1
        v2x, v2y = x2, y2
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
    if not nodes or len(nodes) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            return 0.0
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total


def _apply_route_poison(G: nx.MultiGraph, path_nodes: List[int], factor: float = 8.0) -> nx.MultiGraph:
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not G2.has_edge(u, v):
            continue
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


def _nodes_to_polyline(G: nx.MultiDiGraph, nodes: List[int]) -> Polyline:
    poly: Polyline = []
    for n in nodes:
        node = G.nodes[n]
        lat = float(node.get("y"))
        lng = float(node.get("x"))
        poly.append((lat, lng))
    return poly


# ==========================
# fallback: 기하학적 사각형 루프
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0

    d_lat = (side / 111111.0)
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))

    a = (lat + d_lat, lng)
    b = (lat + d_lat, lng + d_lng)
    c = (lat,        lng + d_lng)
    d = (lat,        lng)

    poly: Polyline = [d, a, b, c, d]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# 메인: 러닝 루프 생성기
# ==========================
def generate_area_loop(G: nx.MultiGraph, lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    [수정됨] G를 외부에서 주입받아 다운로드 시간 제거
    """
    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

    # 스코어링 가중치
    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3

    LENGTH_TOL_FRAC = 0.05
    HARD_ERR_FRAC = 0.30
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

    # 1) 그래프 유효성 체크
    if G is None or len(G.nodes) == 0:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, score=r,
            success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="서버에 로드된 맵 데이터가 없어 사각형 루프를 사용했습니다."
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # G는 이미 undirected라고 가정 (App Lifespan에서 처리)
    undirected = G

    # 2) 시작 노드 스냅
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat) if ox is not None else None
        if start_node is None:
            raise RuntimeError("nearest_nodes 실패")
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, score=r,
            success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message=f"시작 좌표 스냅 실패: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 3) start에서 단일-출발 최단거리
    try:
        dist_from_start: Dict[int, float] = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.8,
            weight="length",
        )
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, score=r,
            success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message=f"탐색 실패: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    rod_target = target_m / 2.0
    rod_min = rod_target * 0.6
    rod_max = rod_target * 1.4

    candidate_nodes = [
        n for n, d in dist_from_start.items()
        if rod_min <= d <= rod_max and n != start_node
    ]

    if len(candidate_nodes) < 5:
        lo = target_m * 0.25
        hi = target_m * 0.75
        candidate_nodes = [
            n for n, d in dist_from_start.items()
            if lo <= d <= hi and n != start_node
        ]

    if not candidate_nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, score=r,
            success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="적절한 반환점 후보 없음"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:120]

    best_score = -1e18
    best_poly: Optional[Polyline] = None
    best_stats: Dict[str, Any] = {}

    # 4) 루프 생성 시도
    for endpoint in candidate_nodes:
        try:
            forward_nodes = nx.shortest_path(undirected, start_node, endpoint, weight="length")
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len <= 0.0 or forward_len < target_m * 0.25 or forward_len > target_m * 0.8:
            continue

        poisoned = _apply_route_poison(undirected, forward_nodes, factor=8.0)

        try:
            back_nodes = nx.shortest_path(poisoned, endpoint, start_node, weight="length")
        except Exception:
            continue

        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        poly = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(poly)
        err = abs(length_m - target_m)

        if err > target_m * HARD_ERR_FRAC:
            continue

        r = polygon_roundness(poly)
        ov = _edge_overlap_fraction(full_nodes)
        cp = _curve_penalty(full_nodes, undirected)
        length_pen = err / (max(1.0, target_m * LENGTH_TOL_FRAC))

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
                "len": length_m, "err": err, "roundness": r,
                "overlap": ov, "curve_penalty": cp, "score": score,
                "length_ok": length_ok,
            }

    # 5) 결과 처리
    if best_poly is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, score=r,
            success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="적절한 경로 생성 실패, 사각형 루프 반환"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 앵커링
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

    meta.update(best_stats)
    meta.update(
        success=bool(best_stats.get("length_ok")),
        used_fallback=False,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message="최적 경로 도출 성공" if meta["length_ok"] else "근사 경로 반환"
    )
    meta["time_s"] = time.time() - start_time

    return safe_list(best_poly), safe_dict(meta)
