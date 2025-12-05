from __future__ import annotations

import heapq
import math
import random
import time
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx
import numpy as np

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
MAX_OSMNX_RADIUS_M = 2500.0
MIN_OSMNX_RADIUS_M = 600.0

LENGTH_TOL_FRAC = 0.05

# 절대 오차 기준
MAX_ABS_ERR_M = 45.0

# Via-node 설정
SECTOR_COUNT = 12
MAX_VIA_TOTAL = 4
MAX_VIA_PAIRS = 6

POISON_FACTOR = 3.0

# micro-loop 설정
MAX_MICRO_LOOPS = 3
MICRO_MIN_EDGE = 20.0
MICRO_MAX_EDGE = 40.0
MAX_MICRO_CANDIDATES = 15


# ==========================
# JSON-safe 유틸
# ==========================
def safe_bool(x: Any) -> bool:
    """numpy.bool_ → Python bool"""
    return bool(x)


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isinf(x) or math.isnan(x):
            return default
    return x


def safe_list(lst: Any) -> list:
    if not isinstance(lst, (list, tuple)):
        return []
    out = []
    for v in lst:
        if isinstance(v, (list, tuple)):
            out.append([safe_float(x) for x in v])
        else:
            if isinstance(v, np.bool_):
                v = bool(v)
            out.append(v)
    return out


def safe_dict(d: Any) -> dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = safe_float(v)
        elif isinstance(v, (list, tuple)):
            out[k] = safe_list(v)
        elif isinstance(v, dict):
            out[k] = safe_dict(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ==========================
# 거리 계산
# ==========================
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polyline_length_m(poly: Polyline) -> float:
    if len(poly) < 2:
        return 0.0
    dist = 0.0
    for (la1, lo1), (la2, lo2) in zip(poly[:-1], poly[1:]):
        dist += haversine_m(la1, lo1, la2, lo2)
    return dist


# ==========================
# fallback 사각형 루프
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float, target_m=None):
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
    length = polyline_length_m(poly)
    err = abs(length - target_m)
    return poly, length, err


# ==========================
# OSM 그래프 생성
# ==========================
def _build_osm_graph(lat, lng, target_m):
    if ox is None:
        raise RuntimeError("osmnx 미설치")

    radius = max(MIN_OSMNX_RADIUS_M, min(MAX_OSMNX_RADIUS_M, target_m * 0.7))

    G_raw = ox.graph_from_point((lat, lng), dist=radius, network_type="walk", simplify=True)
    G_ud = ox.utils_graph.get_undirected(G_raw)
    G_ud = ox.distance.add_edge_lengths(G_ud)

    # 단일 그래프 구성
    G = nx.Graph()
    for u, v, data in G_ud.edges(data=True):
        l = data.get("length", 1.0)
        if G.has_edge(u, v):
            if l < G[u][v]["length"]:
                G[u][v]["length"] = l
        else:
            G.add_edge(u, v, length=l)

    for n, d in G_ud.nodes(data=True):
        if n not in G:
            G.add_node(n)
        G.nodes[n]["y"] = d.get("y")
        G.nodes[n]["x"] = d.get("x")

    start_node = ox.distance.nearest_nodes(G_ud, lng, lat)
    return G, start_node


# ==========================
# via-node 후보
# ==========================
def _select_via_candidates(G, start, target_m, sectors=SECTOR_COUNT, max_total=MAX_VIA_TOTAL):
    sy = G.nodes[start]["y"]
    sx = G.nodes[start]["x"]

    rough_r = target_m / (2 * math.pi)
    min_r = max(rough_r * 0.8, 200.0)
    max_r = rough_r * 1.6

    sector_best = {}

    for n, d in G.nodes(data=True):
        if n == start:
            continue
        lat, lng = d["y"], d["x"]
        dist = haversine_m(sy, sx, lat, lng)
        if dist < min_r or dist > max_r:
            continue

        ang = math.atan2(lat - sy, lng - sx)
        if ang < 0:
            ang += 2 * math.pi
        sec = int(sectors * ang / (2 * math.pi))

        deg = G.degree[n]
        score = (dist / max_r) + 0.3 * (deg / max(1, G.degree[start]))

        if sec not in sector_best or score > sector_best[sec][1]:
            sector_best[sec] = (n, score)

    cands = [v for (v, _) in sector_best.values()]
    random.shuffle(cands)

    return cands[:max_total]


# ==========================
# A* 경로 탐색
# ==========================
def _astar_path(G, src, dst, poison_edges=None):
    if src == dst:
        return 0.0, [src]

    poison_edges = poison_edges or set()

    def h(n):
        return haversine_m(G.nodes[n]["y"], G.nodes[n]["x"], G.nodes[dst]["y"], G.nodes[dst]["x"])

    openh = [(h(src), src)]
    g_cost = {src: 0.0}
    g_dist = {src: 0.0}
    came = {}
    visited = set()

    while openh:
        _, cur = heapq.heappop(openh)
        if cur in visited:
            continue
        visited.add(cur)

        if cur == dst:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return g_dist[dst], list(reversed(path))

        for nb in G.neighbors(cur):
            base = G[cur][nb]["length"]
            e = (min(cur, nb), max(cur, nb))
            w = base * (POISON_FACTOR if e in poison_edges else 1.0)

            tcost = g_cost[cur] + w
            if tcost >= g_cost.get(nb, float("inf")):
                continue

            came[nb] = cur
            g_cost[nb] = tcost
            g_dist[nb] = g_dist[cur] + base
            heapq.heappush(openh, (tcost + h(nb), nb))

    raise nx.NetworkXNoPath(f"{src}->{dst} 경로 없음")


# ==========================
# 품질 평가
# ==========================
def _compute_roundness(poly):
    if len(poly) < 3:
        return 0.0
    ys = [p[0] for p in poly]
    xs = [p[1] for p in poly]
    cy = sum(ys)/len(ys)
    cx = sum(xs)/len(xs)

    ds = [haversine_m(cy, cx, y, x) for y, x in poly]
    mean = sum(ds)/len(ds)
    if mean <= 0:
        return 0.0
    var = sum((d-mean)**2 for d in ds)/len(ds)
    std = math.sqrt(var)
    return max(0.0, min(1.0, 1 - std/(mean+1e-6)))


def _compute_overlap_ratio(path):
    edges = {}
    for u, v in zip(path[:-1], path[1:]):
        if u == v:
            continue
        e = (u, v) if u <= v else (v, u)
        edges[e] = edges.get(e, 0) + 1
    if not edges:
        return 0.0
    rep = sum(1 for c in edges.values() if c > 1)
    return rep / len(edges)


def _compute_curve_penalty(poly):
    if len(poly) < 3:
        return 0.0

    def bearing(a, b):
        la1, lo1 = map(math.radians, a)
        la2, lo2 = map(math.radians, b)
        dlon = lo2 - lo1
        y = math.sin(dlon)*math.cos(la2)
        x = math.cos(la1)*math.sin(la2) - math.sin(la1)*math.cos(la2)*math.cos(dlon)
        ang = math.degrees(math.atan2(y, x))
        return (ang + 360) % 360

    turns = 0
    sharp = 0
    for a, b, c in zip(poly[:-2], poly[1:-1], poly[2:]):
        d1 = bearing(a, b)
        d2 = bearing(b, c)
        diff = abs(d2 - d1)
        if diff > 180:
            diff = 360 - diff
        if diff < 10:
            continue
        turns += 1
        if diff >= 60:
            sharp += 1

    return sharp/turns if turns else 0.0


# ==========================
# micro-loop 후보 선택
# ==========================
def _select_microloop_indices(path):
    n = len(path)
    if n < 4:
        return []
    inner = list(range(1, n-1))
    if len(inner) <= MAX_MICRO_CANDIDATES:
        return inner
    step = max(1, len(inner)//MAX_MICRO_CANDIDATES)
    return inner[::step]


# ==========================
# micro-loop 삽입
# ==========================
def _extend_with_micro_loops(G, path, length, target_m, max_loops=MAX_MICRO_LOOPS):
    new_path = list(path)
    total = length
    idxs = _select_microloop_indices(new_path)

    for _ in range(max_loops):
        missing = target_m - total
        if abs(missing) <= MAX_ABS_ERR_M:
            break
        if missing <= 0:
            break

        best = None
        best_err = float("inf")

        for i in idxs:
            u = new_path[i]
            if G.degree[u] < 3:
                continue

            for w in G.neighbors(u):
                if w == new_path[i-1] or w == new_path[i+1]:
                    continue

                b = G[u][w]["length"]
                if b < MICRO_MIN_EDGE or b > MICRO_MAX_EDGE:
                    continue

                new_len = total + 2*b
                if new_len - target_m > MAX_ABS_ERR_M:
                    continue

                err = abs(new_len - target_m)
                if err < best_err:
                    best_err = err
                    best = (i, w, new_len)

        if best is None:
            break

        i, w, new_len = best
        u = new_path[i]
        insert = i+1
        new_path = new_path[:insert] + [w, u] + new_path[insert:]
        total = new_len

    return new_path, total


# ==========================
# 메인 함수
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    start_time = time.time()

    # 핵심: target_m = 요청거리 + 45m
    target_m = max(MIN_LOOP_M, km * 1000.0 + MAX_ABS_ERR_M)

    ROUNDNESS_WEIGHT = 3.0
    OVERLAP_PENALTY = 3.0
    CURVE_PENALTY_WEIGHT = 0.6
    ERROR_WEIGHT = 2.5

    meta = {
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
        poly, l, err = _fallback_square_loop(lat, lng, km, target_m)
        success = err <= MAX_ABS_ERR_M
        meta.update(
            status="fallback",
            len=float(l),
            err=float(err),
            roundness=_compute_roundness(poly),
            overlap=0.0,
            curve_penalty=_compute_curve_penalty(poly),
            score=None,
            success=safe_bool(success),
            length_ok=safe_bool(success),
            used_fallback=True,
            message=f"OSM 그래프 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 2) via-node 선택
    vias = _select_via_candidates(G, start_node, target_m)
    meta["via_candidates"] = list(vias)

    if not vias:
        poly, l, err = _fallback_square_loop(lat, lng, km, target_m)
        success = err <= MAX_ABS_ERR_M
        meta.update(
            status="no_via",
            len=float(l),
            err=float(err),
            roundness=_compute_roundness(poly),
            overlap=0.0,
            curve_penalty=_compute_curve_penalty(poly),
            success=safe_bool(success),
            length_ok=safe_bool(success),
            used_fallback=True,
            message="via-node 없음",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 3) start→via 계산
    start_via = {}
    for v in vias:
        try:
            l, p = _astar_path(G, start_node, v)
            start_via[v] = (l, p)
        except Exception:
            pass

    if not start_via:
        poly, l, err = _fallback_square_loop(lat, lng, km, target_m)
        success = err <= MAX_ABS_ERR_M
        meta.update(
            status="no_start_via",
            len=float(l),
            err=float(err),
            roundness=_compute_roundness(poly),
            overlap=0.0,
            curve_penalty=_compute_curve_penalty(poly),
            success=safe_bool(success),
            length_ok=safe_bool(success),
            used_fallback=True,
            message="start→via 불가",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    usable = list(start_via.keys())
    pairs = list(combinations(usable, 2))
    random.shuffle(pairs)
    pairs = pairs[:MAX_VIA_PAIRS]
    meta["via_pairs"] = pairs

    best_score = -1e15
    best_path = None
    best_len = None
    best_err = None
    best_r = 0
    best_over = 0
    best_curve = 0

    # 4) via 쌍 평가
    for a, b in pairs:
        meta["routes_checked"] += 1
        try:
            len1, p1 = start_via[a]
            poison = {(min(u, v), max(u, v)) for u, v in zip(p1[:-1], p1[1:])}

            len2, p2 = _astar_path(G, a, b, poison_edges=poison)
            for u, v in zip(p2[:-1], p2[1:]):
                poison.add((min(u, v), max(u, v)))

            len3, p3 = _astar_path(G, b, start_node, poison_edges=poison)

            full = p1 + p2[1:] + p3[1:]
            loop_len = len1 + len2 + len3
            err = abs(loop_len - target_m)

            poly = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in full]
            r = _compute_roundness(poly)
            over = _compute_overlap_ratio(full)
            curve = _compute_curve_penalty(poly)

            meta["routes_validated"] += 1

            score = (
                -ERROR_WEIGHT * (err / target_m)
                + 3.0 * r
                - 3.0 * over
                - 0.6 * curve
            )

            if score > best_score:
                best_score = score
                best_path = full
                best_len = loop_len
                best_err = err
                best_r = r
                best_over = over
                best_curve = curve

        except Exception:
            continue

    if best_path is None:
        poly, l, err = _fallback_square_loop(lat, lng, km, target_m)
        success = err <= MAX_ABS_ERR_M
        meta.update(
            status="no_loop",
            len=float(l),
            err=float(err),
            roundness=_compute_roundness(poly),
            overlap=0.0,
            curve_penalty=_compute_curve_penalty(poly),
            success=safe_bool(success),
            length_ok=safe_bool(success),
            used_fallback=True,
            message="double-via 실패",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 5) micro-loop 보정
    adj_path, adj_len = _extend_with_micro_loops(G, best_path, best_len, target_m)

    adj_err = abs(adj_len - target_m)
    adj_poly = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in adj_path]

    adj_r = _compute_roundness(adj_poly)
    adj_over = _compute_overlap_ratio(adj_path)
    adj_curve = _compute_curve_penalty(adj_poly)

    adj_score = (
        -ERROR_WEIGHT * (adj_err / target_m)
        + 3.0 * adj_r
        - 3.0 * adj_over
        - 0.6 * adj_curve
    )

    success = adj_err <= MAX_ABS_ERR_M

    meta.update(
        status="ok" if success else "approx",
        len=float(adj_len),
        err=float(adj_err),
        roundness=float(adj_r),
        overlap=float(adj_over),
        curve_penalty=float(adj_curve),
        score=float(adj_score),
        success=safe_bool(success),
        length_ok=safe_bool(success),
        used_fallback=False,
        message=(
            "요청 정확도에 맞춘 double-loop 경로 생성 성공!"
            if success else
            f"±{MAX_ABS_ERR_M}m 허용 오차 내에 못 맞춰 가장 근접 경로 반환"
        ),
    )

    meta["time_s"] = time.time() - start_time

    return safe_list(adj_poly), safe_dict(meta)