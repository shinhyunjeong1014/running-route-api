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
# 🔥 그래프 캐시 추가 (성능 최적화 핵심)
# ==========================================================
_GRAPH_CACHE: Dict[Tuple[int, int, int], Tuple[nx.MultiDiGraph, nx.MultiGraph]] = {}
_GRAPH_CACHE_MAX = 8  # 캐시 크기 제한


def _graph_cache_key(lat: float, lng: float, km: float) -> Tuple[int, int, int]:
    """위도/경도/거리 km을 일정 버킷으로 묶어서 캐시 키로 사용"""

    lat_key = int(round(lat / 0.0025))   # 약 250m 단위
    lng_key = int(round(lng / 0.0025))   # 약 250m 단위
    km_key = int(round(km / 0.5))        # 0.5km 단위

    return (lat_key, lng_key, km_key)


def _get_graph_and_undirected(lat: float, lng: float, km: float):
    """_build_pedestrian_graph 호출 결과를 캐시에 저장하고 재사용"""

    key = _graph_cache_key(lat, lng, km)

    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    # 캐시에 없으면 새로 생성
    G = _build_pedestrian_graph(lat, lng, km)

    try:
        undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    # 저장
    _GRAPH_CACHE[key] = (G, undirected)

    # 캐시 용량 초과 시 FIFO 방식으로 제거
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
        math.sin(d_lat/2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(d_lon/2)**2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polyline_length_m(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = sum(
        haversine(lat1, lon1, lat2, lon2)
        for (lat1, lon1), (lat2, lon2)
        in zip(polyline[:-1], polyline[1:])
    )
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ==========================================================
# roundness / overlap / 곡률 계산
# ==========================================================
def _to_local_xy(polyline):
    if not polyline:
        return []
    lats = [p[0] for p in polyline]
    lngs = [p[1] for p in polyline]
    lat0 = sum(lats)/len(lats)
    lng0 = sum(lngs)/len(lngs)
    R = 6371000.0
    xy = []
    for lat,lng in polyline:
        d_lat = math.radians(lat - lat0)
        d_lng = math.radians(lng - lng0)
        x = R * d_lng * math.cos(math.radians(lat0))
        y = R * d_lat
        xy.append((x,y))
    return xy

def polygon_roundness(polyline):
    if not polyline or len(polyline) < 3:
        return 0.0
    xy = _to_local_xy(polyline)
    if xy[0] != xy[-1]:
        xy = xy + [xy[0]]

    area = 0.0
    peri = 0.0
    for (x1,y1),(x2,y2) in zip(xy[:-1], xy[1:]):
        area += x1*y2 - x2*y1
        peri += math.hypot(x2-x1, y2-y1)

    area = abs(area) * 0.5
    if area == 0 or peri == 0:
        return 0.0

    r = 4 * math.pi * area / (peri**2)
    if math.isinf(r) or math.isnan(r):
        return 0.0
    return r


def _edge_overlap_fraction(node_path):
    if len(node_path) < 2:
        return 0.0

    edges = {}
    for u,v in zip(node_path[:-1], node_path[1:]):
        if u == v:
            continue
        e = (u,v) if u<=v else (v,u)
        edges[e] = edges.get(e, 0) + 1

    if not edges:
        return 0.0

    overlap_edges = sum(1 for c in edges.values() if c > 1)
    return overlap_edges / len(edges)


def _curve_penalty(node_path, G):
    if len(node_path) < 3:
        return 0.0

    coords = {}
    for n in node_path:
        if n not in coords:
            node = G.nodes[n]
            coords[n] = (float(node["y"]), float(node["x"]))

    R = 6371000.0
    penalty = 0.0

    def to_xy(lat,lng,lat0,lng0):
        return (
            R * math.radians(lng - lng0) * math.cos(math.radians(lat0)),
            R * math.radians(lat - lat0)
        )

    for i in range(1,len(node_path)-1):
        a,b,c = node_path[i-1], node_path[i], node_path[i+1]
        latA,lngA = coords[a]
        latB,lngB = coords[b]
        latC,lngC = coords[c]

        x1,y1 = to_xy(latA,lngA,latB,lngB)
        x2,y2 = to_xy(latC,lngC,latB,lngB)

        n1 = math.hypot(x1,y1)
        n2 = math.hypot(x2,y2)
        if n1==0 or n2==0:
            continue

        dot = (x1*x2 + y1*y2)/(n1*n2)
        dot = max(-1,min(1,dot))
        theta = math.acos(dot)

        if theta < math.pi/3:
            penalty += (math.pi/3 - theta)

    return penalty


# ==========================================================
# 그래프 관련 함수
# ==========================================================
def _path_length_on_graph(G, nodes):
    if len(nodes) < 2:
        return 0.0
    total = 0.0
    for u,v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u,v):
            return 0.0
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total


def _apply_route_poison(G, nodes, factor=8.0):
    G2 = G.copy()

    for u,v in zip(nodes[:-1], nodes[1:]):
        if G2.has_edge(u,v):
            for key,data in G2[u][v].items():
                if "length" in data:
                    data["length"] = data["length"] * factor

        if G2.has_edge(v,u):
            for key,data in G2[v][u].items():
                if "length" in data:
                    data["length"] = data["length"] * factor

    return G2


def _build_pedestrian_graph(lat,lng,km):
    if ox is None:
        raise RuntimeError("osmnx가 없음")

    radius_m = max(700.0, km*500 + 700)

    G = ox.graph_from_point(
        (lat,lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
        retain_all=False
    )

    if not G.nodes:
        raise RuntimeError("보행자 네트워크 생성 실패")

    return G


def _nodes_to_polyline(G, nodes):
    return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in nodes]
# ==========================================================
# fallback: 사각형 루프
# ==========================================================
def _fallback_square_loop(lat,lng,km):
    target_m = max(200.0, km*1000)
    side = target_m / 4

    d_lat = side / 111111
    d_lng = side / (111111 * math.cos(math.radians(lat)))

    a = (lat+d_lat, lng)
    b = (lat+d_lat, lng+d_lng)
    c = (lat,       lng+d_lng)
    d = (lat,       lng)

    poly = [d,a,b,c,d]
    return poly, polyline_length_m(poly), polygon_roundness(poly)


# ==========================================================
# 🔥 메인 루프 생성 (거리 정밀도 유지 버전)
# ==========================================================
def generate_area_loop(lat, lng, km):

    start_time = time.time()
    target_m = max(200.0, km*1000)

    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3
    LENGTH_TOL_FRAC = 0.05
    HARD_ERR_FRAC = 0.30
    LENGTH_PENALTY_WEIGHT = 8.0

    meta = {
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

    # ---------------------------------------------------------
    # 🔥 (변경된 부분) 그래프 캐시 적용
    # ---------------------------------------------------------
    try:
        G, undirected = _get_graph_and_undirected(lat, lng, km)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat,lng,km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r,
            success=False, used_fallback=True,
            length_ok=(err <= target_m*LENGTH_TOL_FRAC),
            message=f"그래프 생성 실패: {e}"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # ---------------------------------------------------------
    # 2) 시작 노드 스냅
    # ---------------------------------------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat,lng,km)
        err = abs(length-target_m)
        meta.update(
            len=length, err=err, roundness=r,
            success=False, used_fallback=True,
            length_ok=(err <= target_m*LENGTH_TOL_FRAC),
            message=f"시작 노드 스냅 실패: {e}"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # ---------------------------------------------------------
    # 3) rod endpoint 후보 찾기
    # ---------------------------------------------------------
    try:
        dist_map = nx.single_source_dijkstra_path_length(
            undirected, start_node,
            cutoff=target_m*0.8, weight="length"
        )
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat,lng,km)
        err = abs(length-target_m)
        meta.update(
            len=length, err=err, roundness=r,
            success=False, used_fallback=True,
            message=f"rod 후보 탐색 실패: {e}"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    rod_target = target_m/2
    rod_min = rod_target*0.6
    rod_max = rod_target*1.4

    candidates = [
        n for n,d in dist_map.items()
        if rod_min <= d <= rod_max and n!=start_node
    ]

    if len(candidates) < 5:
        lo = target_m*0.25
        hi = target_m*0.75
        candidates = [
            n for n,d in dist_map.items()
            if lo <= d <= hi and n!=start_node
        ]

    if not candidates:
        poly, length, r = _fallback_square_loop(lat,lng,km)
        err = abs(length-target_m)
        meta.update(
            len=length, err=err, roundness=r, used_fallback=True,
            message="rod endpoint 부족"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    random.shuffle(candidates)
    candidates = candidates[:120]

    best_poly = None
    best_score = -1e18
    best_stats = {}

    # ---------------------------------------------------------
    # 4) forward + poisoned backward 루프 생성
    # ---------------------------------------------------------
    for endpoint in candidates:

        # forward
        try:
            forward_nodes = nx.shortest_path(
                undirected, start_node, endpoint, weight="length")
        except: continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len < target_m*0.25 or forward_len > target_m*0.8:
            continue

        poisoned = _apply_route_poison(undirected, forward_nodes, factor=8.0)

        try:
            back_nodes = nx.shortest_path(poisoned, endpoint, start_node, weight="length")
        except: continue

        back_len = _path_length_on_graph(undirected, back_nodes)
        if back_len <= 0: continue

        full_nodes = forward_nodes + back_nodes[1:]
        poly = _nodes_to_polyline(undirected, full_nodes)

        length_m = polyline_length_m(poly)
        if length_m <= 0: continue

        err = abs(length_m - target_m)
        meta["routes_checked"] += 1

        if err > target_m * HARD_ERR_FRAC:
            continue

        r  = polygon_roundness(poly)
        ov = _edge_overlap_fraction(full_nodes)
        cp = _curve_penalty(full_nodes, undirected)

        length_pen = err / max(1, target_m*LENGTH_TOL_FRAC)

        score = (
            2.5*r
            - 2.0*ov
            - 0.3*cp
            - 8.0*length_pen
        )

        length_ok = err <= target_m*LENGTH_TOL_FRAC
        if length_ok: meta["routes_validated"] += 1

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

    # ---------------------------------------------------------
    # 5) fallback 처리
    # ---------------------------------------------------------
    if best_poly is None:
        poly,length,r = _fallback_square_loop(lat,lng,km)
        err = abs(length-target_m)
        meta.update(
            len=length, err=err, roundness=r,
            used_fallback=True, success=False,
            length_ok=(err<=target_m*LENGTH_TOL_FRAC),
            message="루프 생성 실패 (fallback)"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # ---------------------------------------------------------
    # 6) 시작점 앵커링 후 결과 정리
    # ---------------------------------------------------------
    if best_poly:
        if haversine(lat,lng,best_poly[0][0],best_poly[0][1]) > 1:
            best_poly.insert(0,(lat,lng))
        if haversine(lat,lng,best_poly[-1][0],best_poly[-1][1]) > 1:
            best_poly.append((lat,lng))

        L2 = polyline_length_m(best_poly)
        E2 = abs(L2 - target_m)
        best_stats["len"] = L2
        best_stats["err"] = E2
        best_stats["length_ok"] = (E2 <= target_m*LENGTH_TOL_FRAC)

    success = best_stats["length_ok"]

    meta.update(best_stats)
    meta.update(
        success=success,
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success else
            "요청 거리와 약간 차이 있지만 가장 근접한 루프를 반환합니다."
        )
    )
    meta["time_s"] = time.time()-start_time

    return safe_list(best_poly), safe_dict(meta)
