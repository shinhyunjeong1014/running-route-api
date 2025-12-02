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

# -----------------------------
# 🔥 Redzone + R-tree (STRtree)
# -----------------------------
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import json

LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ==========================================
# 0) RedZone Loader (polygon + STRtree)
# ==========================================
def load_redzones_rtree(path: str = "redzones.geojson"):
    """geojson 로드 → polygon 목록 + STRtree 공간 인덱스 반환"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] redzones.geojson 로딩 실패: {e}")
        return [], None

    polys = []
    for elm in data.get("elements", []):
        geom = elm.get("geometry")
        if not geom:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom]  # (x, y) = (lon, lat)
        if len(coords) >= 3:
            try:
                polys.append(Polygon(coords))
            except Exception:
                continue

    if not polys:
        print("[WARN] Redzone polygons 없음")
        return [], None

    tree = STRtree(polys)
    print(f"[INFO] Loaded {len(polys)} redzone polygons with STRtree index.")
    return polys, tree


REDZONE_POLYS, REDZONE_TREE = load_redzones_rtree()


def is_in_redzone(lat: float, lon: float) -> bool:
    """R-tree 로 빠르게 후보 polygon 탐색 후 정확한 검사"""
    if REDZONE_TREE is None:
        return False
    pt = Point(lon, lat)  # shapely = (x=lon, y=lat)
    # 1) R-tree 후보 조회
    candidates = REDZONE_TREE.query(pt)
    # 2) 실제 polygon.contains 검사
    for poly in candidates:
        if poly.contains(pt):
            return True
    return False


def polyline_hits_redzone(poly: Polyline) -> bool:
    """polyline 전체 중 하나라도 redzone에 있으면 True"""
    for la, lo in poly:
        if is_in_redzone(la, lo):
            return True
    return False


# ==========================================
# JSON-safe 변환
# ==========================================
def safe_float(x: Any, default=None):
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
# ==========================================
# 거리 / 길이 유틸
# ==========================================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 거리 (meter)."""
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
    """폴리라인 전체 길이(m) 계산."""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(la1, lo1, la2, lo2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ==========================================
# roundness / overlap / 곡률 페널티
# ==========================================
def _to_local_xy(polyline: Polyline) -> List[Tuple[float, float]]:
    """위경도를 평면 좌표계로 근사 변환."""
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
    """
    isoperimetric quotient 기반 원형도: 4πA / P^2
    (1에 가까울수록 원형, 0에 가까울수록 찌그러진 형태)
    """
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


def _curve_penalty(node_path: List[int], G: nx.Graph) -> float:
    """
    연속 세 점의 각도가 너무 예리하면 페널티를 부여.
    러너가 꺾어야 하는 '급코너' 개념을 근사.
    """
    if len(node_path) < 3:
        return 0.0

    # node -> (lat, lng)
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

        # 벡터 AB, BC를 평면 상에서 근사
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
        theta = math.acos(dot)  # 라디안

        # 60도(π/3)보다 예리한 코너에 비례하여 페널티
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    """그래프 상에서 노드 시퀀스의 길이(미터)."""
    if not nodes or len(nodes) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            return 0.0
        # 멀티엣지 중 첫 번째 length 사용
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total


def _apply_route_poison(
    G: nx.MultiGraph,
    path_nodes: List[int],
    factor: float = 8.0,
) -> nx.MultiGraph:
    """
    forward 경로의 엣지 length를 늘려서
    되돌아올 때는 가급적 다른 길을 쓰도록 유도.
    (factor가 클수록 '다른 길'을 더 강하게 선호)
    """
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


# ==========================================
# OSM 보행자 그래프 구축 / 변환
# ==========================================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    OSMnx 'walk' 네트워크 타입만 사용하여
    안정적인 보행자 그래프를 생성.
    """
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    # ✅ 거리 짧을수록 반경을 조금 줄여서 효율 확보
    if km <= 1.8:
        radius_m = max(500.0, km * 600.0 + 400.0)
    else:
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


# ==========================================
# fallback: 기하학적 사각형 루프
# ==========================================
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """
    OSM/그래프를 전혀 쓰지 못할 때 사용하는 매우 단순한 정사각형 루프.
    - 실제 도로망과 맞지 않을 수 있지만, API가 완전히 죽었을 때의 최후 수단.
    """
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4.0

    # 위도 1m ≈ 1/111111 deg
    d_lat = (side / 111111.0)
    # 경도 1m ≈ 1/(111111 cos φ) deg
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))

    a = (lat + d_lat, lng)
    b = (lat + d_lat, lng + d_lng)
    c = (lat,        lng + d_lng)
    d = (lat,        lng)

    poly: Polyline = [d, a, b, c, d]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r
# ============================================================
# 1.8km 이하 전용 Local Loop Builder
# ============================================================

def _generate_local_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    1.8km 이하 요청 시 사용하는 '근거리 루프 생성기'.
    - rod/poisoning 사용 안함
    - 반경 r 내의 subgraph에서 모든 노드-노드 루프 탐색
    - roundness / overlap / curve_penalty 기반 최적 루프 선택
    - redzone 완전 회피
    """

    start_time = time.time()
    target_m = max(300.0, km * 1000.0)

    # -----------------------------------------
    # 스코어링 파라미터
    # -----------------------------------------
    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY  = 2.0
    CURVE_WEIGHT     = 0.3
    LENGTH_TOL_FRAC  = 0.05   # ±5%
    HARD_ERR_FRAC    = 0.25   # ±25%는 폐기
    LEN_PEN_WEIGHT   = 7.0

    meta = dict(
        len=0, err=0, roundness=0, overlap=0, curve_penalty=0,
        score=-1e18, success=False, length_ok=False, used_fallback=False,
        routes_checked=0, routes_validated=0,
        km_requested=km, target_m=target_m,
        time_s=0.0, message=""
    )

    # -----------------------------------------
    # 1) 보행자 그래프 로딩 (근거리 반경)
    # -----------------------------------------
    try:
        # radius = max(300 m, km*600 + 300)
        radius_m = max(300.0, km * 600.0 + 300.0)
        G = ox.graph_from_point(
            (lat, lng),
            dist=radius_m,
            network_type="walk",
            simplify=True,
            retain_all=False,
        )
    except Exception as e:
        # fallback
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length-target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message=f"local graph load 실패: {e}"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    if not G.nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length-target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message="local graph empty"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # undirected
    try:
        UG = ox.utils_graph.get_undirected(G)
    except Exception:
        UG = G.to_undirected()

    # start node 찾기
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length-target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message=f"local start snap 실패: {e}"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # -----------------------------------------
    # 2) start에서 Dijkstra로 400~800m 탐색
    # -----------------------------------------
    try:
        dist_map = nx.single_source_dijkstra_path_length(
            UG, start_node,
            cutoff=max(300.0, target_m*0.8),
            weight="length"
        )
    except:
        dist_map = {}

    if not dist_map:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length-target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message="local dijkstra empty"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # -----------------------------------------
    # 3) 루프 endpoint 후보 추출
    #    목표는: start → u → ... → v → start
    #    대략 loop length ≈ 1000m ~ 1800m 범위
    # -----------------------------------------
    min_forward = target_m * 0.3
    max_forward = target_m * 1.0

    endpoints = [n for n, d in dist_map.items()
                 if min_forward <= d <= max_forward]

    # redzone 제거
    filtered = []
    for n in endpoints:
        lat_n = float(UG.nodes[n]["y"])
        lon_n = float(UG.nodes[n]["x"])
        if not is_in_redzone(lat_n, lon_n):
            filtered.append(n)
    endpoints = filtered

    if len(endpoints) == 0:
        # fallback
        poly, length, r = _fallback_square_loop(lat, lng, km)
        meta.update(
            len=length, err=abs(length-target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True,
            message="local endpoints 없음"
        )
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # 너무 많으면 샘플링
    random.shuffle(endpoints)
    endpoints = endpoints[:80]

    best_poly = None
    best_score = -1e18
    best_stats = {}

    # -----------------------------------------
    # 4) 모든 endpoint u,v 쌍 탐색
    # -----------------------------------------
    for u in endpoints:
        # 4-1) start→u path
        try:
            path1 = nx.shortest_path(UG, start_node, u, weight="length")
        except:
            continue

        # redzone check
        skip = False
        for n in path1:
            la_n = float(UG.nodes[n]["y"])
            lo_n = float(UG.nodes[n]["x"])
            if is_in_redzone(la_n, lo_n):
                skip = True
                break
        if skip:
            continue

        path1_len = _path_length_on_graph(UG, path1)
        if path1_len <= 0:
            continue

        # u 기준 다시 도달 가능한 endpoint v들
        for v in endpoints:
            if u == v:
                continue

            # u→v
            try:
                path2 = nx.shortest_path(UG, u, v, weight="length")
            except:
                continue

            # v→start
            try:
                path3 = nx.shortest_path(UG, v, start_node, weight="length")
            except:
                continue

            full_nodes = path1 + path2[1:] + path3[1:]
            meta["routes_checked"] += 1

            # polyline 변환
            poly = _nodes_to_polyline(UG, full_nodes)
            length_m = polyline_length_m(poly)
            if length_m <= 0:
                continue

            # redzone 검사
            if polyline_hits_redzone(poly):
                continue

            # 거리 오차 너무 큰 것은 제외
            err = abs(length_m - target_m)
            if err > target_m * HARD_ERR_FRAC:
                continue

            r = polygon_roundness(poly)
            ov = _edge_overlap_fraction(full_nodes)
            cp = _curve_penalty(full_nodes, UG)

            length_pen = err / (max(1.0, target_m * LENGTH_TOL_FRAC))
            score = (
                ROUNDNESS_WEIGHT*r
                - OVERLAP_PENALTY*ov
                - CURVE_WEIGHT*cp
                - LEN_PEN_WEIGHT*length_pen
            )

            length_ok = (err <= target_m * LENGTH_TOL_FRAC)
            if length_ok:
                meta["routes_validated"] += 1

            if score > best_score:
                best_score = score
                best_poly  = poly
                best_stats = dict(
                    len=length_m, err=err, roundness=r,
                    overlap=ov, curve_penalty=cp,
                    score=score, length_ok=length_ok
                )

    # -----------------------------------------
    # 5) fallback
    # -----------------------------------------
    if best_poly is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        best_stats = dict(
            len=length, err=abs(length-target_m),
            roundness=r, overlap=0, curve_penalty=0,
            score=r, length_ok=False
        )
        meta.update(best_stats)
        meta["used_fallback"] = True
        meta["message"] = "local loop 생성 실패(fallback)"
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # -----------------------------------------
    # 6) 시작점 앵커링
    # -----------------------------------------
    first_la, first_lo = best_poly[0]
    if haversine(lat, lng, first_la, first_lo) > 1.0:
        best_poly.insert(0, (lat, lng))

    last_la, last_lo = best_poly[-1]
    if haversine(lat, lng, last_la, last_lo) > 1.0:
        best_poly.append((lat, lng))

    # 거리 업데이트
    length2 = polyline_length_m(best_poly)
    err2 = abs(length2 - target_m)
    best_stats["len"] = length2
    best_stats["err"] = err2
    best_stats["length_ok"] = (err2 <= target_m * LENGTH_TOL_FRAC)

    meta.update(best_stats)
    meta["success"] = best_stats["length_ok"]
    meta["message"] = "근거리 최적 루프 생성 완료"
    meta["time_s"] = time.time() - start_time

    return safe_list(best_poly), safe_dict(meta)
# ============================================================
# 메인: 러닝 루프 생성기 (통합 버전)
# ============================================================

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    요청 좌표(lat, lng)와 목표 거리(km)를 기반으로
    '요청거리 정확도'와 '루프 모양'을 동시에 고려한 러닝 루프를 생성한다.

    - km <= 1.8  : 근거리 Local Loop Builder (_generate_local_loop)
    - km >  1.8  : 기존 rod + poisoning 기반 루프 (모양/길이 최적화)
    - redzones.geojson 에 정의된 아파트 단지 등은 절대 진입하지 않음
    """
    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

    # --------------------------------------------------------
    # km <= 1.8km : 근거리 전용 알고리즘 사용
    # --------------------------------------------------------
    if km <= 1.8:
        poly, meta = _generate_local_loop(lat, lng, km)

        # poly는 tuple 리스트 그대로, meta만 JSON-safe 처리
        meta = safe_dict(meta)
        meta["time_s"] = time.time() - start_time
        return poly, meta

    # --------------------------------------------------------
    # km > 1.8km : 기존 rod + poisoning 루프 생성
    # --------------------------------------------------------

    # 스코어링 가중치
    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3

    # 길이 관련 가중치
    LENGTH_TOL_FRAC = 0.05       # "정상 범위" ±5%
    HARD_ERR_FRAC = 0.30         # 이 범위를 벗어나면 아예 버린다 (±30%)
    LENGTH_PENALTY_WEIGHT = 8.0  # 오차 5%일 때 -8, 10%일 때 -16 정도

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
        "valhalla_calls": 0,
        "kakao_calls": 0,
        "routes_checked": 0,
        "routes_validated": 0,
        "km_requested": km,
        "target_m": target_m,
        "time_s": 0.0,
        "message": "",
    }

    # --------------------------------------------------------
    # 1) 보행자 그래프 로딩
    # --------------------------------------------------------
    try:
        G = _build_pedestrian_graph(lat, lng, km)
    except Exception as e:
        # 그래프 생성 자체가 안 되면 바로 기하학적 사각형 루프 사용
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
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message=f"OSM 보행자 그래프 생성 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # undirected 그래프
    try:
        undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    # --------------------------------------------------------
    # 1-1) redzone 노드 제거 (아파트 단지 등)
    # --------------------------------------------------------
    # 노드 좌표가 redzone 안에 있으면 해당 노드 삭제
    remove_nodes = []
    for n, data in list(undirected.nodes(data=True)):
        la = float(data.get("y"))
        lo = float(data.get("x"))
        if is_in_redzone(la, lo):
            remove_nodes.append(n)
    if remove_nodes:
        undirected.remove_nodes_from(remove_nodes)

    if undirected.number_of_nodes() == 0:
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
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="redzone 필터링 후 사용 가능한 노드가 없어 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # --------------------------------------------------------
    # 2) 시작 노드 스냅
    # --------------------------------------------------------
    try:
        start_node = ox.distance.nearest_nodes(undirected, X=lng, Y=lat)
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
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message=f"시작 좌표를 그래프에 스냅하지 못해 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # --------------------------------------------------------
    # 3) start에서 단일-출발 최단거리 (rod 후보 탐색)
    #    - target/2 근처 노드를 rod endpoint 후보로 사용
    # --------------------------------------------------------
    try:
        dist_from_start: Dict[int, float] = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.8,  # 너무 멀리까지는 탐색하지 않음
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
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message=f"그래프 최단거리 탐색 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    rod_target = target_m / 2.0
    rod_min = rod_target * 0.6   # ≈ 0.3 * target
    rod_max = rod_target * 1.4   # ≈ 0.7 * target

    candidate_nodes = [
        n for n, d in dist_from_start.items()
        if rod_min <= d <= rod_max and n != start_node
    ]

    # 후보가 너무 적으면 조건을 조금 완화
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
            len=length,
            err=err,
            roundness=r,
            overlap=0.0,
            curve_penalty=0.0,
            score=r,
            success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="적절한 rod endpoint 후보를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # 너무 많으면 샘플링
    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:120]

    best_score = -1e18
    best_poly: Optional[Polyline] = None
    best_stats: Dict[str, Any] = {}

    # --------------------------------------------------------
    # 4) 각 endpoint에 대해 'forward + poisoned backward' 루프 생성
    # --------------------------------------------------------
    for endpoint in candidate_nodes:
        # 4-1. forward
        try:
            forward_nodes = nx.shortest_path(
                undirected,
                start_node,
                endpoint,
                weight="length",
            )
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len <= 0.0:
            continue

        # 지나치게 짧은 rod / 지나치게 긴 rod는 제외
        if forward_len < target_m * 0.25 or forward_len > target_m * 0.8:
            continue

        # 4-2. forward poisoning 적용
        poisoned = _apply_route_poison(undirected, forward_nodes, factor=8.0)

        # 4-3. poisoned 그래프에서 backward
        try:
            back_nodes = nx.shortest_path(
                poisoned,
                endpoint,
                start_node,
                weight="length",
            )
        except Exception:
            continue

        back_len = _path_length_on_graph(undirected, back_nodes)
        if back_len <= 0.0:
            continue

        # forward + backward를 붙여서 하나의 루프
        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        poly = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(poly)
        if length_m <= 0.0:
            continue

        # 🔴 redzone을 한 번이라도 지나면 버림
        if polyline_hits_redzone(poly):
            continue

        err = abs(length_m - target_m)

        # 길이가 너무 짧거나 너무 길면 (±30% 이상) 아예 후보에서 제외
        if err > target_m * HARD_ERR_FRAC:
            continue

        r = polygon_roundness(poly)
        ov = _edge_overlap_fraction(full_nodes)
        cp = _curve_penalty(full_nodes, undirected)

        # 길이 오차를 "허용 오차 대비 몇 배"인지로 정규화
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
                "len": length_m,
                "err": err,
                "roundness": r,
                "overlap": ov,
                "curve_penalty": cp,
                "score": score,
                "length_ok": length_ok,
            }

    # --------------------------------------------------------
    # 5) 후보 루프가 하나도 없으면 fallback
    # --------------------------------------------------------
    if best_poly is None:
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
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="논문 기반 OSM 루프 생성에 실패하여 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return poly, safe_dict(meta)

    # --------------------------------------------------------
    # 6) 시작 좌표 앵커링 + 길이/오차 재계산
    # --------------------------------------------------------
    used_fallback = False

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
        length_ok2 = err2 <= target_m * LENGTH_TOL_FRAC

        # 길이 관련 메타데이터 업데이트
        best_stats["len"] = length2
        best_stats["err"] = err2
        best_stats["length_ok"] = length_ok2

    success = bool(best_stats.get("length_ok"))

    meta.update(best_stats)
    meta.update(
        success=success,
        used_fallback=used_fallback,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else f"요청 오차(±{int(target_m * LENGTH_TOL_FRAC)}m)를 초과하지만, 가장 인접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    return best_poly, safe_dict(meta)