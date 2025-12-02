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

# 🔥 Redzone 회피 기능: shapely + geojson
from shapely.geometry import Polygon, Point
import json

LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ============================================================
# 🔥 0) Red Zone Loader (아파트 단지 / 주거지역 제외)
# ============================================================
def load_redzones(path: str = "redzones.geojson") -> List[Polygon]:
    """Overpass로 생성한 redzones.geojson을 읽어 polygon 목록 반환."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] redzones.geojson 로딩 실패: {e}")
        return []

    polys = []
    for elm in data.get("elements", []):
        geom = elm.get("geometry")
        if not geom:
            continue

        coords = [(p["lon"], p["lat"]) for p in geom]
        if len(coords) >= 3:
            try:
                polys.append(Polygon(coords))
            except Exception:
                continue

    print(f"[INFO] Loaded {len(polys)} redzone polygons")
    return polys


REDZONES = load_redzones()


def is_in_redzone(lat: float, lon: float) -> bool:
    """경로 polyline 좌표가 redzone polygon 안에 포함되면 True."""
    pt = Point(lon, lat)  # shapely는 (x=lon, y=lat)
    for poly in REDZONES:
        if poly.contains(pt):
            return True
    return False


# ============================================================
# JSON-safe 변환
# ============================================================
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


# ============================================================
# 거리/길이
# ============================================================
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
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polyline_length_m(polyline: Polyline) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(la1, lo1, la2, lo2)
    return total


# ============================================================
# roundness / overlap / curve penalty
# ============================================================
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
    if len(polyline) < 3:
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
    return float(4 * math.pi * area / (perimeter ** 2))


def _edge_overlap_fraction(node_path: List[int]) -> float:
    if len(node_path) < 2:
        return 0.0
    edge_counts = {}
    for u, v in zip(node_path[:-1], node_path[1:]):
        if u == v:
            continue
        e = (u, v) if u <= v else (v, u)
        edge_counts[e] = edge_counts.get(e, 0) + 1
    if not edge_counts:
        return 0.0
    overlap = sum(1 for c in edge_counts.values() if c > 1)
    return overlap / len(edge_counts)


def _curve_penalty(node_path: List[int], G: nx.Graph) -> float:
    if len(node_path) < 3:
        return 0.0

    coords = {}
    for n in node_path:
        if n in coords:
            continue
        node = G.nodes[n]
        coords[n] = (float(node["y"]), float(node["x"]))

    penalty = 0.0
    R = 6371000.0
    for i in range(1, len(node_path) - 1):
        a, b, c = node_path[i - 1], node_path[i], node_path[i + 1]
        lat_a, lon_a = coords[a]
        lat_b, lon_b = coords[b]
        lat_c, lon_c = coords[c]

        def to_xy(lat, lon):
            return (
                R * math.radians(lon - lon_b) * math.cos(math.radians(lat_b)),
                R * math.radians(lat - lat_b),
            )

        x1, y1 = to_xy(lat_a, lon_a)
        x2, y2 = to_xy(lat_c, lon_c)

        n1 = math.hypot(x1, y1)
        n2 = math.hypot(x2, y2)
        if n1 == 0 or n2 == 0:
            continue
        dot = (x1 * x2 + y1 * y2) / (n1 * n2)
        dot = max(-1, min(1, dot))
        theta = math.acos(dot)

        if theta < math.pi / 3:
            penalty += (math.pi / 3 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    total = 0
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
            data["length"] = float(data["length"]) * factor
        if G2.has_edge(v, u):
            for key in list(G2[v][u].keys()):
                data = G2[v][u][key]
                data["length"] = float(data["length"]) * factor
    return G2


# ============================================================
# OSM 그래프 구축
# ============================================================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    if ox is None:
        raise RuntimeError("osmnx가 설치되지 않음")

    radius_m = max(700.0, km * 500.0 + 700.0)
    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )
    if not G.nodes:
        raise RuntimeError("그래프 생성 실패")
    return G


def _nodes_to_polyline(G: nx.MultiDiGraph, nodes: List[int]) -> Polyline:
    poly = []
    for n in nodes:
        node = G.nodes[n]
        poly.append((float(node["y"]), float(node["x"])))
    return poly


# ============================================================
# fallback 사각형 루프
# ============================================================
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    target_m = max(200.0, km * 1000.0)
    side = target_m / 4

    d_lat = side / 111111.0
    d_lng = side / (111111.0 * math.cos(math.radians(lat)))

    a = (lat + d_lat, lng)
    b = (lat + d_lat, lng + d_lng)
    c = (lat, lng + d_lng)
    d = (lat, lng)

    poly = [d, a, b, c, d]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ============================================================
# 메인 loop 생성기
# ============================================================
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3
    LENGTH_TOL_FRAC = 0.05
    HARD_ERR_FRAC = 0.30
    LENGTH_PENALTY_WEIGHT = 8.0

    meta = dict(
        len=0, err=0, roundness=0, overlap=0, curve_penalty=0,
        score=-1e18, success=False, length_ok=False,
        used_fallback=False, valhalla_calls=0, kakao_calls=0,
        routes_checked=0, routes_validated=0,
        km_requested=km, target_m=target_m, time_s=0.0, message=""
    )

    # --------------------------------------------------
    # 1) 그래프 로딩
    # --------------------------------------------------
    try:
        G = _build_pedestrian_graph(lat, lng, km)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0, curve_penalty=0,
            score=r, success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True, message=f"그래프 생성 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    try:
        undirected = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    # --------------------------------------------------
    # 2) 시작 노드 스냅
    # --------------------------------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0, curve_penalty=0,
            score=r, success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True, message=f"시작점 스냅 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------------------------------
    # 3) start에서 dijkstra 탐색
    # --------------------------------------------------
    try:
        dist_from_start = nx.single_source_dijkstra_path_length(
            undirected, start_node, cutoff=target_m * 0.8, weight="length"
        )
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0, curve_penalty=0,
            score=r, success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True, message=f"최단거리 탐색 실패: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    rod_target = target_m / 2
    rod_min = rod_target * 0.6
    rod_max = rod_target * 1.4

    # --------------------------------------------------
    # 3-A) rod endpoint 후보 추출 + 레드존 필터
    # --------------------------------------------------
    candidate_nodes = []
    for n, d in dist_from_start.items():
        if n == start_node:
            continue
        if rod_min <= d <= rod_max:
            lat_n = float(undirected.nodes[n]["y"])
            lon_n = float(undirected.nodes[n]["x"])
            if not is_in_redzone(lat_n, lon_n):      # 🔥 레드존 노드 제외
                candidate_nodes.append(n)

    if len(candidate_nodes) < 5:
        lo = target_m * 0.25
        hi = target_m * 0.75
        for n, d in dist_from_start.items():
            if lo <= d <= hi and n != start_node:
                lat_n = float(undirected.nodes[n]["y"])
                lon_n = float(undirected.nodes[n]["x"])
                if not is_in_redzone(lat_n, lon_n):  # 🔥 레드존
                    candidate_nodes.append(n)

    if not candidate_nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0, curve_penalty=0,
            score=r, success=False, length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True, message="rod endpoint 후보 없음",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:120]

    best_score = -1e18
    best_poly = None
    best_stats = {}

    # --------------------------------------------------
    # 4) 후보 endpoint로 루프 생성
    # --------------------------------------------------
    for endpoint in candidate_nodes:
        try:
            forward_nodes = nx.shortest_path(
                undirected, start_node, endpoint, weight="length"
            )
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len <= 0:
            continue
        if forward_len < target_m * 0.25 or forward_len > target_m * 0.8:
            continue

        # forward segment 레드존 체크
        for n in forward_nodes:
            lat_n = float(undirected.nodes[n]["y"])
            lon_n = float(undirected.nodes[n]["x"])
            if is_in_redzone(lat_n, lon_n):       # 🔥 forward 경로에서 레드존이면 폐기
                forward_nodes = None
                break
        if forward_nodes is None:
            continue

        poisoned = _apply_route_poison(undirected, forward_nodes, factor=8.0)

        try:
            back_nodes = nx.shortest_path(
                poisoned, endpoint, start_node, weight="length"
            )
        except Exception:
            continue

        back_len = _path_length_on_graph(undirected, back_nodes)
        if back_len <= 0:
            continue

        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        # polyline 생성
        poly = _nodes_to_polyline(undirected, full_nodes)
        if not poly:
            continue

        # 🔥 polyline 전체 레드존 체크
        skip = False
        for la, lo in poly:
            if is_in_redzone(la, lo):           # 🔥 루프 전체 진입 금지
                skip = True
                break
        if skip:
            continue

        length_m = polyline_length_m(poly)
        if length_m <= 0:
            continue

        err = abs(length_m - target_m)
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
            best_stats = dict(
                len=length_m, err=err, roundness=r,
                overlap=ov, curve_penalty=cp,
                score=score, length_ok=length_ok
            )

    # --------------------------------------------------
    # 5) fallback
    # --------------------------------------------------
    if best_poly is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)

        # fallback도 레드존 침투 검사
        if any(is_in_redzone(la, lo) for la, lo in poly):
            # fallback까지 실패 → 완전불가
            meta.update(
                len=0, err=0, roundness=0, overlap=0,
                curve_penalty=0, score=-1e18,
                success=False, length_ok=False,
                used_fallback=True,
                message="레드존 때문에 fallback 루프도 생성 불가"
            )
            meta["time_s"] = time.time() - start_time
            return [], safe_dict(meta)

        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0,
            curve_penalty=0, score=r, success=False,
            length_ok=(err <= target_m * LENGTH_TOL_FRAC),
            used_fallback=True,
            message="fallback 정사각형 루프 사용",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------------------------------
    # 6) best_poly 정리 및 앵커링
    # --------------------------------------------------
    first_la, first_lo = best_poly[0]
    if haversine(lat, lng, first_la, first_lo) > 1:
        best_poly.insert(0, (lat, lng))

    last_la, last_lo = best_poly[-1]
    if haversine(lat, lng, last_la, last_lo) > 1:
        best_poly.append((lat, lng))

    length2 = polyline_length_m(best_poly)
    err2 = abs(length2 - target_m)
    length_ok2 = err2 <= target_m * LENGTH_TOL_FRAC

    best_stats.update(len=length2, err=err2, length_ok=length_ok2)

    meta.update(best_stats)
    meta.update(
        success=best_stats.get("length_ok", False),
        used_fallback=False,
        message="최적의 정밀 경로가 도출되었습니다."
        if best_stats.get("length_ok", False)
        else f"요청 오차 초과(±{int(target_m * LENGTH_TOL_FRAC)}m)지만 최적 루프 반환",
    )
    meta["time_s"] = time.time() - start_time

    return safe_list(best_poly), safe_dict(meta)