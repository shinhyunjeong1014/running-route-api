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

# ─────────────────────────────────────────
# 🔥 Redzone (아파트/주거지) 회피용: shapely + STRtree(R-tree)
# ─────────────────────────────────────────
try:
    from shapely.geometry import Polygon, Point
    try:
        from shapely.strtree import STRtree
    except Exception:
        STRtree = None  # shapely 버전에 따라 없을 수 있음
except Exception:
    Polygon = None
    Point = None
    STRtree = None

import json

LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ============================================================
# 0) Red Zone Loader + R-tree 인덱스
# ============================================================

REDZONE_POLYGONS: List[Any] = []
REDZONE_INDEX: Optional[Any] = None  # STRtree or None


def _init_redzones(path: str = "redzones.geojson") -> None:
    """
    Overpass로 생성한 redzones.geojson을 읽어
    Polygon 목록과 STRtree(R-tree) 인덱스를 준비한다.
    """
    global REDZONE_POLYGONS, REDZONE_INDEX

    if Polygon is None:
        print("[INFO] shapely 설치 안 됨 → redzone 기능 비활성화")
        REDZONE_POLYGONS = []
        REDZONE_INDEX = None
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] redzones.geojson 로딩 실패: {e} → redzone 기능 비활성화")
        REDZONE_POLYGONS = []
        REDZONE_INDEX = None
        return

    polys: List[Any] = []
    for elm in data.get("elements", []):
        geom = elm.get("geometry")
        if not geom:
            continue
        # Overpass "out geom" 형식: [{"lat":..., "lon":...}, ...]
        coords = [(g["lon"], g["lat"]) for g in geom]
        if len(coords) < 3:
            continue
        try:
            polys.append(Polygon(coords))
        except Exception:
            continue

    REDZONE_POLYGONS = polys
    if polys and STRtree is not None:
        REDZONE_INDEX = STRtree(polys)
        print(f"[INFO] Redzone polygons: {len(polys)}개, STRtree 인덱스 생성 완료")
    else:
        REDZONE_INDEX = None
        print(f"[INFO] Redzone polygons: {len(polys)}개, STRtree 사용 불가 (shapely 버전/설치 확인 필요)")


def is_in_redzone(lat: float, lon: float) -> bool:
    """
    주어진 위경도 좌표가 redzone polygon 안에 있으면 True.
    STRtree(R-tree) 인덱스를 사용해 빠르게 검사한다.
    """
    if not REDZONE_POLYGONS or Point is None:
        return False

    pt = Point(lon, lat)

    # 인덱스 있으면 후보만 조회
    if REDZONE_INDEX is not None:
        candidates = REDZONE_INDEX.query(pt)
    else:
        # 최악의 경우: 선형 스캔 (성능 저하 있지만 fallback)
        candidates = REDZONE_POLYGONS

    for poly in candidates:
        try:
            if poly.contains(pt):
                return True
        except Exception:
            continue
    return False


def polyline_hits_redzone(poly: Polyline, sample_step: int = 3) -> bool:
    """
    polyline이 redzone과 교차하는지 간단히 검사.
    - 모든 점을 다 검사하지 않고 sample_step 간격으로 샘플링.
    - 시작점/끝점은 항상 검사.
    """
    if not REDZONE_POLYGONS or not poly:
        return False

    n = len(poly)
    for i, (lat, lon) in enumerate(poly):
        if i not in (0, n - 1) and (i % sample_step != 0):
            continue
        if is_in_redzone(lat, lon):
            return True
    return False


# 모듈 import 시점에 한 번만 초기화
_init_redzones()


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
# 거리 / 길이
# ============================================================
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
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(la1, lo1, la2, lo2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ============================================================
# roundness / overlap / 곡률 페널티
# ============================================================
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

    coords: Dict[int, Tuple[float, float]] = {}
    for n in node_path:
        if n in coords:
            continue
        node = G.nodes[n]
        coords[n] = (float(node.get("y")), float(node.get("x")))

    penalty = 0.0
    R = 6371000.0

    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        lat_a, lng_a = coords[a]
        lat_b, lng_b = coords[b]
        lat_c, lng_c = coords[c]

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
        data = next(iter(G[u][v].values()))
        total += float(data.get("length", 0.0))
    return total


def _apply_route_poison(G: nx.MultiGraph, path_nodes: List[int], factor: float = 8.0) -> nx.MultiGraph:
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


# ============================================================
# OSM 보행자 그래프 구축
# ============================================================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    OSMnx 'walk' 네트워크 타입만 사용하여
    안정적인 보행자 그래프를 생성.
    """
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    # API 부하와 커버리지를 고려한 반경 (meter)
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


# ============================================================
# fallback: 기하학적 사각형 루프
# ============================================================
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
# 메인: 러닝 루프 생성기 (레드존 회피 + R-tree)
# ============================================================
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    요청 좌표(lat, lng)와 목표 거리(km)를 기반으로
    '요청거리 정확도'와 '루프 모양'을 동시에 고려한 러닝 루프를 생성한다.

    - OSM walk 그래프 + rod + poisoning 기반
    - Redzone(아파트/주거지) 회피: endpoint, forward, full polyline에 모두 적용
    """
    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

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
        # fallback도 레드존 검사
        if polyline_hits_redzone(poly):
            meta.update(
                len=0.0,
                err=0.0,
                roundness=0.0,
                overlap=0.0,
                curve_penalty=0.0,
                score=-1e18,
                success=False,
                length_ok=False,
                used_fallback=True,
                message=f"그래프 생성 실패 + 레드존 충돌로 경로 생성 불가: {e}",
            )
            meta["time_s"] = time.time() - start_time
            return [], safe_dict(meta)

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
        return safe_list(poly), safe_dict(meta)

    # undirected 그래프
    try:
        undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    # --------------------------------------------------------
    # 2) 시작 노드 스냅
    # --------------------------------------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat) if ox is not None else None
        if start_node is None:
            raise RuntimeError("nearest_nodes 실패")
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        if polyline_hits_redzone(poly):
            meta.update(
                len=0.0,
                err=0.0,
                roundness=0.0,
                overlap=0.0,
                curve_penalty=0.0,
                score=-1e18,
                success=False,
                length_ok=False,
                used_fallback=True,
                message=f"시작 좌표 스냅 실패 + 레드존 충돌로 경로 생성 불가: {e}",
            )
            meta["time_s"] = time.time() - start_time
            return [], safe_dict(meta)

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
        return safe_list(poly), safe_dict(meta)

    # --------------------------------------------------------
    # 3) start에서 단일-출발 최단거리
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
        if polyline_hits_redzone(poly):
            meta.update(
                len=0.0,
                err=0.0,
                roundness=0.0,
                overlap=0.0,
                curve_penalty=0.0,
                score=-1e18,
                success=False,
                length_ok=False,
                used_fallback=True,
                message=f"그래프 최단거리 탐색 실패 + 레드존 충돌로 경로 생성 불가: {e}",
            )
            meta["time_s"] = time.time() - start_time
            return [], safe_dict(meta)

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
        return safe_list(poly), safe_dict(meta)

    rod_target = target_m / 2.0
    rod_min = rod_target * 0.6   # ≈ 0.3 * target
    rod_max = rod_target * 1.4   # ≈ 0.7 * target

    candidate_nodes: List[int] = []

    # rod 범위 내 + 레드존이 아닌 노드를 endpoint 후보로 사용
    for n, d in dist_from_start.items():
        if n == start_node:
            continue
        if rod_min <= d <= rod_max:
            node = undirected.nodes[n]
            lat_n = float(node.get("y"))
            lng_n = float(node.get("x"))
            if not is_in_redzone(lat_n, lng_n):
                candidate_nodes.append(n)

    # 후보가 너무 적으면 조건 완화
    if len(candidate_nodes) < 5:
        lo = target_m * 0.25
        hi = target_m * 0.75
        for n, d in dist_from_start.items():
            if n == start_node:
                continue
            if lo <= d <= hi:
                node = undirected.nodes[n]
                lat_n = float(node.get("y"))
                lng_n = float(node.get("x"))
                if not is_in_redzone(lat_n, lng_n):
                    candidate_nodes.append(n)

    if not candidate_nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        if polyline_hits_redzone(poly):
            meta.update(
                len=0.0,
                err=0.0,
                roundness=0.0,
                overlap=0.0,
                curve_penalty=0.0,
                score=-1e18,
                success=False,
                length_ok=False,
                used_fallback=True,
                message="rod endpoint 후보 없음 + 레드존 충돌로 경로 생성 불가",
            )
            meta["time_s"] = time.time() - start_time
            return [], safe_dict(meta)

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
        return safe_list(poly), safe_dict(meta)

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

        # 🔥 forward 경로가 레드존을 지나면 제외
        forward_poly = _nodes_to_polyline(undirected, forward_nodes)
        if polyline_hits_redzone(forward_poly):
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

        # 🔥 전체 루프가 레드존을 침범하면 후보 제외
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

        # fallback도 레드존 검사
        if polyline_hits_redzone(poly):
            meta.update(
                len=0.0,
                err=0.0,
                roundness=0.0,
                overlap=0.0,
                curve_penalty=0.0,
                score=-1e18,
                success=False,
                length_ok=False,
                used_fallback=True,
                message="논문 기반 OSM 루프 + fallback 모두 레드존과 충돌하여 경로 생성 불가",
            )
            meta["time_s"] = time.time() - start_time
            return [], safe_dict(meta)

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
        return safe_list(poly), safe_dict(meta)

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

    return safe_list(best_poly), safe_dict(meta)