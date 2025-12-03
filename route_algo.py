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


# ==========================
# 그래프 캐시 (성능 최적화)
# ==========================
# 동일/인접 좌표에서 반복 호출 시 OSMnx 그래프를 재사용하기 위한 캐시.
# - 키: (라운딩된 위도, 경도, 요청거리(km) 버킷)
# - 값: {"G": nx.MultiDiGraph, "undirected": nx.MultiGraph}
_GRAPH_CACHE: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
_GRAPH_CACHE_MAX_SIZE = 8  # 메모리 보호용 최대 캐시 개수 (필요시 조정 가능)


def _graph_cache_key(lat: float, lng: float, km: float) -> Tuple[int, int, int]:
    """위/경도는 0.0025도(약 250m) 단위로 라운딩, km는 0.5km 단위 버킷.
    너무 촘촘하면 캐시효과가 떨어지고, 너무 넓으면 과도한 재사용이 발생하므로
    실험적으로 적당한 수준으로 설정.
    """
    lat_key = int(round(lat / 0.0025))
    lng_key = int(round(lng / 0.0025))
    km_key = int(max(1, round(km / 0.5)))
    return lat_key, lng_key, km_key


def _get_graph_and_undirected(lat: float, lng: float, km: float) -> Tuple[nx.MultiDiGraph, nx.MultiGraph]:
    """osmnx 그래프 + undirected 그래프를 캐시 기반으로 반환.

    - 최초 호출: OSMnx에서 그래프 구성 후 undirected 변환까지 수행한 뒤 캐시에 저장
    - 동일/인접 좌표 & 거리 버킷에서 재호출 시: 캐시에서 즉시 반환
    """
    if ox is None:
        # 배포 환경에서 osmnx 미설치일 수 있으므로 기존 예외 방식 유지
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    key = _graph_cache_key(lat, lng, km)
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached["G"], cached["undirected"]

    # 캐시에 없으면 새로 생성
    G = _build_pedestrian_graph(lat, lng, km)
    try:
        undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)
    except Exception:
        undirected = G.to_undirected()

    _GRAPH_CACHE[key] = {"G": G, "undirected": undirected}

    # 아주 단순한 FIFO 기반 캐시 축소 (LRU까지는 필요 없다고 가정)
    if len(_GRAPH_CACHE) > _GRAPH_CACHE_MAX_SIZE:
        # 가장 먼저 추가된 키 하나 제거
        first_key = next(iter(_GRAPH_CACHE.keys()))
        if first_key != key:
            _GRAPH_CACHE.pop(first_key, None)

    return G, undirected


# ==========================
# JSON-safe 변환 유틸
# ==========================
def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """NaN / Inf 값을 JSON 직렬화 가능한 값으로 변환."""
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
            out.append(v)
    return out


def safe_dict(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = safe_float(v)
        elif isinstance(v, (list, tuple)):
            out[k] = safe_list(v)
        elif isinstance(v, dict):
            out[k] = safe_dict(v)
        else:
            out[k] = v
    return out


# ==========================
# 거리 계산 / polyline 길이
# ==========================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 대원거리(m)."""
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2.0) ** 2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline: Polyline) -> float:
    """polyline 전체 길이(m)."""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lng1), (lat2, lng2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lng1, lat2, lng2)
    return total


# ==========================
# roundness / overlap / 곡률 페널티
# ==========================
def _to_local_xy(polyline: Polyline) -> List[Tuple[float, float]]:
    """위경도를 평면 좌표계로 근사 변환."""
    if not polyline:
        return []
    lats = [p[0] for p in polyline]
    lngs = [p[1] for p in polyline]
    lat0 = sum(lats) / len(lats)
    lng0 = sum(lngs) / len(lngs)
    R = 6371000.0

    res: List[Tuple[float, float]] = []
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

    # shoelace formula로 면적
    area = 0.0
    perimeter = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        area += x1 * y2 - x2 * y1
        dx = x2 - x1
        dy = y2 - y1
        perimeter += math.hypot(dx, dy)
    area = abs(area) / 2.0
    if perimeter <= 0.0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


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
    return overlap_edges / float(len(edge_counts))


def _curve_penalty(polyline: Polyline) -> float:
    """
    polyline의 곡률이 너무 심한 부분(급격한 꺾임)을 페널티로 환산.
    단위: (라디안 차이 누적) / N
    """
    if not polyline or len(polyline) < 3:
        return 0.0
    xy = _to_local_xy(polyline)
    if len(xy) < 3:
        return 0.0

    total_angle_change = 0.0
    count = 0
    for (x1, y1), (x2, y2), (x3, y3) in zip(xy[:-2], xy[1:-1], xy[2:]):
        v1x, v1y = x2 - x1, y2 - y1
        v2x, v2y = x3 - x2, y3 - y2
        norm1 = math.hypot(v1x, v1y)
        norm2 = math.hypot(v2x, v2y)
        if norm1 <= 0 or norm2 <= 0:
            continue
        dot = (v1x * v2x + v1y * v2y) / (norm1 * norm2)
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)
        total_angle_change += angle
        count += 1
    if count == 0:
        return 0.0
    return total_angle_change / count


# ==========================
# OSMnx 그래프 관련 유틸
# ==========================
def _path_length_on_graph(G: nx.MultiDiGraph, path: List[int]) -> float:
    """그래프 노드 시퀀스 path의 실제 거리(m)."""
    dist = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        # MultiDiGraph 이므로 임의 edge 하나 사용
        edge_data = next(iter(data.values()))
        length = edge_data.get("length")
        if length is None:
            # length가 없으면 haversine 근사
            lat1 = G.nodes[u].get("y")
            lng1 = G.nodes[u].get("x")
            lat2 = G.nodes[v].get("y")
            lng2 = G.nodes[v].get("x")
            if None in (lat1, lng1, lat2, lng2):
                continue
            length = haversine(lat1, lng1, lat2, lng2)
        dist += float(length)
    return dist


def _apply_route_poison(G: nx.MultiDiGraph, base_path: List[int], poison_factor: float = 2.0) -> None:
    """
    base_path를 많이 지나가는 edge에 가중 페널티를 줘서,
    다음 shortest path는 다른 루트를 찾도록 유도.
    """
    for u, v in zip(base_path[:-1], base_path[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        for key, ed in data.items():
            length = ed.get("length", 1.0)
            ed["length"] = float(length) * poison_factor


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
    for nid in nodes:
        n = G.nodes[nid]
        lat = n.get("y")
        lng = n.get("x")
        if lat is None or lng is None:
            continue
        poly.append((float(lat), float(lng)))
    return poly


# ==========================
# fallback 사각형 루프 (OSM 실패 시)
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """
    OSM 그래프 생성 실패 시 사용하는 단순 사각형 루프.
    - 대략적인 거리만 맞추고, 카카오 지도에서 틀어지지 않도록 사각형 형태 유지.
    """
    target_m = max(200.0, km * 1000.0)
    # 한 변 길이 (정사각형 4변)
    side_m = target_m / 4.0

    # 위도/경도 당 meter 근사
    lat_m = 111_320.0
    lng_m = 111_320.0 * math.cos(math.radians(lat))

    d_lat = side_m / lat_m
    d_lng = side_m / lng_m

    p1 = (lat, lng)
    p2 = (lat, lng + d_lng)
    p3 = (lat + d_lat, lng + d_lng)
    p4 = (lat + d_lat, lng)
    poly = [p1, p2, p3, p4, p1]

    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# 메인 루프 생성 알고리즘
# ==========================
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    요청 좌표(lat, lng)와 목표 거리(km)를 기반으로
    '요청거리 정확도'와 '루프 모양'을 동시에 고려한 러닝 루프를 생성한다.

    - 1202_3_v2: poisoning 기반 rod + detour 루프 (모양이 좋음)
    - 251202  : 목표 거리 오차(±5%)를 강하게 제어 (길이 정확도 좋음)
    """
    start_time = time.time()
    target_m = max(200.0, km * 1000.0)

    # 스코어링 가중치
    ROUNDNESS_WEIGHT = 2.5
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3

    # 길이 허용 오차 (예: ±5%)
    LENGTH_TOL_FRAC = 0.05

    meta: Dict[str, Any] = {
        "target_m": target_m,
        "roundness_weight": ROUNDNESS_WEIGHT,
        "overlap_penalty": OVERLAP_PENALTY,
        "curve_penalty_weight": CURVE_PENALTY_WEIGHT,
        "length_tol_frac": LENGTH_TOL_FRAC,
        "sampled_rods": 0,
        "sampled_detours": 0,
        "routes_checked": 0,
        "routes_validated": 0,
        "fallback_used": False,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "time_s": None,
    }

    # --------------------------------------------------------
    # 1) OSMnx 보행자 그래프 생성 (캐시 활용)
    # --------------------------------------------------------
    try:
        # 그래프 + undirected 그래프를 캐시 기반으로 한 번에 획득
        G, undirected = _get_graph_and_undirected(lat, lng, km)
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
            message=f"OSM 보행자 그래프 생성/필터링 실패로 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    meta["graph_node_count"] = len(G.nodes)
    meta["graph_edge_count"] = len(G.edges)

    # --------------------------------------------------------
    # 2) 시작 노드 스냅
    # --------------------------------------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat) if ox is not None else None
        if start_node is None:
            # osmnx.distance를 못 쓰는 경우 fallback
            min_d = float("inf")
            best_n = None
            for nid, data in G.nodes(data=True):
                nlat = data.get("y")
                nlng = data.get("x")
                if nlat is None or nlng is None:
                    continue
                d = haversine(lat, lng, nlat, nlng)
                if d < min_d:
                    min_d = d
                    best_n = nid
            if best_n is None:
                raise RuntimeError("시작 노드를 찾지 못했습니다.")
            start_node = best_n
    except Exception:
        # 예외가 나도 자체 nearest 로 대체
        min_d = float("inf")
        best_n = None
        for nid, data in G.nodes(data=True):
            nlat = data.get("y")
            nlng = data.get("x")
            if nlat is None or nlng is None:
                continue
            d = haversine(lat, lng, nlat, nlng)
            if d < min_d:
                min_d = d
                best_n = nid
        if best_n is None:
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
                message="시작 노드 스냅에 실패하여 사각형 루프를 사용했습니다.",
            )
            meta["time_s"] = time.time() - start_time
            return safe_list(poly), safe_dict(meta)

        start_node = best_n

    # --------------------------------------------------------
    # 3) 기본 rod 경로(왕복 뼈대) 샘플링
    # --------------------------------------------------------
    # rod 후보 길이 범위: (target_m / 2)에 근접한 out-and-back
    rod_min = target_m * 0.45
    rod_max = target_m * 0.60

    # rod 후보 개수
    ROD_SAMPLES = 16

    # 무작위 목적지 후보 노드: start_node에서 일정 거리 떨어진 노드들
    all_nodes = list(undirected.nodes)
    random.shuffle(all_nodes)

    rod_candidates: List[Tuple[List[int], float]] = []  # (왕복 path, total_len)
    attempts = 0
    for nid in all_nodes:
        if nid == start_node:
            continue
        if attempts >= 200:
            break
        attempts += 1

        try:
            # undirected 기준 최단경로
            path = nx.shortest_path(
                undirected,
                source=start_node,
                target=nid,
                weight="length",
            )
        except Exception:
            continue

        one_way = _path_length_on_graph(undirected, path)
        total_len = one_way * 2.0
        if rod_min <= total_len <= rod_max:
            rod = path + path[-2::-1]  # 왕복
            rod_candidates.append((rod, total_len))
            meta["sampled_rods"] += 1
            if len(rod_candidates) >= ROD_SAMPLES:
                break

    if not rod_candidates:
        # rod 자체를 못 찾으면 fallback
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
            message="rod 후보를 찾지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # rod 후보를 길이와 간단한 random으로 섞어 다양화
    rod_candidates.sort(key=lambda x: x[1])
    random.shuffle(rod_candidates)

    # --------------------------------------------------------
    # 4) detour 결합하여 루프 다변화
    # --------------------------------------------------------
    BEST_LIMIT = 32  # 최종 후보 최대 개수
    best_routes: List[Tuple[float, Dict[str, Any], List[int], Polyline]] = []
    # (score, stats, node_path, polyline)

    for rod_path, rod_len in rod_candidates:
        # rod_path에서 중간 두 지점을 고르고, detour를 붙여 루프를 구성
        # detour는 길게 2~3km 내외, 또는 전체의 일부만 변형
        if len(rod_path) < 6:
            continue

        # rod_path 중간 index 대략적으로 골라서 detour 포인트
        mid_idx = len(rod_path) // 2
        # detour 시작/끝 index 후보
        possible_idxs = list(range(1, len(rod_path) - 1))
        random.shuffle(possible_idxs)

        detour_tries = 0
        for i_idx in possible_idxs:
            if detour_tries >= 4:
                break
            j_idx = None
            # mid_idx와 어느 정도 떨어진 지점 선택
            for cand in possible_idxs:
                if abs(cand - i_idx) >= 3:
                    j_idx = cand
                    break
            if j_idx is None:
                continue

            detour_tries += 1
            meta["sampled_detours"] += 1

            i_node = rod_path[i_idx]
            j_node = rod_path[j_idx]

            # detour는 기존 rod_path를 poison한 그래프에서 최단 경로로 찾는다.
            G_detour = undirected.copy()
            _apply_route_poison(G_detour, rod_path, poison_factor=3.0)

            try:
                detour = nx.shortest_path(
                    G_detour,
                    source=i_node,
                    target=j_node,
                    weight="length",
                )
            except Exception:
                continue

            # detour가 너무 짧거나 rod에서 거의 겹치면 의미 없음
            detour_len = _path_length_on_graph(G_detour, detour)
            if detour_len < 150.0:
                continue

            # detour를 포함한 전체 루프 노드 시퀀스 구성
            if i_idx < j_idx:
                prefix = rod_path[:i_idx]
                between = rod_path[i_idx:j_idx + 1]
                suffix = rod_path[j_idx + 1:]
            else:
                prefix = rod_path[:j_idx]
                between = rod_path[j_idx:i_idx + 1]
                suffix = rod_path[i_idx + 1:]

            # between을 detour로 대체
            # detour는 양끝 노드가 i_node, j_node 라고 가정
            # between[0] == i_node, between[-1] == j_node 이어야 함
            if not between or between[0] != i_node or between[-1] != j_node:
                continue

            new_path = prefix + detour + suffix

            # 루프가 아니면 마지막에 start_node로 닫아준다
            if new_path[0] != new_path[-1]:
                new_path = new_path + [new_path[0]]

            # 길이/스코어 계산
            total_len = _path_length_on_graph(undirected, new_path)
            err = abs(total_len - target_m)
            length_ok = (err <= target_m * LENGTH_TOL_FRAC)

            poly = _nodes_to_polyline(undirected, new_path)
            if len(poly) < 3:
                continue

            r = polygon_roundness(poly)
            overlap = _edge_overlap_fraction(new_path)
            curve_penalty = _curve_penalty(poly)

            # 스코어는 roundness↑, overlap↓, curve_penalty↓가 좋음
            score = (
                ROUNDNESS_WEIGHT * r
                - OVERLAP_PENALTY * overlap
                - CURVE_PENALTY_WEIGHT * curve_penalty
            )

            stats = {
                "len": total_len,
                "err": err,
                "length_ok": length_ok,
                "roundness": r,
                "overlap": overlap,
                "curve_penalty": curve_penalty,
                "score": score,
            }
            meta["routes_checked"] += 1

            best_routes.append((score, stats, new_path, poly))

    # rod+detour 기반 루프를 하나도 만들지 못했을 경우
    if not best_routes:
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
            message="rod+detour 루프를 만들지 못해 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 스코어 순으로 정렬 후 상위 BEST_LIMIT개만 유지
    best_routes.sort(key=lambda x: x[0], reverse=True)
    best_routes = best_routes[:BEST_LIMIT]

    # 길이 오차가 작고 스코어가 좋은 것을 우선
    best_routes.sort(
        key=lambda x: (
            x[1]["length_ok"] is False,  # length_ok인 것이 앞에 오도록
            x[1]["err"],                 # 오차 작을수록
            -x[0],                      # 스코어 클수록
        )
    )

    best_score, best_stats, best_node_path, best_poly = best_routes[0]
    meta["routes_validated"] = len(best_routes)

    success = bool(best_stats["length_ok"])
    used_fallback = not success

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
