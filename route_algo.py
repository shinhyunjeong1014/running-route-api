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

# ------------------------------------------------------------
# 상수 설정 (옵션 A: 초고속 모드)
# ------------------------------------------------------------
GRAPH_RADIUS_MIN = 500.0       # 최소 그래프 반경 (m)
GRAPH_RADIUS_MAX = 2000.0      # 최대 그래프 반경 (m)
GRAPH_RADIUS_FACTOR = 0.35     # target_m * factor 로 반경 설정

VIA_MIN_FACTOR = 0.35          # via 최소 거리 비율 (target_m * 0.35)
VIA_MAX_FACTOR = 0.70          # via 최대 거리 비율 (target_m * 0.70)
MAX_VIA_CANDIDATES = 4         # via 후보 최대 개수
LENGTH_TOLERANCE_M = 45.0      # 요청 거리 허용 오차 (±45m)

MIN_LOOP_M = 200.0             # 최소 루프 길이 (fallback용)


# ------------------------------------------------------------
# 유틸: 거리 및 polyline 길이
# ------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """위경도 거리(m)"""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def polyline_length_m(poly: Polyline) -> float:
    """단순 polyline 길이(m) - app.py에서 유효성 체크용"""
    if not poly or len(poly) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(poly[:-1], poly[1:]):
        total += haversine_m(la1, lo1, la2, lo2)
    return total


def _edge_length(G: nx.Graph, u: int, v: int) -> float:
    """osmnx MultiGraph/Graph에서 edge length 추출 (가장 짧은 것)"""
    data = G.get_edge_data(u, v)
    if data is None:
        return 0.0

    # 단순 Graph: {'length': ...}
    if "length" in data:
        return float(data.get("length", 0.0))

    # MultiGraph: {key: {'length': ...}, ...}
    min_len = None
    for _, attr in data.items():
        length = float(attr.get("length", 0.0))
        if min_len is None or length < min_len:
            min_len = length
    return float(min_len or 0.0)


def _path_length_graph(G: nx.Graph, path: List[int]) -> float:
    """그래프 기반의 실제 경로 길이 (엣지 length 합)"""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += _edge_length(G, u, v)
    return total


def _path_to_polyline(G: nx.Graph, path: List[int]) -> Polyline:
    """노드 경로 → (lat, lng) 리스트"""
    poly: Polyline = []
    for nid in path:
        data = G.nodes[nid]
        lat = float(data.get("y"))
        lng = float(data.get("x"))
        poly.append((lat, lng))
    return poly


# ------------------------------------------------------------
# fallback: 단순 정사각형 루프
# ------------------------------------------------------------
def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """OSM 그래프 실패 시 사용하는 단순 정사각형 루프"""
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
    length_m = polyline_length_m(poly)
    err = abs(length_m - target_m)
    return poly, length_m, err


# ------------------------------------------------------------
# OSM 그래프 빌드 (초간단, 초고속 버전)
# ------------------------------------------------------------
def _build_osm_graph(lat: float, lng: float, target_m: float) -> Tuple[nx.Graph, int]:
    """osmnx 보행자 그래프 로딩 및 시작 노드 계산"""
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    radius = target_m * GRAPH_RADIUS_FACTOR
    radius = max(GRAPH_RADIUS_MIN, min(GRAPH_RADIUS_MAX, radius))

    # MultiDiGraph 로딩
    G_raw = ox.graph_from_point(
        (lat, lng),
        dist=radius,
        network_type="walk",
        simplify=True,
    )

    # undirected + edge length 보장
    G_ud = ox.utils_graph.get_undirected(G_raw)
    G_ud = ox.distance.add_edge_lengths(G_ud)

    # MultiGraph 그대로 사용 (길이 계산만 직접)
    start_node = ox.distance.nearest_nodes(G_ud, lng, lat)

    return G_ud, start_node


# ------------------------------------------------------------
# via 후보 선택 (초고속 단순 버전)
# ------------------------------------------------------------
def _select_via_candidates(
    G: nx.Graph,
    center_lat: float,
    center_lng: float,
    target_m: float,
) -> List[int]:
    """
    시작점에서 일정 거리 범위 안의 노드를 via 후보로 선택.
    - 거리 범위: [target_m * VIA_MIN_FACTOR, target_m * VIA_MAX_FACTOR]
    - 최대 MAX_VIA_CANDIDATES개 랜덤 선택
    """
    min_r = target_m * VIA_MIN_FACTOR
    max_r = target_m * VIA_MAX_FACTOR

    candidates: List[Tuple[int, float]] = []
    for nid, data in G.nodes(data=True):
        lat = data.get("y")
        lng = data.get("x")
        if lat is None or lng is None:
            continue
        d = haversine_m(center_lat, center_lng, float(lat), float(lng))
        if min_r <= d <= max_r:
            candidates.append((nid, d))

    if not candidates:
        return []

    # 거리가 비슷한 후보 중에서 최대 MAX_VIA_CANDIDATES 개만 랜덤 사용
    # (속도 최적화: 너무 많은 후보는 필요 없음)
    random.shuffle(candidates)
    candidates = candidates[:MAX_VIA_CANDIDATES]

    return [nid for nid, _ in candidates]


# ------------------------------------------------------------
# 메인: 초고속 러닝 루프 생성
# ------------------------------------------------------------
def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    초고속 러닝 루프 생성 (옵션 A)
    - 보행자 네트워크 기반
    - start → via → start 형태의 단순 루프
    - 시간 목표: time_s ≈ 4초대 이하
    - 거리 오차: ±45m 이내면 success
    """
    t0 = time.time()
    target_m = max(MIN_LOOP_M, km * 1000.0)

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
        "via_pairs": [],
        "message": "",
    }

    # --------------------------------------------------------
    # 1) OSM 그래프 로딩
    # --------------------------------------------------------
    try:
        G, start_node = _build_osm_graph(lat, lng, target_m)
    except Exception as e:
        # 그래프 로딩 실패 → 정사각형 fallback
        poly, length_m, err = _fallback_square_loop(lat, lng, km)
        meta.update(
            status="fallback",
            len=float(length_m),
            err=float(err),
            success=bool(err <= LENGTH_TOLERANCE_M),
            length_ok=bool(err <= LENGTH_TOLERANCE_M),
            used_fallback=True,
            message=f"OSM 그래프 생성에 실패하여 단순 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = float(time.time() - t0)
        return poly, meta

    # --------------------------------------------------------
    # 2) via 후보 선택
    # --------------------------------------------------------
    via_candidates = _select_via_candidates(G, lat, lng, target_m)
    meta["via_candidates"] = via_candidates

    if not via_candidates:
        # via 없으면: 단순 out-and-back fallback
        try:
            poly, length_m = _simple_out_and_back(G, start_node, target_m)
            err = abs(length_m - target_m)
            meta.update(
                status="approx",
                len=float(length_m),
                err=float(err),
                success=bool(err <= LENGTH_TOLERANCE_M),
                length_ok=bool(err <= LENGTH_TOLERANCE_M),
                used_fallback=True,
                message="via 후보를 찾지 못해 단순 왕복(out-and-back) 경로를 사용했습니다.",
            )
            meta["time_s"] = float(time.time() - t0)
            return poly, meta
        except Exception as e:
            poly, length_m, err = _fallback_square_loop(lat, lng, km)
            meta.update(
                status="fallback",
                len=float(length_m),
                err=float(err),
                success=bool(err <= LENGTH_TOLERANCE_M),
                length_ok=bool(err <= LENGTH_TOLERANCE_M),
                used_fallback=True,
                message=f"경로 탐색 실패로 사각형 루프를 사용했습니다: {e}",
            )
            meta["time_s"] = float(time.time() - t0)
            return poly, meta

    # --------------------------------------------------------
    # 3) 각 via에 대해 start→via→start 루프 생성
    # --------------------------------------------------------
    best_path: Optional[List[int]] = None
    best_len: float = 0.0
    best_err: float = float("inf")
    via_pairs: List[Tuple[int, int]] = []

    for v in via_candidates:
        meta["routes_checked"] += 1
        try:
            path1 = nx.shortest_path(G, start_node, v, weight="length")
            path2 = nx.shortest_path(G, v, start_node, weight="length")
        except Exception:
            continue

        full = path1 + path2[1:]
        L = _path_length_graph(G, full)
        err = abs(L - target_m)

        meta["routes_validated"] += 1
        via_pairs.append((start_node, v))

        if err < best_err:
            best_err = err
            best_len = L
            best_path = full

            # 아주 근접하면 바로 종료해 속도 확보
            if best_err <= LENGTH_TOLERANCE_M / 2.0:
                break

    meta["via_pairs"] = via_pairs

    if best_path is None:
        # 모든 via 실패 → fallback
        try:
            poly, length_m = _simple_out_and_back(G, start_node, target_m)
            err = abs(length_m - target_m)
            meta.update(
                status="approx",
                len=float(length_m),
                err=float(err),
                success=bool(err <= LENGTH_TOLERANCE_M),
                length_ok=bool(err <= LENGTH_TOLERANCE_M),
                used_fallback=True,
                message="via 경로를 찾지 못해 단순 왕복(out-and-back) 경로를 사용했습니다.",
            )
            meta["time_s"] = float(time.time() - t0)
            return poly, meta
        except Exception as e:
            poly, length_m, err = _fallback_square_loop(lat, lng, km)
            meta.update(
                status="fallback",
                len=float(length_m),
                err=float(err),
                success=bool(err <= LENGTH_TOLERANCE_M),
                length_ok=bool(err <= LENGTH_TOLERANCE_M),
                used_fallback=True,
                message=f"경로 탐색 실패로 사각형 루프를 사용했습니다: {e}",
            )
            meta["time_s"] = float(time.time() - t0)
            return poly, meta

    # --------------------------------------------------------
    # 4) 최종 경로 결과 구성
    # --------------------------------------------------------
    polyline = _path_to_polyline(G, best_path)
    err = best_err
    success = bool(err <= LENGTH_TOLERANCE_M)

    meta.update(
        status="ok" if success else "approx",
        len=float(best_len),
        err=float(err),
        success=success,
        length_ok=success,
        used_fallback=False,
        message=(
            "요청 거리를 고정밀로 만족하는 초고속 루프 경로가 생성되었습니다."
            if success
            else f"요청 오차(±{int(LENGTH_TOLERANCE_M)}m)를 약간 초과했지만, "
                 f"가장 근접한 초고속 루프 경로를 반환합니다."
        ),
    )
    meta["time_s"] = float(time.time() - t0)
    return polyline, meta


# ------------------------------------------------------------
# 단순 out-and-back 경로 (fallback용)
# ------------------------------------------------------------
def _simple_out_and_back(
    G: nx.Graph,
    start_node: int,
    target_m: float,
) -> Tuple[Polyline, float]:
    """
    start에서 적당히 떨어진 노드까지 shortest_path로 가서,
    그대로 되돌아오는 단순 왕복 경로.
    - 그래프가 좁거나 via가 없을 때 사용.
    """
    # 가장 멀리 있는 후보 노드를 하나 선택
    nodes = list(G.nodes)
    if len(nodes) < 2:
        raise RuntimeError("그래프 노드 수가 너무 적습니다.")

    # 거리 기반으로 몇 개만 샘플링해서 가장 먼 노드 선택
    random.shuffle(nodes)
    sample = nodes[: min(30, len(nodes))]

    best_node = None
    best_dist = 0.0

    sy = G.nodes[start_node]["y"]
    sx = G.nodes[start_node]["x"]

    for n in sample:
        ny = G.nodes[n]["y"]
        nx_ = G.nodes[n]["x"]
        d = haversine_m(sy, sx, float(ny), float(nx_))
        if d > best_dist:
            best_dist = d
            best_node = n

    if best_node is None or best_node == start_node:
        raise RuntimeError("왕복 경로 생성에 실패했습니다.")

    # start → best_node → start
    path1 = nx.shortest_path(G, start_node, best_node, weight="length")
    path2 = list(reversed(path1))
    full = path1 + path2[1:]
    L = _path_length_graph(G, full)
    poly = _path_to_polyline(G, full)
    return poly, L