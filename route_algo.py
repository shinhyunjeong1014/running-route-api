from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import osmnx as ox

# 타입 별칭
LatLng = Tuple[float, float]
Polyline = List[LatLng]

# ----------------------------
# 기본 설정값
# ----------------------------

EARTH_RADIUS_M = 6371000.0

# 요청 거리 허용 오차 (±5%)
LENGTH_TOL_FRAC = 0.05

# 루프 생성 시 사용할 후보 endpoint 개수
ENDPOINT_CANDIDATES = 80

# endpoint 를 고를 때, 시작점에서의 "대략적인" 직선거리 범위
ENDPOINT_MIN_FRAC = 0.25   # target_m * 0.25
ENDPOINT_MAX_FRAC = 0.65   # target_m * 0.65

# U턴 / 겹침 관련 설정
LOCAL_UTURN_DIST_M = 25.0         # 사용자 요구: 25m 이내에서의 급격한 방향 전환은 보기 싫음
LOCAL_UTURN_ANGLE_DEG = 160.0     # 180도에 가까우면 U턴에 가깝다고 판단
REPEATED_EDGE_PENALTY_POWER = 2.0 # 동일 edge 재사용 비율을 더 강하게 패널티

# 점수 가중치 (길이 우선, 그 다음 형태)
W_LEN = 14.0       # 길이 오차 패널티
W_ROUND = 4.0      # roundness 보너스
W_OVERLAP = 8.0    # 동일/역방향 edge 재사용(겹침) 패널티
W_UTURN = 10.0     # 근거리 U턴 패널티

RNG = random.Random(42)


# ----------------------------
# 유틸리티: 거리 / 각도
# ----------------------------

def haversine_m(a: LatLng, b: LatLng) -> float:
    """두 위경도 좌표 사이의 거리(m)를 계산."""
    lat1, lon1 = a
    lat2, lon2 = b
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    s = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(s))


def polyline_length_m(polyline: Polyline) -> float:
    """폴리라인 전체 길이(m). app.py 에서도 사용하므로 반드시 유지."""
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for p, q in zip(polyline, polyline[1:]):
        total += haversine_m(p, q)
    return total


# ----------------------------
# OSM 그래프 생성
# ----------------------------

def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    보행자 전용 그래프 생성.
    - network_type='walk' 를 사용해 자동차 전용도로를 피함
    - highway 태그 기준으로 한 번 더 필터링
    """
    # 러닝 거리의 1.2배 정도 반경 안에서 그래프 생성
    # (루프 구조를 만들기 위해 어느 정도 여유를 둠)
    approx_radius = max(500.0, km * 700.0)

    G: nx.MultiDiGraph = ox.graph_from_point(
        (lat, lng),
        dist=approx_radius,
        network_type="walk",
        simplify=True,
        retain_all=True,
    )

    # walk 네트워크여도 간혹 자동차 중심 edge 가 포함되므로, highway 기반으로 한 번 더 필터
    keep_highways = {
        "footway",
        "path",
        "steps",
        "pedestrian",
        "living_street",
        "residential",
        "service",
        "track",
        "cycleway",
        "alley",
        "sidewalk",
    }

    edges_to_remove = []
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get("highway")
        if hw is None:
            continue
        if isinstance(hw, (list, tuple, set)):
            hw_set = set(hw)
        else:
            hw_set = {hw}
        if not (hw_set & keep_highways):
            edges_to_remove.append((u, v, k))
    for u, v, k in edges_to_remove:
        G.remove_edge(u, v, k)

    # 모든 edge 에 length 가 없으면 직접 계산
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" not in data:
            lat1 = G.nodes[u]["y"]
            lon1 = G.nodes[u]["x"]
            lat2 = G.nodes[v]["y"]
            lon2 = G.nodes[v]["x"]
            data["length"] = haversine_m((lat1, lon1), (lat2, lon2))

    return G


def _nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int:
    """그래프에서 시작점과 가장 가까운 노드 찾기."""
    # osmnx 2.x 에서는 distance.nearest_nodes 사용
    try:
        return ox.distance.nearest_nodes(G, lng, lat)
    except Exception:
        # 호환용: 수동 탐색
        best = None
        best_d = float("inf")
        for nid, data in G.nodes(data=True):
            d = haversine_m((lat, lng), (data["y"], data["x"]))
            if d < best_d:
                best_d = d
                best = nid
        if best is None:
            raise RuntimeError("그래프에 노드가 없습니다.")
        return best


# ----------------------------
# 경로 -> 좌표 변환 및 형태 분석
# ----------------------------

def _path_to_coords(G: nx.MultiDiGraph, path: Sequence[int]) -> Polyline:
    """노드 id 리스트를 위경도 좌표 리스트로 변환."""
    coords: Polyline = []
    for nid in path:
        node = G.nodes[nid]
        coords.append((float(node["y"]), float(node["x"])))
    return coords


def _compute_roundness(coords: Polyline) -> float:
    """
    형상 roundness 측정.
    4πA / P² (0~1 사이, 1에 가까울수록 원형에 가까움)
    coords 는 폐곡선 기준 (시작 ≈ 끝)
    """
    if len(coords) < 4:
        return 0.0

    # 위경도를 로컬 직교 좌표계로 투영
    lats = [p[0] for p in coords]
    lngs = [p[1] for p in coords]
    lat0 = math.radians(sum(lats) / len(lats))
    lon0 = math.radians(sum(lngs) / len(lngs))

    xy: List[Tuple[float, float]] = []
    for lat, lon in coords:
        phi = math.radians(lat)
        lam = math.radians(lon)
        x = EARTH_RADIUS_M * (lam - lon0) * math.cos(lat0)
        y = EARTH_RADIUS_M * (phi - lat0)
        xy.append((x, y))

    # shoelace 로 면적 계산
    area = 0.0
    perim = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        area += x1 * y2 - x2 * y1
        dx = x2 - x1
        dy = y2 - y1
        perim += math.hypot(dx, dy)

    area = abs(area) * 0.5
    if perim <= 1e-6 or area <= 1e-6:
        return 0.0

    return max(0.0, min(1.0, 4.0 * math.pi * area / (perim ** 2)))


def _edge_overlap_ratio(path: Sequence[int]) -> float:
    """
    동일 edge (역방향 포함)를 여러 번 사용하는 정도.
    - 0 이면 한 번씩만 사용된 셈 → 겹침 없음
    - 1 에 가까울수록 동일 길을 많이 왔다갔다 한 것
    """
    if len(path) < 3:
        return 0.0

    norm_edges: List[Tuple[int, int]] = []
    for u, v in zip(path, path[1:]):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        norm_edges.append((a, b))

    if not norm_edges:
        return 0.0

    from collections import Counter

    cnt = Counter(norm_edges)
    total_edges = len(norm_edges)
    repeated_edges = sum(c - 1 for c in cnt.values() if c > 1)
    return repeated_edges / total_edges


def _local_uturn_ratio(coords: Polyline) -> float:
    """
    LOCAL_UTURN_DIST_M 안에서 발생하는 급격한 방향 전환 비율.
    - 0 이면 근거리 U턴 거의 없음
    """
    if len(coords) < 3:
        return 0.0

    ucount = 0
    candidates = 0

    for i in range(1, len(coords) - 1):
        p0 = coords[i - 1]
        p1 = coords[i]
        p2 = coords[i + 1]

        d01 = haversine_m(p0, p1)
        d12 = haversine_m(p1, p2)
        if d01 > LOCAL_UTURN_DIST_M or d12 > LOCAL_UTURN_DIST_M:
            continue

        # 벡터 각도 계산 (위경도 그대로 사용해도 근거리에서는 문제 없음)
        v1x = p1[1] - p0[1]
        v1y = p1[0] - p0[0]
        v2x = p2[1] - p1[1]
        v2y = p2[0] - p1[0]

        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-9 or n2 < 1e-9:
            continue

        dot = v1x * v2x + v1y * v2y
        cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
        angle_deg = math.degrees(math.acos(cos_theta))

        # 180도에 가까우면 U턴에 가깝다고 본다
        if angle_deg >= LOCAL_UTURN_ANGLE_DEG:
            ucount += 1
        candidates += 1

    if candidates == 0:
        return 0.0
    return ucount / candidates


# ----------------------------
# endpoint 후보 선택 및 경로 계산
# ----------------------------

def _endpoint_candidates(
    G: nx.MultiDiGraph, lat: float, lng: float, target_m: float, count: int
) -> List[int]:
    """시작점 기준으로 적당한 거리(직선 기준)에 있는 노드들을 endpoint 후보로 선택."""
    nodes: List[int] = list(G.nodes)
    coords = {nid: (G.nodes[nid]["y"], G.nodes[nid]["x"]) for nid in nodes}

    min_d = target_m * ENDPOINT_MIN_FRAC
    max_d = target_m * ENDPOINT_MAX_FRAC

    cands: List[int] = []
    for nid in nodes:
        d = haversine_m((lat, lng), coords[nid])
        if min_d <= d <= max_d:
            cands.append(nid)

    if len(cands) == 0:
        # 너무 빡셌다면 조건을 완화해서라도 조금은 뽑는다
        for nid in nodes:
            d = haversine_m((lat, lng), coords[nid])
            if target_m * 0.1 <= d <= target_m * 0.9:
                cands.append(nid)

    if len(cands) == 0:
        return []

    RNG.shuffle(cands)
    return cands[:count]


def _shortest_path_safe(
    G: nx.MultiDiGraph, src: int, dst: int, weight: str = "length"
) -> Optional[List[int]]:
    try:
        return nx.shortest_path(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return None
    except nx.NetworkXError:
        return None


def _second_path_detour(
    G: nx.MultiDiGraph, src: int, dst: int, primary_path: Sequence[int], weight: str = "length"
) -> Optional[List[int]]:
    """
    primary_path 에 사용된 edge 에 큰 penalty 를 줘서,
    같은 길을 덜 쓰는 "다른" 경로를 찾으려는 근사적인 2nd path.
    """
    if not primary_path or len(primary_path) < 2:
        return None

    # MultiDiGraph 를 얕게 복사해서 weight 를 수정
    H: nx.MultiDiGraph = G.copy()
    penalty_factor = 10.0

    for u, v in zip(primary_path, primary_path[1:]):
        if H.has_edge(u, v):
            for k, data in H[u][v].items():
                base = float(data.get(weight, 1.0))
                data[weight] = base * penalty_factor
        if H.has_edge(v, u):
            for k, data in H[v][u].items():
                base = float(data.get(weight, 1.0))
                data[weight] = base * penalty_factor

    return _shortest_path_safe(H, src, dst, weight=weight)


# ----------------------------
# 루프 스코어링
# ----------------------------

@dataclass
class LoopMetrics:
    length_m: float
    err_m: float
    roundness: float
    overlap: float
    uturn_ratio: float
    score: float
    length_ok: bool


def _evaluate_loop(
    path: Sequence[int],
    coords: Polyline,
    target_m: float,
) -> LoopMetrics:
    # 루프 길이
    length_m = polyline_length_m(coords)
    err_m = abs(length_m - target_m)
    length_ok = err_m <= target_m * LENGTH_TOL_FRAC

    # 형상 관련 지표
    # coords 가 폐곡선이 아닐 수 있으므로, 임시로 닫아서 roundness 계산
    closed_coords = coords
    if coords[0] != coords[-1]:
        closed_coords = coords + [coords[0]]

    roundness = _compute_roundness(closed_coords)
    overlap = _edge_overlap_ratio(path)
    uturn_ratio = _local_uturn_ratio(coords)

    # 점수 계산
    norm_err = err_m / max(target_m, 1.0)
    # 길이 오차는 제곱으로 더 강하게 페널티
    len_penalty = norm_err ** 2

    overlap_term = overlap ** REPEATED_EDGE_PENALTY_POWER

    score = (
        + W_ROUND * roundness
        - W_LEN * len_penalty
        - W_OVERLAP * overlap_term
        - W_UTURN * uturn_ratio
    )

    return LoopMetrics(
        length_m=length_m,
        err_m=err_m,
        roundness=roundness,
        overlap=overlap,
        uturn_ratio=uturn_ratio,
        score=score,
        length_ok=length_ok,
    )


# ----------------------------
# Fallback: 기하학적 사각형 루프
# ----------------------------

def _fallback_square_loop(lat: float, lng: float, km: float) -> Polyline:
    """
    네트워크 탐색에 완전히 실패했을 때만 사용하는 사각형 루프.
    - 사용 빈도는 매우 낮아야 함.
    """
    target_m = km * 1000.0
    # 한 변의 길이를 target 의 1/4 로 설정 (정사각형 둘레 = 4a)
    side = target_m / 4.0

    # 위경도에서 대략적인 변환 (소규모 거리)
    dlat = (side / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlng = dlat / math.cos(math.radians(lat))

    p1 = (lat, lng)
    p2 = (lat + dlat, lng)
    p3 = (lat + dlat, lng + dlng)
    p4 = (lat, lng + dlng)
    return [p1, p2, p3, p4, p1]


# ----------------------------
# 메인 엔트리포인트
# ----------------------------

def generate_area_loop(lat: float, lng: float, km: float):
    """
    보행자 전용 러닝 루프 생성기 (품질 우선 버전)

    - OSMnx walk 그래프 기반
    - start 에서 적당히 떨어진 endpoint 후보들을 선정
    - start → endpoint 최단경로 (rod)
    - endpoint → start detour 경로 (rod 에 penalty를 줘서 다른 길을 선호)
    - 두 경로를 이어서 폐곡선 루프 생성
    - 길이 오차(±5%)를 강하게 벌점, 그 안에서는 roundness ↑, overlap ↓, U턴 ↓ 를 우선
    """
    start_time = time.time()
    target_m = km * 1000.0

    meta: Dict[str, Any] = {
        "len": None,
        "err": None,
        "roundness": None,
        "overlap": None,
        "curve_penalty": None,   # 여기서는 근거리 U턴 비율
        "score": None,
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

    try:
        G = _build_pedestrian_graph(lat, lng, km)
        if G.number_of_nodes() == 0:
            raise RuntimeError("보행자 그래프 노드가 없습니다.")

        start_node = _nearest_node(G, lat, lng)

        # endpoint 후보 뽑기
        endpoints = _endpoint_candidates(G, lat, lng, target_m, ENDPOINT_CANDIDATES)
        if not endpoints:
            raise RuntimeError("endpoint 후보를 찾지 못했습니다.")

        best_path: Optional[List[int]] = None
        best_coords: Optional[Polyline] = None
        best_metrics: Optional[LoopMetrics] = None
        best_score = -1e18

        routes_checked = 0
        routes_validated = 0

        for end_node in endpoints:
            routes_checked += 1

            # 1) start → endpoint
            path1 = _shortest_path_safe(G, start_node, end_node, weight="length")
            if not path1 or len(path1) < 2:
                continue

            # 너무 짧으면 루프 길이가 부족해질 수 있으므로 skip
            coords1 = _path_to_coords(G, path1)
            len1 = polyline_length_m(coords1)
            if len1 < target_m * 0.25:
                continue

            # 2) endpoint → start (rod 를 피하도록 penalty를 준 그래프에서 탐색)
            path2 = _second_path_detour(G, end_node, start_node, primary_path=path1)
            if not path2 or len(path2) < 2:
                continue

            full_path = list(path1) + list(path2[1:])  # endpoint 중복 제거
            if len(full_path) < 4:
                continue

            coords = _path_to_coords(G, full_path)

            # 메트릭 계산
            metrics = _evaluate_loop(full_path, coords, target_m)
            routes_validated += 1

            # 길이 범위를 아주 크게 벗어나면 고려 X
            if metrics.length_m < target_m * 0.6 or metrics.length_m > target_m * 1.5:
                continue

            # 점수 기반 최적 루프 갱신
            if metrics.score > best_score:
                best_score = metrics.score
                best_metrics = metrics
                best_path = full_path
                best_coords = coords

        meta["routes_checked"] = routes_checked
        meta["routes_validated"] = routes_validated

        if best_coords is None or best_metrics is None:
            # 완전히 실패하면 fallback
            polyline = _fallback_square_loop(lat, lng, km)
            length_m = polyline_length_m(polyline)
            meta.update(
                len=length_m,
                err=abs(length_m - target_m),
                roundness=0.0,
                overlap=0.0,
                curve_penalty=0.0,
                score=0.0,
                success=False,
                length_ok=abs(length_m - target_m) <= target_m * LENGTH_TOL_FRAC,
                used_fallback=True,
                message="요청 거리의 ±5% 이내에 해당하는 보행 루프를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
            )
            meta["time_s"] = time.time() - start_time
            return polyline, meta

        # 최적 루프가 존재하는 경우
        # 실제 시작 좌표(lat, lng)를 폴리라인 앞뒤에 붙여서 사용자 시각적 기준을 맞춰준다.
        polyline: Polyline = []
        polyline.append((lat, lng))
        polyline.extend(best_coords)
        if best_coords[-1] != (lat, lng):
            polyline.append((lat, lng))

        length_m = polyline_length_m(polyline)
        err_m = abs(length_m - target_m)
        length_ok = err_m <= target_m * LENGTH_TOL_FRAC

        meta.update(
            len=length_m,
            err=err_m,
            roundness=best_metrics.roundness,
            overlap=best_metrics.overlap,
            curve_penalty=best_metrics.uturn_ratio,
            score=best_metrics.score,
            success=True,
            length_ok=length_ok,
            used_fallback=False,
            message=(
                "요청 거리 대비 오차가 있지만, 형태와 보행 적합성을 고려해 최적 루프를 선택했습니다."
                if not length_ok
                else "길이와 형태를 모두 고려한 최적의 보행 루프를 생성했습니다."
            ),
        )
        meta["time_s"] = time.time() - start_time
        return polyline, meta

    except Exception as e:
        # 예외 상황에서는 fallback 사각형을 사용
        polyline = _fallback_square_loop(lat, lng, km)
        length_m = polyline_length_m(polyline)
        meta.update(
            len=length_m,
            err=abs(length_m - target_m),
            roundness=0.0,
            overlap=0.0,
            curve_penalty=0.0,
            score=0.0,
            success=False,
            length_ok=abs(length_m - target_m) <= target_m * LENGTH_TOL_FRAC,
            used_fallback=True,
            message=f"루트 생성 중 오류 발생: {e}. 기하학적 사각형 루프로 대체했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return polyline, meta
