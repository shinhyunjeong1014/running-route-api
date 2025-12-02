from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx

try:
    import osmnx as ox
except Exception:  # 배포 환경에서 import 실패 대비
    ox = None

LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ============================================================
# 1. 기본 유틸 함수들
# ============================================================

def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    위도/경도 두 점 사이의 거리를 미터 단위로 계산.
    """
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def polyline_length_m(polyline: Polyline) -> float:
    """
    polyline의 총 길이(미터).
    app.py에서 불러 사용하는 함수이므로 반드시 유지.
    """
    if not polyline or len(polyline) < 2:
        return 0.0
    length = 0.0
    for (lat1, lng1), (lat2, lng2) in zip(polyline[:-1], polyline[1:]):
        length += haversine(lat1, lng1, lat2, lng2)
    return float(length)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_list(xs: Any) -> List:
    if isinstance(xs, list):
        return xs
    return list(xs) if xs is not None else []


def safe_dict(d: Any) -> Dict:
    if isinstance(d, dict):
        return d
    return dict(d) if d is not None else {}


# ------------------------------------------------------------
# 좌표 변환 / 기하 유틸
# ------------------------------------------------------------

def _to_local_xy(lat: float, lng: float, center_lat: float, center_lng: float) -> Tuple[float, float]:
    """
    위경도를 중심점 기준 로컬 x,y로 변환 (m 단위 근사).
    """
    dx = haversine(center_lat, center_lng, center_lat, lng)
    dy = haversine(center_lat, center_lng, lat, center_lng)
    if lng < center_lng:
        dx = -dx
    if lat < center_lat:
        dy = -dy
    return dx, dy


def polygon_roundness(poly: Polyline) -> float:
    """
    다각형의 '둥근 정도'를 0~1 사이 값으로 근사.
    1에 가까울수록 타원/원에 가깝고, 0에 가까우면 가늘고 긴 형태.
    """
    if len(poly) < 3:
        return 0.0

    xs: List[float] = []
    ys: List[float] = []
    for lat, lng in poly:
        xs.append(lat)
        ys.append(lng)

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    pts_xy: List[Tuple[float, float]] = []
    for lat, lng in poly:
        x, y = _to_local_xy(lat, lng, cx, cy)
        pts_xy.append((x, y))

    rs = [math.hypot(x, y) for x, y in pts_xy]
    if not rs:
        return 0.0

    r_mean = sum(rs) / len(rs)
    if r_mean <= 1e-6:
        return 0.0

    var = sum((r - r_mean) ** 2 for r in rs) / len(rs)
    std = math.sqrt(var)
    # 표준편차가 작을수록 원에 가까우므로 roundness를 높게
    roundness = max(0.0, 1.0 - std / (r_mean + 1e-6))
    return float(roundness)


def _edge_overlap_fraction(poly: Polyline) -> float:
    """
    경로 겹침 정도(0~1). 재방문하는 구간이 많을수록 값 증가.
    """
    if len(poly) < 3:
        return 0.0

    seen = {}
    total = 0
    overlap = 0

    def quantize(x: float) -> int:
        return int(round(x * 10))  # 0.1m 정도 단위

    for (lat1, lng1), (lat2, lng2) in zip(poly[:-1], poly[1:]):
        seg_len = haversine(lat1, lng1, lat2, lng2)
        steps = max(1, int(seg_len / 3.0))  # 3m 간격 정도로 샘플링
        for i in range(steps + 1):
            t = i / max(1, steps)
            lat = lat1 + (lat2 - lat1) * t
            lng = lng1 + (lng2 - lng1) * t
            key = (quantize(lat), quantize(lng))
            total += 1
            if key in seen:
                overlap += 1
            else:
                seen[key] = True

    if total == 0:
        return 0.0
    return float(overlap / total)


def _curve_penalty(poly: Polyline) -> float:
    """
    급격한 커브(꺾이는 각도)에 대한 패널티.
    값이 클수록 '꺾이는' 지점이 많다는 뜻.
    """
    if len(poly) < 3:
        return 0.0

    penalty = 0.0
    for (lat0, lng0), (lat1, lng1), (lat2, lng2) in zip(poly[:-2], poly[1:-1], poly[2:]):
        x0, y0 = _to_local_xy(lat0, lng0, lat1, lng1)
        x2, y2 = _to_local_xy(lat2, lng2, lat1, lng1)
        dot = x0 * x2 + y0 * y2
        norm1 = math.hypot(x0, y0)
        norm2 = math.hypot(x2, y2)
        if norm1 < 1e-3 or norm2 < 1e-3:
            continue
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        angle = math.degrees(math.acos(cos_theta))
        # 180º 직진일수록 패널티 0, 90º 꺾이면 패널티↑
        penalty += max(0.0, (180.0 - angle) / 90.0)

    return float(penalty / (len(poly) - 2))


def _graph_path_length(G: nx.Graph, path: List[int]) -> float:
    """
    그래프 상의 path 길이를 edge 'length' 속성 기준으로 계산.
    MultiGraph/Graph 모두 안전하게 처리.
    """
    if not path or len(path) < 2:
        return 0.0

    length = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G[u][v]

        # MultiGraph: {key: {attr...}, ...}
        # Graph: {attr_key: attr_val, ...}
        if isinstance(data, dict) and data:
            # MultiGraph 인지 검사: 값이 dict면 multi-edge 구조
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict):
                # 여러 edge 중 length가 가장 짧은 것 사용
                best_attr = min(data.values(), key=lambda x: x.get("length", 1.0))
                length += float(best_attr.get("length", 0.0))
            else:
                # 일반 Graph: data 자체가 attr dict
                length += float(data.get("length", 0.0))
        else:
            # 예외적인 구조는 0으로
            length += 0.0

    return float(length)


# ============================================================
# 2. Fallback 사각형 루프 (요청 거리 근사)
# ============================================================

def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """
    네트워크 기반 루프를 만들지 못했을 때 사용하는 기하학적 사각형 루프.
    중심은 요청 좌표, 한 변 길이는 대략 요청 거리의 1/4.
    """
    target_m = km * 1000.0
    side_m = target_m / 4.0  # 4변 합치면 대략 target_m

    # 대략적인 위도/경도 변환 (대략적인 값)
    delta_deg_lat = side_m / 111_000.0
    delta_deg_lng = side_m / (111_000.0 * math.cos(math.radians(lat)) + 1e-6)

    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]

    # 중심 재보정
    center_lat = (a[0] + c[0]) / 2.0
    center_lng = (b[1] + d[1]) / 2.0
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]

    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ============================================================
# 3. OSMnx 보행자 그래프 구성
# ============================================================

def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    OSMnx 'walk' 네트워크 + 보행자 친화 도로 필터.
    자동차 전용 도로는 제외하고, 보행 가능한 도로 유형만 남긴다.
    """
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    # API 부하를 줄이면서도 충분한 범위를 덮도록 반경 설정
    radius_m = max(700.0, km * 500.0 + 700.0)

    custom_filter = (
        '["highway"~"footway|path|sidewalk|cycleway|steps|pedestrian|track|service|'
        'residential|living_street|unclassified|tertiary|secondary|alley"]'
    )

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk",
        custom_filter=custom_filter,
        simplify=True,
        retain_all=False,
    )

    if not G.nodes:
        raise ValueError("OSM 보행자 네트워크를 생성하지 못했습니다.")

    return G


def _nodes_to_polyline(G: nx.MultiDiGraph, nodes: List[int]) -> Polyline:
    """
    노드 ID 리스트를 (lat, lng) polyline으로 변환.
    """
    poly: Polyline = []
    for nid in nodes:
        data = G.nodes[nid]
        lat = safe_float(data.get("y"))
        lng = safe_float(data.get("x"))
        poly.append((lat, lng))
    return poly


# ============================================================
# 4. 메인 루프 생성 함수 (삼각 루프 기반)
# ============================================================

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    요청 좌표(lat, lng)와 목표 거리(km)를 기반으로
    보행자용 러닝 루프를 생성한다.

    설계 포인트:
      - OSMnx walk + custom_filter 기반 보행자 그래프 사용
      - start → A → B → start 삼각 루프를 여러 개 샘플링
      - 중첩(overlap)과 급커브를 강하게 패널티
      - 길이 오차 ±5% 이내만 length_ok=True 로 인정
      - polyline 앞/뒤에 요청 start 좌표를 앵커링
    """
    start_time = time.time()
    target_m = km * 1000.0

    meta: Dict[str, Any] = {
        "len": 0.0,
        "err": 0.0,
        "roundness": 0.0,
        "overlap": 0.0,
        "curve_penalty": 0.0,
        "score": -1e9,
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

    # 스코어링 가중치
    ROUNDNESS_WEIGHT = 1.5
    OVERLAP_PENALTY = 3.0      # 겹침 패널티 강화 (직선 왕복 제거용)
    CURVE_PENALTY_WEIGHT = 0.4
    LENGTH_PENALTY_WEIGHT = 10.0
    LENGTH_TOL_FRAC = 0.05      # 허용거리 오차 비율 (±5%)

    # --------------------------------------------------------
    # 1) 보행자 그래프 로딩
    # --------------------------------------------------------
    try:
        G = _build_pedestrian_graph(lat, lng, km)
    except Exception as e:
        # 그래프 생성에 실패하면 바로 fallback 사각형 루프
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
            message=f"보행자 네트워크 생성 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    UG: nx.Graph = G.to_undirected()

    # --------------------------------------------------------
    # 2) 시작 노드 스냅
    # --------------------------------------------------------
    try:
        start_node = ox.distance.nearest_nodes(G, lng, lat)
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
            message=f"시작 지점을 그래프에 스냅하지 못해 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------------------------------------
    # 3) start 기준 거리 맵 (Dijkstra)
    # --------------------------------------------------------
    try:
        dist_from_start = nx.single_source_dijkstra_path_length(UG, start_node, weight="length")
    except Exception:
        dist_from_start = {}

    if not dist_from_start:
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
            message="start 기준 거리 계산에 실패하여 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------------------------------------
    # 4) 삼각 루프용 앵커 노드 후보 (start에서 적당히 떨어진 노드)
    #    - 너무 가까운 노드는 루프가 찌그러지고,
    #      너무 먼 노드는 삼각 루프가 과도하게 긴 경우가 많음.
    # --------------------------------------------------------
    ring_min = target_m * 0.25   # start에서 1/4 지점 이상
    ring_max = target_m * 0.75   # 3/4 지점 이하

    ring_nodes: List[int] = [
        nid for nid, d in dist_from_start.items()
        if ring_min <= d <= ring_max
    ]

    if len(ring_nodes) < 3:
        # 루프를 구성할 앵커가 부족하면 fallback
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
            message="삼각 루프 구성에 필요한 앵커 노드가 부족해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    random.shuffle(ring_nodes)
    MAX_TRIANGLE_TRIALS = 400  # 삼각 루프 샘플링 횟수 상한

    best_score = -1e9
    best_poly: Optional[Polyline] = None
    best_meta_stats: Dict[str, Any] = {
        "len": 0.0,
        "err": 0.0,
        "roundness": 0.0,
        "overlap": 0.0,
        "curve_penalty": 0.0,
        "score": -1e9,
        "length_ok": False,
    }

    trials = 0
    n_ring = len(ring_nodes)

    # --------------------------------------------------------
    # 5) start → A → B → start 삼각 루프를 랜덤하게 여러 개 생성
    # --------------------------------------------------------
    while trials < MAX_TRIANGLE_TRIALS:
        trials += 1
        meta["routes_checked"] += 1

        # A, B를 ring_nodes에서 랜덤하게 선택
        if n_ring < 2:
            break

        a = ring_nodes[trials % n_ring]
        b = random.choice(ring_nodes)
        if a == b:
            continue

        try:
            # 세 구간의 최단 경로
            path1 = nx.shortest_path(UG, start_node, a, weight="length")
            path2 = nx.shortest_path(UG, a, b, weight="length")
            path3 = nx.shortest_path(UG, b, start_node, weight="length")

            full_nodes = path1 + path2[1:] + path3[1:]  # 중복 노드 제거
            poly = _nodes_to_polyline(G, full_nodes)

            L = polyline_length_m(poly)
            if L <= 0.0:
                continue

            r = polygon_roundness(poly)
            ov = _edge_overlap_fraction(poly)
            cp = _curve_penalty(poly)

            # 직선 왕복에 가까운 패턴은 early reject
            if ov > 0.9 and r < 0.2:
                continue

            err = abs(L - target_m)
            length_pen = err / max(1.0, target_m)

            score = (
                ROUNDNESS_WEIGHT * r
                - OVERLAP_PENALTY * ov
                - CURVE_PENALTY_WEIGHT * cp
                - LENGTH_PENALTY_WEIGHT * length_pen
            )

            is_length_ok = err <= target_m * LENGTH_TOL_FRAC

            meta["routes_validated"] += 1

            # 길이 오차가 작고, 스코어가 높은 쪽을 우선
            if score > best_score:
                best_score = score
                best_poly = poly
                best_meta_stats = {
                    "len": L,
                    "err": err,
                    "roundness": r,
                    "overlap": ov,
                    "curve_penalty": cp,
                    "score": score,
                    "length_ok": is_length_ok,
                }

                # 아주 좋은 해가 나오면 조기 종료 (성능/반응속도 절충)
                if is_length_ok and r > 0.4 and ov < 0.4:
                    break

        except Exception:
            # 경로 미존재 등은 무시하고 다음 시도
            continue

    # --------------------------------------------------------
    # 6) 삼각 루프 후보가 없으면 fallback 사각형
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
            routes_checked=meta["routes_checked"],
            routes_validated=meta["routes_validated"],
            message="OSM 기반 삼각 러닝 루프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------------------------------------
    # 7) 최종 meta 구성 + 시작 좌표 앵커링
    # --------------------------------------------------------
    used_fallback = False

    # 시작 좌표(요청 파라미터)를 polyline의 처음/끝에 앵커링
    if best_poly:
        first_lat, first_lng = best_poly[0]
        if haversine(lat, lng, first_lat, first_lng) > 1.0:
            best_poly.insert(0, (lat, lng))

        last_lat, last_lng = best_poly[-1]
        if haversine(lat, lng, last_lat, last_lng) > 1.0:
            best_poly.append((lat, lng))

        # 앵커링 후 길이/오차를 다시 계산
        length_m2 = polyline_length_m(best_poly)
        err2 = abs(length_m2 - target_m)
        length_ok2 = err2 <= target_m * LENGTH_TOL_FRAC

        best_meta_stats["len"] = length_m2
        best_meta_stats["err"] = err2
        best_meta_stats["length_ok"] = length_ok2

    success = best_meta_stats["length_ok"]

    meta.update(best_meta_stats)
    meta.update(
        success=success,
        used_fallback=used_fallback,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else "요청 오차(±5%)를 초과하지만, 가장 인접한 보행자용 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
