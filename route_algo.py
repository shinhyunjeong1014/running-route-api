from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx

try:
    import osmnx as ox
except Exception:
    ox = None

LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ============================================================
# 1. 기본 유틸 함수들 (유지)
# ============================================================

def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def polyline_length_m(polyline: Polyline) -> float:
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
    dx = haversine(center_lat, center_lng, center_lat, lng)
    dy = haversine(center_lat, center_lng, lat, center_lng)
    if lng < center_lng:
        dx = -dx
    if lat < center_lat:
        dy = -dy
    return dx, dy


def polygon_roundness(poly: Polyline) -> float:
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
    roundness = max(0.0, 1.0 - std / (r_mean + 1e-6))
    return float(roundness)


def _edge_overlap_fraction(poly: Polyline) -> float:
    if len(poly) < 3:
        return 0.0

    seen = {}
    total = 0
    overlap = 0

    def quantize(x: float) -> int:
        return int(round(x * 10))

    for (lat1, lng1), (lat2, lng2) in zip(poly[:-1], poly[1:]):
        seg_len = haversine(lat1, lng1, lat2, lng2)
        steps = max(1, int(seg_len / 3.0))
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
        penalty += max(0.0, (180.0 - angle) / 90.0)

    return float(penalty / (len(poly) - 2))


def _graph_path_length(G: nx.Graph, path: List[int]) -> float:
    """
    그래프 상의 path 길이를 edge 'length' 속성 기준으로 계산.
    """
    if not path or len(path) < 2:
        return 0.0

    length = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G[u][v]
        if hasattr(data, "values"):
            edge_datas = list(data.values())
            if edge_datas and isinstance(edge_datas[0], dict):
                best = min(edge_datas, key=lambda x: x.get("length", 1.0))
                length += float(best.get("length", 0.0))
            else:
                length += float(data.get("length", 0.0))
        else:
            length += 0.0
    return float(length)


def _apply_route_poison(G: nx.MultiDiGraph, path_nodes: List[int], factor: float = 10.0) -> nx.MultiDiGraph:
    """
    [V4-C 유지] Poisoning factor를 10.0으로 설정.
    """
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G2.has_edge(u, v):
            for k in list(G2[u][v].keys()):
                data = G2[u][v][k]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
            if G2.has_edge(v, u):
                for k in list(G2[v][u].keys()):
                    data = G2[v][u][k]
                    if "length" in data:
                        data["length"] = float(data["length"]) * factor
    return G2


# ============================================================
# 3. OSMnx 보행자 그래프 구성 (V4-C 안정화)
# ============================================================

def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    [V4-C 유지] 아파트 단지 진입 및 부적합 경로를 차단하는 엄격한 필터 적용.
    """
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다.")

    radius_m = max(700.0, km * 500.0 + 700.0)

    # [V4-C 필터링] 보행자 전용 및 친화 경로만 포함 (residential, service, steps, motorway 등 제외)
    custom_filter = ('["highway"~"footway|path|pedestrian|track|cycleway|sidewalk"]') + \
                    '["access"!~"private"]'

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        custom_filter=custom_filter,
        network_type="all_private",
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
# 4. Fallback 사각형 루프 (유지)
# ============================================================

def _fallback_square_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, float, float]:
    """
    네트워크 기반 루프를 만들지 못했을 때 사용하는 기하학적 사각형 루프.
    """
    target_m = km * 1000.0
    side_m = target_m / 4.0

    delta_deg_lat = side_m / 111_000.0
    delta_deg_lng = side_m / (111_000.0 * math.cos(math.radians(lat)) + 1e-6)

    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]

    center_lat = (a[0] + c[0]) / 2.0
    center_lng = (b[1] + d[1]) / 2.0
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]

    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ============================================================
# 5. 메인 루프 생성 함수
# ============================================================

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[Polyline, Dict[str, Any]]:
    """
    V4-C (PSP 형태의 Rod 후보 선정 및 Detour 제약 완화)
    """
    start_time = time.time()
    target_m = km * 1000.0

    # 스코어링 가중치 (Overlap Penalty 극단적 강화)
    ROUNDNESS_WEIGHT = 3.0
    OVERLAP_PENALTY = 20.0
    CURVE_PENALTY_WEIGHT = 0.3
    LENGTH_PENALTY_WEIGHT = 10.0
    LENGTH_TOL_FRAC = 0.05  # 허용거리 오차 비율 (±5%)

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

    # --------------------------------------------------------
    # 1) 보행자 그래프 로딩
    # --------------------------------------------------------
    try:
        G = _build_pedestrian_graph(lat, lng, km)
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
            message=f"보행자 네트워크 생성 실패로 기하학적 사각형 루프를 사용했습니다: {e}",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    UG: nx.MultiDiGraph = G

    # --------------------------------------------------------
    # 2) 시작 노드 스냅 및 Rod 후보 탐색
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

    try:
        lengths = nx.single_source_dijkstra_path_length(UG, start_node, weight="length")
    except Exception:
        lengths = {}

    # [V4-C 변경] Rod 후보 넓은 범위 허용 (PSP 원칙)
    min_leg = target_m * 0.30  # 30%까지 허용
    max_leg = target_m * 0.70  # 70%까지 허용

    candidate_nodes: List[int] = []
    for nid, dist_m in lengths.items():
        if min_leg <= dist_m <= max_leg and dist_m >= 0.0 and nid != start_node:
            candidate_nodes.append(nid)

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
            message="적절한 rod endpoint 후보를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------------------------------------
    # 3) 각 endpoint 후보에 대해 루프 구성 및 스코어링
    # --------------------------------------------------------
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

    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:250]

    for endpoint in candidate_nodes:
        meta["routes_checked"] += 1
        try:
            # 3-1. Rod (Start -> End)
            path_out: List[int] = nx.shortest_path(UG, start_node, endpoint, weight="length")
            dist_out = _graph_path_length(UG, path_out)

            # 3-2. Detour를 위한 Poisoning 적용
            poisoned_G = _apply_route_poison(UG, path_out, factor=10.0) 

            # 3-3. Detour (End -> Start)
            path_back: List[int] = nx.shortest_path(poisoned_G, endpoint, start_node, weight="length")
            dist_back = _graph_path_length(UG, path_back)

            # 3-4. [V4-C 제약 완화] Detour 길이 제약
            
            # Detour가 Rod 길이의 90% 미만이면 Overlap이 높다고 판단하여 폐기
            if dist_back < dist_out * 0.9: 
                 continue
            
            # [V4-C 제약 완화] Rod와 Detour의 합이 너무 크면 길이 폭발로 판단하여 폐기
            if dist_out + dist_back > target_m * 1.7: # 170% 초과 금지
                continue


            # 3-5. 왕복 루프 구성 및 검증
            full_nodes = path_out + path_back[1:]
            poly = _nodes_to_polyline(G, full_nodes)

            L = polyline_length_m(poly)
            if L <= 0.0:
                continue

            r = polygon_roundness(poly)
            ov = _edge_overlap_fraction(full_nodes) # 노드 목록으로 overlap 계산
            cp = _curve_penalty(poly)

            err = abs(L - target_m)
            length_pen = err / max(1.0, target_m)

            score = (
                ROUNDNESS_WEIGHT * r
                - OVERLAP_PENALTY * ov
                - CURVE_PENALTY_WEIGHT * cp
                - LENGTH_PENALTY_WEIGHT * length_pen
            )

            is_length_ok = err <= target_m * LENGTH_TOL_FRAC

            # 최종 점수 경쟁
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
                meta["routes_validated"] += 1

        except Exception:
            continue

    # --------------------------------------------------------
    # 4) 최종 유효성 검사 및 Fallback
    # --------------------------------------------------------
    if best_poly is None or not best_meta_stats.get("length_ok", False):
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        final_length_ok = err <= target_m * LENGTH_TOL_FRAC
        
        meta_fallback = {
            "len": length,
            "err": err,
            "roundness": r,
            "overlap": 0.0,
            "curve_penalty": 0.0,
            "score": r,
            "success": final_length_ok,
            "length_ok": final_length_ok,
            "used_fallback": True,
            "message": f"요청 거리 ({km}km)의 ±5% 내에 적합한 OSM 경로를 찾지 못하여 Fallback 루프를 사용했습니다."
        }
        
        meta.update(meta_fallback)
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)


    # --------------------------------------------------------
    # 5) 최종 meta 구성 + 시작 좌표 앵커링
    # --------------------------------------------------------
    used_fallback = False
    success = best_meta_stats["length_ok"]

    if best_poly:
        # 시작/끝 좌표 앵커링 (V1-A 로직 유지)
        first_lat, first_lng = best_poly[0]
        if haversine(lat, lng, first_lat, first_lng) > 1.0:
            best_poly.insert(0, (lat, lng))

        last_lat, last_lng = best_poly[-1]
        if haversine(lat, lng, last_lat, last_lng) > 1.0:
            best_poly.append((lat, lng))
            
        length_m2 = polyline_length_m(best_poly)
        err2 = abs(length_m2 - target_m)
        length_ok2 = err2 <= target_m * LENGTH_TOL_FRAC

        best_meta_stats["len"] = length_m2
        best_meta_stats["err"] = err2
        best_meta_stats["length_ok"] = length_ok2
        success = length_ok2


    meta.update(best_meta_stats)
    meta.update(
        success=success,
        used_fallback=used_fallback,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else "요청 오차(±5%)를 초과하지만, 가장 인접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    return safe_list(best_poly), safe_dict(meta)
