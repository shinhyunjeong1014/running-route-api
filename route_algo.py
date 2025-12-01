import math
import random
import time
from typing import List, Tuple, Dict, Any
import networkx as nx
import osmnx as ox


# ==========================
# JSON-safe 변환 유틸
# ==========================
def safe_float(x: Any):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def safe_list(lst):
    out = []
    for v in lst:
        if isinstance(v, float):
            out.append(safe_float(v))
        elif isinstance(v, dict):
            out.append(safe_dict(v))
        elif isinstance(v, list):
            out.append(safe_list(v))
        else:
            out.append(v)
    return out


def safe_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = safe_float(v)
        elif isinstance(v, dict):
            out[k] = safe_dict(v)
        elif isinstance(v, list):
            out[k] = safe_list(v)
        else:
            out[k] = v
    return out


# ==========================
# 거리 계산
# ==========================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline: List[Tuple[float, float]]) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lon1, lat2, lon2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ==========================
# 개선된 Roundness 계산 (RUNAMIC 스타일)
# ==========================
def _to_local_xy(polyline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not polyline:
        return []
    ref_lat = polyline[0][0]
    ref_lon = polyline[0][1]
    R = 6371000.0
    cos_ref = math.cos(math.radians(ref_lat))
    xy = []
    for lat, lon in polyline:
        x = (lon - ref_lon) * cos_ref * R
        y = (lat - ref_lat) * R
        xy.append((x, y))
    return xy


def polygon_roundness_enhanced(polyline: List[Tuple[float, float]]) -> float:
    """
    개선된 roundness: 
    1) 기본 4πA/P² (원형도)
    2) 중심으로부터 거리 분산 (편차 페널티)
    3) 볼록도 (convexity) 고려
    """
    if len(polyline) < 4:
        return 0.0
    
    xy = _to_local_xy(polyline)
    
    # 1) 면적과 둘레
    area = 0.0
    perimeter = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        area += x1 * y2 - x2 * y1
        perimeter += math.hypot(x2 - x1, y2 - y1)
    area = abs(area) * 0.5
    
    if area == 0 or perimeter == 0:
        return 0.0
    
    # 기본 roundness
    basic_r = 4 * math.pi * area / (perimeter ** 2)
    
    # 2) 중심으로부터 거리 분산 계산
    cx = sum(p[0] for p in xy) / len(xy)
    cy = sum(p[1] for p in xy) / len(xy)
    
    distances = [math.hypot(p[0] - cx, p[1] - cy) for p in xy]
    mean_dist = sum(distances) / len(distances)
    
    if mean_dist > 0:
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_dist  # 변동계수
        
        # CV가 작을수록 원에 가까움 (0.2 이하가 이상적)
        uniformity_score = max(0, 1.0 - cv / 0.3)
    else:
        uniformity_score = 0.0
    
    # 3) 최종 점수: 기본 roundness + uniformity 가중평균
    final_score = 0.6 * basic_r + 0.4 * uniformity_score
    
    return max(0.0, min(1.0, final_score))


# ==========================
# Overlap/Curve 계산
# ==========================
def _edge_overlap_fraction(node_path: List[int]) -> float:
    if len(node_path) < 2:
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
    if len(node_path) < 3:
        return 0.0
    penalty = 0.0
    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        
        if b not in G.nodes or a not in G.nodes or c not in G.nodes:
            continue
            
        ya, xa = G.nodes[a]["y"], G.nodes[a]["x"]
        yb, xb = G.nodes[b]["y"], G.nodes[b]["x"]
        yc, xc = G.nodes[c]["y"], G.nodes[c]["x"]

        v1x = xb - xa
        v1y = yb - ya
        v2x = xc - xb
        v2y = yc - yb
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0 or n2 == 0:
            continue

        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)

        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    if len(nodes) < 2:
        return 0.0
    length = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            continue
        data = min(G[u][v].values(), key=lambda d: d.get("length", 1.0))
        length += float(data.get("length", 0.0))
    return length


def _apply_route_poison(G: nx.Graph, path_nodes: List[int], factor: float = 10.0) -> nx.Graph:
    """RUNAMIC 스타일 route poisoning"""
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


# ==========================
# OSM 보행자 그래프 구축
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    radius_m = max(700.0, km * 500.0 + 700.0)
    
    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="walk", 
        simplify=True,
        retain_all=False
    )
    
    if not G.nodes:
         raise ValueError("Filtered graph has no nodes.")
         
    return G


def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> List[Tuple[float, float]]:
    poly = []
    for n in nodes:
        node = G.nodes[n]
        lat = float(node["y"])
        lon = float(node["x"])
        poly.append((lat, lon))
    return poly


# ==========================
# Fallback: 기하학적 사각형 루프
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float):
    target_m = km * 1000.0
    side = target_m / 4.0
    delta_deg_lat = side / 111000.0
    cos_lat = math.cos(math.radians(lat))
    delta_deg_lng = side / (111000.0 * cos_lat if cos_lat != 0 else 111000.0)

    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]
    
    center_lat = (a[0] + c[0]) / 2
    center_lng = (b[1] + d[1]) / 2
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]
    
    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness_enhanced(poly)
    return poly, length, r


# ==========================
# 메인: 개선된 러닝 루프 생성기
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    [V6 개선판] RUNAMIC + Efficient Computation 논문 기반
    
    주요 개선사항:
    1. 다중 후보 생성 및 비교 (RUNAMIC 방식)
    2. 길이 범위 축소 (0.4~0.5 * target)
    3. 개선된 roundness 계산
    4. Pareto 최적화 고려
    """
    start_time = time.time()
    target_m = km * 1000.0
    
    # [개선] 스코어링 가중치 재조정
    ROUNDNESS_WEIGHT = 4.0  # 원형도 중요성 증가
    OVERLAP_PENALTY = 3.0   # 중복 페널티 강화
    CURVE_PENALTY_WEIGHT = 0.5
    LENGTH_PENALTY_WEIGHT = 15.0  # 길이 오차 페널티 더욱 강화

    meta: Dict[str, Any] = {
        "len": None,
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
        "km_requested": km,
        "target_m": target_m,
        "time_s": None,
        "message": ""
    }

    # 1) OSM 보행자 그래프 구축
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            message=f"OSM 그래프 생성 실패로 fallback 사용: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0.0, curve_penalty=0.0,
            score=r, success=False, length_ok=(err <= 99.0), used_fallback=True,
            message=f"시작 노드 매칭 실패: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)

    # 2) [개선] Rod 길이 범위 축소 (정확도 향상)
    try:
        dist = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.7,  # 탐색 범위 확대
            weight="length"
        )
    except Exception as e:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0.0, curve_penalty=0.0,
            score=r, success=False, length_ok=(err <= 99.0), used_fallback=True,
            message=f"최단거리 탐색 실패: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # [핵심 개선] Rod 길이를 0.4~0.5로 축소 (더 정확한 루프)
    min_leg = target_m * 0.40
    max_leg = target_m * 0.50
    candidate_nodes = [n for n, d in dist.items() if min_leg <= d <= max_leg and n != start_node]

    if not candidate_nodes:
        candidate_nodes = [n for n, d in dist.items() if d >= target_m * 0.3]

    if not candidate_nodes:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0.0, curve_penalty=0.0,
            score=r, success=False, length_ok=(err <= 99.0), used_fallback=True,
            message="적절한 rod endpoint 후보를 찾지 못함"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # [개선] 더 많은 후보 탐색 (50개)
    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:50]

    best_score = -1e18
    best_poly = None
    best_meta_stats = {}

    # 3) [개선] 다중 후보 비교 (RUNAMIC 방식)
    for endpoint in candidate_nodes:
        
        try:
            forward_nodes = nx.shortest_path(undirected, start_node, endpoint, weight="length")
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len <= 0:
            continue

        # Poisoning
        poisoned = _apply_route_poison(undirected, forward_nodes, factor=10.0)

        try:
            back_nodes = nx.shortest_path(poisoned, endpoint, start_node, weight="length")
        except Exception:
            continue

        back_len = _path_length_on_graph(undirected, back_nodes)
        if back_len <= 0:
            continue

        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        polyline = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(polyline)
        if length_m <= 0:
            continue

        err = abs(length_m - target_m)
        
        # [개선] 향상된 roundness 계산 사용
        roundness = polygon_roundness_enhanced(polyline)
        overlap = _edge_overlap_fraction(full_nodes)
        curve_penalty = _curve_penalty(full_nodes, undirected)

        length_ok = err <= 99.0
        if length_ok:
            meta["routes_validated"] += 1
            
        # [개선] 길이 오차에 비선형 페널티 적용
        length_pen = (err / target_m) ** 1.5  # 제곱근 페널티로 큰 오차 더 강하게 처벌
        
        score = (
            roundness * ROUNDNESS_WEIGHT
            - overlap * OVERLAP_PENALTY
            - curve_penalty * CURVE_PENALTY_WEIGHT
            - length_pen * LENGTH_PENALTY_WEIGHT
        )
        
        # 최고 점수 갱신
        if score > best_score:
            best_score = score
            best_poly = polyline
            best_meta_stats = {
                "len": length_m,
                "err": err,
                "roundness": roundness,
                "overlap": overlap,
                "curve_penalty": curve_penalty,
                "score": score,
                "length_ok": length_ok,
            }

    # 4) 결과 반환
    if best_poly is None:
        poly, length, r = _fallback_square_loop(lat, lng, km)
        err = abs(length - target_m)
        meta.update(
            len=length, err=err, roundness=r, overlap=0.0, curve_penalty=0.0,
            score=r, success=False, length_ok=(err <= 99.0), used_fallback=True,
            routes_checked=meta["routes_checked"],
            routes_validated=meta["routes_validated"],
            message="루프 생성 실패로 fallback 사용"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 5) 최종 메타 구성
    success = best_meta_stats["length_ok"]

    meta.update(best_meta_stats)
    meta.update(
        success=success,
        used_fallback=False,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else f"오차 {best_meta_stats['err']:.1f}m로 가장 근접한 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    return safe_list(best_poly), safe_dict(meta)
