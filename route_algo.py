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
    """NaN / Inf 를 JSON에서 허용 가능한 값(None)으로 변환."""
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
# 거리 계산 (Haversine)
# ==========================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 점 사이의 대원거리 (미터)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline: List[Tuple[float, float]]) -> float:
    """경로(위도,경도 리스트)의 총 길이 (미터)."""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lon1, lat2, lon2)
    if math.isinf(total) or math.isnan(total):
        return 0.0
    return total


# ==========================
# roundness 계산용 로컬 좌표 변환
# ==========================
def _to_local_xy(polyline: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """작은 영역에서는 위경도를 간단한 평면 좌표(미터)로 근사."""
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


def polygon_roundness(polyline: List[Tuple[float, float]]) -> float:
    """
    4πA / P^2 로 정의되는 roundness.
    1에 가까울수록 원에 가까운 형태, 0에 가까울수록 길쭉/찌그러진 형태.
    """
    if len(polyline) < 4:
        return 0.0
    xy = _to_local_xy(polyline)
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
    r = 4 * math.pi * area / (perimeter ** 2)
    if math.isinf(r) or math.isnan(r):
        return 0.0
    return r


# ==========================
# overlap / 커브 페널티
# ==========================
def _edge_overlap_fraction(node_path: List[int]) -> float:
    """
    노드 시퀀스에서 같은 간선을 여러 번 쓰는 비율.
    (rod + detour 구조에서 중복을 줄이기 위해 사용)
    """
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
    # 전체 고유 간선 수 대비 중복 간선 수 비율
    overlap_edges = sum(1 for c in edge_counts.values() if c > 1)
    return overlap_edges / len(edge_counts)


def _curve_penalty(node_path: List[int], G: nx.Graph) -> float:
    """
    경로에서 너무 급격한 커브(예: 60도 이하)를 얼마나 많이 만드는지 측정.
    작을수록 부드러운 경로.
    """
    if len(node_path) < 3:
        return 0.0
    penalty = 0.0
    for i in range(1, len(node_path) - 1):
        a = node_path[i - 1]
        b = node_path[i]
        c = node_path[i + 1]
        
        # 실제 위경도 좌표 사용 (y: lat, x: lon)
        if b not in G.nodes or a not in G.nodes or c not in G.nodes:
            continue
            
        ya, xa = G.nodes[a]["y"], G.nodes[a]["x"]
        yb, xb = G.nodes[b]["y"], G.nodes[b]["x"]
        yc, xc = G.nodes[c]["y"], G.nodes[c]["x"]

        # 벡터 계산: v1 = a->b, v2 = b->c
        v1x = xb - xa
        v1y = yb - ya
        v2x = xc - xb
        v2y = yc - yb
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0 or n2 == 0:
            continue

        # 내적을 이용한 각도 계산
        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        theta = math.acos(dot)  # 0 ~ pi (180도)

        # 너무 급한 커브(60도 미만, pi/3)에 페널티
        if theta < math.pi / 3.0:
            penalty += (math.pi / 3.0 - theta)

    return penalty


def _path_length_on_graph(G: nx.Graph, nodes: List[int]) -> float:
    """그래프 상에서 node 경로의 길이 (edge의 length 합)."""
    if len(nodes) < 2:
        return 0.0
    length = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(u, v):
            continue
        # MultiGraph 이므로 여러 edge 중 length가 최소인 것 사용
        data = min(G[u][v].values(), key=lambda d: d.get("length", 1.0))
        length += float(data.get("length", 0.0))
    return length


def _apply_route_poison(G: nx.Graph, path_nodes: List[int], factor: float = 6.0) -> nx.Graph:
    """
    RUNAMIC 스타일 route poisoning:
    rod 에 해당하는 간선의 비용(length)을 factor만큼 늘려
    되돌아올 때 같은 길을 반복해서 타는 것을 억제.
    """
    G2 = G.copy()
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if G2.has_edge(u, v):
            # MultiGraph 이므로 모든 key에 대해 업데이트
            for k in list(G2[u][v].keys()):
                data = G2[u][v][k]
                if "length" in data:
                    data["length"] = float(data["length"]) * factor
            if G2.has_edge(v, u): # 역방향 엣지도 동일하게 처리
                for k in list(G2[v][u].keys()):
                    data = G2[v][u][k]
                    if "length" in data:
                        data["length"] = float(data["length"]) * factor
    return G2


# ==========================
# OSM 보행자 그래프 구축 (개선)
# ==========================
def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.MultiDiGraph:
    """
    [V2 개선 적용] 러닝에 부적합한 도로는 필터링합니다. (아파트 단지 통과 문제 완화)
    (Residential, Service, Steps 등 차량/사유 통행로 비선호)
    """
    # km가 커질수록 반경을 넉넉히 잡되, 너무 크게는 안 가게 제한
    radius_m = max(800.0, km * 700.0 + 500.0)
    
    # [V2 필터 구문 수정 및 강화]
    # residential, service (사유 통행로), steps (계단) 등을 제외하고 러닝에 적합한 경로 위주로 필터링
    custom_filter = (
        '["highway"~"footway|path|pedestrian|track|cycleway|bridleway|tertiary|unclassified"]' + 
        '["area"!~"yes"]' # OSMnx가 기본으로 넣는 필터 재사용
    )
    
    # Overpass API가 'highway' 태그 외에도 'area' 태그를 가진 객체를 반환할 수 있으므로,
    # 'highway' 값 목록만으로 필터링하는 것이 가장 안전합니다.
    # OpenStreetMap wiki의 보행자/러닝 경로 기준을 엄격하게 적용
    
    # V2: 더 안전하고 엄격한 필터링 구문 사용
    strict_pedestrian_filter = (
        '["highway"~"footway|path|pedestrian|track|cycleway|bridleway|tertiary_link|secondary_link|primary_link"]' +
        '["area"!~"yes"]' +
        '["service"!~"parking_aisle|driveway|private"]'
    )
    
    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        custom_filter=strict_pedestrian_filter,
        network_type="all_private", # custom_filter가 적용되도록 all_private 사용
        simplify=True,
        retain_all=False
    )
    
    # 필터링 후 노드가 없으면 에러
    if not G.nodes:
         raise ValueError("Filtered graph has no nodes.")
         
    return G


def _nodes_to_polyline(G: nx.Graph, nodes: List[int]) -> List[Tuple[float, float]]:
    """그래프 노드 시퀀스를 (lat, lng) polyline으로 변환."""
    poly = []
    for n in nodes:
        node = G.nodes[n]
        lat = float(node["y"])
        lon = float(node["x"])
        poly.append((lat, lon))
    return poly


# ==========================
# fallback: 기하학적 사각형 루프
# ==========================
def _fallback_square_loop(lat: float, lng: float, km: float):
    """
    모든 고급 알고리즘 실패 시 사용되는 마지막 안전 장치.
    요청 거리 km를 대략 만족하는 사각형 루프 생성.
    """
    target_m = km * 1000.0
    # roundness가 1에 가까운 4변형
    side = target_m / 4.0  
    delta_deg_lat = side / 111000.0
    cos_lat = math.cos(math.radians(lat))
    delta_deg_lng = side / (111000.0 * cos_lat if cos_lat != 0 else 111000.0)

    # 원점에서 시작하여 시계 방향으로 4개 꼭짓점
    a = (lat + delta_deg_lat, lng)
    b = (lat, lng + delta_deg_lng)
    c = (lat - delta_deg_lat, lng)
    d = (lat, lng - delta_deg_lng)
    poly = [a, b, c, d, a]
    
    # 시작점을 중앙으로 재배치
    center_lat = (a[0] + c[0]) / 2
    center_lng = (b[1] + d[1]) / 2
    poly = [(p[0] - center_lat + lat, p[1] - center_lng + lng) for p in poly]
    
    poly = [(float(x), float(y)) for x, y in poly]
    length = polyline_length_m(poly)
    r = polygon_roundness(poly)
    return poly, length, r


# ==========================
# 메인: 러닝 루프 생성기
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    PURE PEDESTRIAN 러닝 루프 생성기 (V2 개선)
    """
    start_time = time.time()
    target_m = km * 1000.0
    
    # [V2 개선] 스코어링 가중치 상향 정의
    ROUNDNESS_WEIGHT = 3.0 # Roundness 가중치 강화 (1.5 -> 3.0)
    OVERLAP_PENALTY = 2.0
    CURVE_PENALTY_WEIGHT = 0.3
    LENGTH_PENALTY_WEIGHT = 5.0 # 길이 오차 페널티 강화 (3.0 -> 5.0)

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
        "valhalla_calls": 0,
        "kakao_calls": 0,
        "routes_checked": 0,
        "routes_validated": 0,
        "km_requested": km,
        "target_m": target_m,
        "time_s": None,
        "message": ""
    }

    # --------------------------
    # 1) OSM 보행자 그래프 구축
    # --------------------------
    try:
        # [V2 개선] 필터링 강화된 함수 사용
        G = _build_pedestrian_graph(lat, lng, km) 
    except Exception as e:
        # 그래프 자체를 못 가져오면 바로 fallback
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
            routes_checked=0,
            routes_validated=0,
            message=f"OSM 보행자 그래프 생성/필터링 실패로 기하학적 사각형 루프를 사용했습니다: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # undirected 그래프로 변환 (보행자 왕복 경로에 더 적합)
    try:
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
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
            routes_checked=0,
            routes_validated=0,
            message=f"시작 노드 매칭 실패로 기하학적 사각형 루프를 사용했습니다: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    undirected: nx.MultiGraph = ox.utils_graph.get_undirected(G)

    # --------------------------
    # 2) start에서의 단일-출발 최단거리 (rod 후보 탐색)
    # --------------------------
    try:
        # target_m의 80% 정도까지의 노드들 거리
        dist = nx.single_source_dijkstra_path_length(
            undirected,
            start_node,
            cutoff=target_m * 0.8,
            weight="length"
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=0,
            routes_validated=0,
            message=f"그래프 최단거리 탐색 실패로 기하학적 사각형 루프를 사용했습니다: {e}"
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # [V2 개선] rod 길이 후보: 0.35 ~ 0.6 * target_m 사이 (max_leg 축소)
    min_leg = target_m * 0.35
    max_leg = target_m * 0.60
    candidate_nodes = [n for n, d in dist.items() if min_leg <= d <= max_leg and n != start_node]

    # 후보가 너무 적으면 조건 완화
    if not candidate_nodes:
        candidate_nodes = [n for n, d in dist.items() if d >= target_m * 0.25]

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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=0,
            routes_validated=0,
            message="적절한 rod endpoint 후보를 찾지 못해 기하학적 사각형 루프를 사용했습니다."
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # 너무 많은 노드를 다 쓰면 느려지므로 랜덤 샘플링
    random.shuffle(candidate_nodes)
    candidate_nodes = candidate_nodes[:40]

    best_score = -1e18
    best_poly = None
    
    # [V2 개선] Phase 1: 길이 허용 오차 내 경로가 있는지 체크 (Flag)
    found_length_ok = False 
    
    best_meta = {}

    # --------------------------
    # 3) 각 endpoint에 대해 rod + detour 루프 생성
    # --------------------------
    for endpoint in candidate_nodes:
        
        # 3-1) start -> endpoint rod (최단 경로)
        try:
            forward_nodes = nx.shortest_path(
                undirected,
                start_node,
                endpoint,
                weight="length"
            )
        except Exception:
            continue

        forward_len = _path_length_on_graph(undirected, forward_nodes)
        if forward_len <= 0:
            continue

        # 3-2) rod 간선에 penalty를 줘서 detour 경로 유도 (RUNAMIC 아이디어)
        # penalty factor = 6.0 유지
        poisoned = _apply_route_poison(undirected, forward_nodes, factor=6.0)

        # endpoint -> start detour
        try:
            # MultiGraph.shortest_path()는 기본적으로 MultiDiGraph의 최단 경로를 찾는다.
            # undirected로 변환했어도 shortest_path가 동작
            back_nodes = nx.shortest_path(
                poisoned,
                endpoint,
                start_node,
                weight="length"
            )
        except Exception:
            continue

        back_len = _path_length_on_graph(undirected, back_nodes)
        if back_len <= 0:
            continue

        # full loop = rod + detour (노드 중복 방지 위해 back_nodes[1:]부터 이어붙임)
        full_nodes = forward_nodes + back_nodes[1:]
        meta["routes_checked"] += 1

        polyline = _nodes_to_polyline(undirected, full_nodes)
        length_m = polyline_length_m(polyline)
        if length_m <= 0:
            continue

        err = abs(length_m - target_m)
        roundness = polygon_roundness(polyline)
        overlap = _edge_overlap_fraction(full_nodes)
        curve_penalty = _curve_penalty(full_nodes, undirected)

        length_ok = err <= 99.0
        
        if length_ok:
            found_length_ok = True
            meta["routes_validated"] += 1
            
        # [V2 개선] 스코어링 가중치 적용
        length_pen = err / target_m 
        score = (
            roundness * ROUNDNESS_WEIGHT
            - overlap * OVERLAP_PENALTY
            - curve_penalty * CURVE_PENALTY_WEIGHT
            - length_pen * LENGTH_PENALTY_WEIGHT
        )
        
        # [V2 개선] 최적 경로 선택 로직 강화:
        # Phase 1: 길이 충족 경로가 발견되면, 이후는 길이 충족 경로만 경쟁 가능
        if not found_length_ok or length_ok:
            if score > best_score:
                best_score = score
                best_poly = polyline
                best_meta = {
                    "len": length_m,
                    "err": err,
                    "roundness": roundness,
                    "overlap": overlap,
                    "curve_penalty": curve_penalty,
                    "score": score,
                    "length_ok": length_ok,
                }
        
    # --------------------------
    # 4) 후보 루프가 하나도 없을 때
    # --------------------------
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
            length_ok=(err <= 99.0),
            used_fallback=True,
            routes_checked=meta["routes_checked"],
            routes_validated=meta["routes_validated"],
            message="논문 기반 OSM 루프 생성에 실패하여 기하학적 사각형 루프를 사용했습니다."
        )
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    # --------------------------
    # 5) 최종 meta 구성
    # --------------------------
    used_fallback = False
    success = best_meta["length_ok"]

    meta.update(best_meta)
    meta.update(
        success=success,
        used_fallback=used_fallback,
        routes_checked=meta["routes_checked"],
        routes_validated=meta["routes_validated"],
        message=(
            "최적의 정밀 경로가 도출되었습니다."
            if success
            else "요청 오차(±99m)를 초과하지만, 가장 인접한 러닝 루프를 반환합니다."
        ),
    )
    meta["time_s"] = time.time() - start_time

    safe_meta = safe_dict(meta)
    safe_poly = safe_list(best_poly)
    return safe_poly, safe_meta
