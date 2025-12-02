import math
import random
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx
import osmnx as ox

# ---------------------------------------------------------
# 공통 타입 정의
# ---------------------------------------------------------
LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ---------------------------------------------------------
# 1. 기본 유틸리티 함수들 (거리, 폴리라인 길이 등)
# ---------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 위경도 좌표 사이의 직선(대원) 거리(m)를 계산.
    """
    R = 6371000.0  # 지구 반지름 (m)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def polyline_length_m(polyline: Polyline) -> float:
    """
    폴리라인(위경도 리스트)의 총 길이(m)를 계산.
    (app.py에서 직접 import 해서 사용하는 함수이므로 반드시 유지)
    """
    if len(polyline) < 2:
        return 0.0

    dist = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        dist += haversine_m(lat1, lon1, lat2, lon2)

    return dist


# 내부에서만 사용하는 버전 (같은 계산이지만 이름만 다르게 둠)
def _polyline_length_m(polyline: Polyline) -> float:
    return polyline_length_m(polyline)


# ---------------------------------------------------------
# 2. OSMnx 그래프 구축
# ---------------------------------------------------------
def _build_pedestrian_graph(
    lat: float,
    lng: float,
    km: float,
    network_type: str = "walk",
) -> nx.MultiDiGraph:
    """
    중심 좌표 (lat, lng)를 기준으로 km 반경 내 보행 네트워크 그래프를 구축한다.
    """
    # 반경을 조금 넉넉하게 (요청 거리의 절반~3/4 정도의 선으로)
    # 예: 3km 요청이면 반경 약 2km 정도
    radius_m = max(500.0, km * 650.0)

    # OSMnx에서 보행 네트워크 그래프 추출
    G = ox.graph_from_point(
        center_point=(lat, lng),
        dist=radius_m,
        network_type=network_type,
        simplify=True,
    )

    # 노드에 lat/lng 속성이 이미 있음
    # 엣지 길이는 OSMnx가 기본적으로 'length' 속성에 넣어준다.
    return G


def _nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int:
    """
    그래프 G에서 (lat, lng)에 가장 가까운 노드 ID를 반환.
    """
    return ox.distance.nearest_nodes(G, lng, lat)


def _node_latlng(G: nx.MultiDiGraph, node: int) -> LatLng:
    return (G.nodes[node]["y"], G.nodes[node]["x"])


# ---------------------------------------------------------
# 3. 그래프 기반 경로 유틸리티
# ---------------------------------------------------------
def _graph_path_to_polyline(G: nx.MultiDiGraph, path: List[int]) -> Polyline:
    """
    그래프 상의 노드 시퀀스를 실제 위경도 폴리라인으로 변환.
    """
    return [_node_latlng(G, n) for n in path]


def _graph_path_length(G: nx.MultiDiGraph, path: List[int]) -> float:
    """
    그래프 상의 노드 경로에 대한 실제 도로 길이(m).
    (osmnx에서 엣지마다 'length' 속성이 들어있다고 가정)
    """
    if len(path) < 2:
        return 0.0

    length = 0.0
    for u, v in zip(path[:-1], path[1:]):
        # MultiDiGraph이므로 여러 엣지가 있을 수 있어 가장 짧은 length 사용
        edge_data_dict = G.get_edge_data(u, v)
        if not edge_data_dict:
            # 만약 엣지 정보가 없다면, 하버사인으로 대충 보정
            lat1, lon1 = _node_latlng(G, u)
            lat2, lon2 = _node_latlng(G, v)
            length += haversine_m(lat1, lon1, lat2, lon2)
            continue

        # 가장 짧은 엣지를 선택
        best_edge = min(edge_data_dict.values(), key=lambda d: d.get("length", 1.0))
        length += best_edge.get("length", 0.0)

    return length


def _edge_overlap_ratio(polyline: Polyline) -> float:
    """
    폴리라인에서 "같은 선분을 얼마나 많이 왕복하는지" 대략적인 중복 비율 평가.
    """
    if len(polyline) < 2:
        return 0.0

    # lat/lng를 조금 그리디하게 버킷화해서 동일 선분 판단
    def _bucket(p: LatLng) -> Tuple[int, int]:
        return (int(p[0] * 10000), int(p[1] * 10000))

    seg_count: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}

    for p1, p2 in zip(polyline[:-1], polyline[1:]):
        b1 = _bucket(p1)
        b2 = _bucket(p2)
        if b1 < b2:
            key = (b1, b2)
        else:
            key = (b2, b1)
        seg_count[key] = seg_count.get(key, 0) + 1

    if not seg_count:
        return 0.0

    total = sum(seg_count.values())
    # 1번만 지나간 선분은 중복이 아님. 2번 이상부터 "왕복"으로 본다.
    overlap = sum(c - 1 for c in seg_count.values() if c > 1)

    return overlap / float(total)


def _curve_penalty(polyline: Polyline) -> float:
    """
    폴리라인에서 '너무 급격하게 꺾인' 부분을 패널티로 환산.
    - 루프가 너무 톱니처럼 삐죽삐죽할 경우 점수에서 깎기 위함.
    """
    if len(polyline) < 3:
        return 0.0

    penalty = 0.0

    for (x1, y1), (x2, y2), (x3, y3) in zip(polyline[:-2], polyline[1:-1], polyline[2:]):
        # 벡터 v1 = p2 - p1, v2 = p3 - p2
        vx1, vy1 = x2 - x1, y2 - y1
        vx2, vy2 = x3 - x2, y3 - y2

        # 길이
        norm1 = math.hypot(vx1, vy1)
        norm2 = math.hypot(vx2, vy2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        # cos(theta) = (v1·v2) / (|v1||v2|)
        dot = vx1 * vx2 + vy1 * vy2
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        angle_deg = math.degrees(math.acos(cos_theta))

        # 180도에 가까우면 직선이므로 패널티 거의 없음.
        # 0도~30도처럼 심하게 꺾이면 패널티 부여.
        if angle_deg < 150:
            penalty += (150 - angle_deg) / 150.0

    return penalty


def polygon_roundness(points: Polyline) -> float:
    """
    폴리라인의 "둥근 정도"를 0~1 사이 값으로 근사.
    1에 가까울수록 원형/정다각형에 가깝고, 0에 가까울수록 선에 가깝다.
    간단히 bounding box 대비 폴리라인의 넓이/길이를 활용.
    """
    if len(points) < 3:
        return 0.0

    # 폴리곤 면적 (Shoelace formula)
    area = 0.0
    xs = [p[1] for p in points]
    ys = [p[0] for p in points]
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    area = abs(area) / 2.0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_area = (max_x - min_x) * (max_y - min_y)
    if bbox_area <= 1e-12:
        return 0.0

    # 박스 안을 얼마나 "채우는지" 비율
    fill_ratio = area / bbox_area  # 0~1

    # 박스가 너무 찌그러진 경우(매우 긴 직사각형)에 대한 보정
    width = max_x - min_x
    height = max_y - min_y
    if width <= 1e-9 or height <= 1e-9:
        return 0.0

    ratio = min(width, height) / max(width, height)  # 0~1

    # fill_ratio와 ratio를 곱해서 사용
    return fill_ratio * ratio


# ---------------------------------------------------------
# 4. Bézier 기반 폴리라인 스무딩 (C 옵션의 핵심 아이디어)
#    - 현재 버전에서는 그래프 기반 루프의 "점수"에 반영되는
#      라운드니스/곡률/오버랩만 조정하고,
#      실제 도로를 벗어나지는 않도록 좌표는 그대로 유지.
# ---------------------------------------------------------


def _ensure_closed(polyline: Polyline, start: LatLng, tol_m: float = 30.0) -> Polyline:
    """
    루프가 충분히 닫혀 있지 않으면 시작점 근처로 보정.
    (현재 그래프 기반 알고리즘은 이미 닫힌 루프를 만드는 편이지만,
     혹시 모를 수 m 이내 허용.)
    """
    if not polyline:
        return polyline

    end = polyline[-1]
    dist = haversine_m(start[0], start[1], end[0], end[1])

    # 이미 충분히 가깝다면 그대로 둔다.
    if dist <= tol_m:
        # 끝점이 약간 틀어져 있더라도, 시작점과 정확히 같게 맞춰주면
        # 시각적으로 "확실히 닫힌 루프"로 보이기 좋다.
        polyline[-1] = (start[0], start[1])
        return polyline

    # 그렇지 않다면, 마지막 점을 시작점으로 강제 스냅
    polyline[-1] = (start[0], start[1])
    return polyline


def _dedupe_consecutive_points(polyline: Polyline) -> Polyline:
    """
    연속해서 같은 좌표가 반복되는 경우 제거 (시각적인 톱니/스파이크 감소).
    """
    if not polyline:
        return polyline

    cleaned = [polyline[0]]
    for p in polyline[1:]:
        if haversine_m(cleaned[-1][0], cleaned[-1][1], p[0], p[1]) > 1e-3:
            cleaned.append(p)
    return cleaned


# ---------------------------------------------------------
# 5. 그래프에서 "루프" 후보 생성 로직
# ---------------------------------------------------------

def _sample_loop_candidates(
    G: nx.MultiDiGraph,
    start_node: int,
    target_m: float,
    n_candidates: int = 400,
) -> List[Polyline]:
    """
    보행 그래프에서 '루프' 후보를 여러 개 샘플링한 뒤
    폴리라인 리스트로 반환한다.
    """
    candidates: List[Polyline] = []

    # 그래프가 너무 작을 수 있으므로, 노드 리스트를 가져온다.
    nodes = list(G.nodes())
    if len(nodes) < 3:
        return []

    # 타겟 거리의 대략 절반~70% 정도 되는 거리까지 "멀어졌다가 돌아오는"
    # 여러 형태의 루프를 시도해본다.
    min_leg = max(200.0, target_m * 0.3)
    max_leg = target_m * 0.7

    for _ in range(n_candidates):
        # 1) 시작점에서 임의의 노드까지 하나의 경로
        #    (start_node -> mid_node)
        mid_node = random.choice(nodes)

        try:
            path1 = nx.shortest_path(
                G, source=start_node, target=mid_node, weight="length"
            )
        except nx.NetworkXNoPath:
            continue

        dist1 = _graph_path_length(G, path1)
        if not (min_leg <= dist1 <= max_leg):
            continue

        # 2) mid_node에서 다시 시작점으로 돌아오는 경로
        try:
            path2 = nx.shortest_path(
                G, source=mid_node, target=start_node, weight="length"
            )
        except nx.NetworkXNoPath:
            continue

        dist2 = _graph_path_length(G, path2)
        total_dist = dist1 + dist2

        # 너무 짧거나 너무 길면 버림 (대략 50%~150% 범위 정도는 허용)
        if not (target_m * 0.5 <= total_dist <= target_m * 1.5):
            continue

        # 중간에서 다시 mid_node로 돌아오면서,
        # 루프 모양을 만들기 위해 노드 중복을 최소화하려 시도할 수 있으나,
        # 여기서는 단순히 그래프가 주는 경로를 사용하고,
        # 나중에 edge overlap으로 평가한다.
        merged_nodes = path1[:-1] + path2
        poly = _graph_path_to_polyline(G, merged_nodes)

        candidates.append(poly)

    return candidates


# ---------------------------------------------------------
# 6. 메인: 루프 생성 generate_area_loop
# ---------------------------------------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[Polyline, Dict[str, Any]]:
    """
    중심 위경도 (lat, lng)와 원하는 거리(km)를 받아
    - 닫힌 루프 형태의 폴리라인(위경도 리스트)
    - 메타데이터(길이, 라운드니스, 오버랩, 스코어 등)를 반환한다.
    """
    # 0. 목표 거리 (m)
    target_m = max(200.0, km * 1000.0)

    # 길이 허용 오차 비율 (±5%)
    LENGTH_TOL_FRAC = 0.05

    # 스코어 가중치 (C 옵션 반영)
    LENGTH_PENALTY_WEIGHT = 10.0     # 길이 오차 패널티
    ROUNDNESS_WEIGHT = 2.5           # 둥글고 넓게 퍼질수록 선호도 강화
    OVERLAP_PENALTY = 6.0            # 같은 길을 왕복하는 비율이 높을수록 강하게 패널티
    CURVE_PENALTY_WEIGHT = 0.3  # 스파이크만 억제, 부드러운 곡선은 허용

    # 1. 그래프 구축
    G = _build_pedestrian_graph(lat, lng, km, network_type="walk")
    start_node = _nearest_node(G, lat, lng)
    start_latlng = (lat, lng)

    # 2. 루프 후보들 샘플링
    candidate_routes = _sample_loop_candidates(G, start_node, target_m, n_candidates=400)

    # 3. 후보 루프들 스코어링
    best_poly: Optional[Polyline] = None
    best_score = -1e18
    best_meta: Dict[str, Any] = {}
    routes_checked = 0
    routes_validated = 0

    for poly in candidate_routes:
        routes_checked += 1

        L = _polyline_length_m(poly)
        err = abs(L - target_m)
        if err > target_m * LENGTH_TOL_FRAC:
            # 길이 허용 범위 밖이면 사용하지 않음
            continue

        routes_validated += 1

        # 시각적/형태적 평가
        r = polygon_roundness(poly)       # 0~1 : 둥글수록 1
        ov = _edge_overlap_ratio(poly)    # 0~1 : 왕복이 많을수록 1
        cp = _curve_penalty(poly)         # 0 이상 : 스파이크 많을수록 큼

        length_pen = err / max(1.0, target_m)

        # '직선 왕복'에 가까운 루프(엣지 중복이 매우 크고, 퍼짐이 작을 경우)는
        # 시각적으로 루프처럼 보이지 않으므로 후보에서 제외한다.
        #   - ov  > 0.95 : 전체 경로의 대부분이 왕복
        #   - r   < 0.40 : 원형/다각형의 퍼짐이 충분히 크지 않음
        if ov > 0.95 and r < 0.40:
            continue

        score = (
            ROUNDNESS_WEIGHT * r
            - OVERLAP_PENALTY * ov
            - CURVE_PENALTY_WEIGHT * cp
            - LENGTH_PENALTY_WEIGHT * length_pen
        )

        if score > best_score:
            best_score = score
            best_poly = poly
            best_meta = {
                "len": L,
                "err": err,
                "roundness": r,
                "overlap": ov,
                "curve_penalty": cp,
                "score": score,
            }

    # 4. 만약 위에서 아무 후보도 선택되지 않았다면,
    #    마지막 fallback: 단순 기하학적 사각형 루프 생성
    used_fallback = False

    if best_poly is None:
        used_fallback = True

        # 간단한 직사각형 루프를 그리되, 시작점 근처에 만들기
        # (이 부분은 기존 v1의 fallback 로직을 그대로 유지)
        # 가로/세로 비율을 1:1~2:1 정도에서 랜덤하게 선택
        aspect = random.uniform(1.0, 2.0)
        # 타겟 거리에 맞춰 둘레가 target_m이 되도록 가로/세로를 결정
        # 2*(w + h) = target_m, w = aspect*h
        # => h = target_m / (2*(aspect + 1)), w = aspect*h
        h = target_m / (2 * (aspect + 1))
        w = aspect * h

        # meters -> degrees 근사 변환 (위도 / 경도)
        lat2m = 111_320.0
        lon2m = 111_320.0 * math.cos(math.radians(lat))

        dlat_h = (h / lat2m)
        dlon_w = (w / lon2m)

        cx, cy = lng, lat

        # 사각형 4점
        p1 = (cy, cx)
        p2 = (cy, cx + dlon_w)
        p3 = (cy + dlat_h, cx + dlon_w)
        p4 = (cy + dlat_h, cx)

        best_poly = [p1, p2, p3, p4, p1]

        L = _polyline_length_m(best_poly)
        err = abs(L - target_m)
        r = polygon_roundness(best_poly)
        ov = _edge_overlap_ratio(best_poly)
        cp = _curve_penalty(best_poly)
        score = (
            ROUNDNESS_WEIGHT * r
            - OVERLAP_PENALTY * ov
            - CURVE_PENALTY_WEIGHT * cp
            - LENGTH_PENALTY_WEIGHT * (err / max(1.0, target_m))
        )

        best_score = score
        best_meta = {
            "len": L,
            "err": err,
            "roundness": r,
            "overlap": ov,
            "curve_penalty": cp,
            "score": score,
        }

    # 5. 루프 형태/시각적 마무리
    best_poly = _dedupe_consecutive_points(best_poly)
    best_poly = _ensure_closed(best_poly, start_latlng, tol_m=30.0)

    # 메타 정보 완성
    L = _polyline_length_m(best_poly)
    err = abs(L - target_m)

    best_meta.update(
        {
            "success": True,
            "length_ok": err <= target_m * LENGTH_TOL_FRAC,
            "used_fallback": used_fallback,
            "valhalla_calls": 0,
            "kakao_calls": 0,
            "routes_checked": len(candidate_routes),
            "routes_validated": routes_validated,
            "km_requested": km,
            "target_m": target_m,
            "message": (
                "최적의 정밀 경로가 도출되었습니다."
                if not used_fallback
                else "요청 거리의 ±5% 이내에 해당하는 보행 루프를 찾지 못해 기하학적 사각형 루프를 사용했습니다."
            ),
        }
    )

    return best_poly, best_meta
