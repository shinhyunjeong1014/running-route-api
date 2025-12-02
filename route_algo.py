import math
import random
import time
from typing import List, Tuple, Dict, Any

import networkx as nx
import osmnx as ox


# ==========================
# JSON-safe 변환 유틸 (유지)
# ==========================
def safe_float(x: Any):
    """NaN / Inf 를 JSON에서 허용 가능한 값(None)으로 변환."""
    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x):
            return None
    return x


def safe_int(x: Any):
    """INT 변환 시 에러나면 None."""
    try:
        return int(x)
    except Exception:
        return None


def safe_bool(x: Any):
    """Bool 변환 유틸."""
    try:
        return bool(x)
    except Exception:
        return None


def safe_str(x: Any):
    """문자열 변환 유틸."""
    try:
        return str(x)
    except Exception:
        return ""


def safe_list(xs: Any):
    """리스트 변환 유틸."""
    if isinstance(xs, list):
        return xs
    return []


def safe_dict(d: Any):
    """딕셔너리 변환 유틸."""
    if isinstance(d, dict):
        return d
    return {}


# ==========================
# 거리/지오메트리 유틸
# ==========================
def haversine(lat1, lon1, lat2, lon2):
    """위경도 두 점 사이의 거리(m)."""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2
    ) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline: List[Tuple[float, float]]) -> float:
    """단순 polyline의 총 길이(m). FastAPI 쪽에서도 호출하므로 export 유지."""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lon1, lat2, lon2)
    return total


# ==========================
# OSMnx / Graph 유틸
# ==========================
def _build_pedestrian_graph(
    lat: float,
    lng: float,
    km: float,
    network_type: str = "walk",
) -> nx.MultiDiGraph:
    """
    OSMnx 기반 보행자용 그래프 생성.

    - network_type = 'walk'
    - custom_filter 로 보행/생활도로만 남기고, 자동차 전용/고속도로 등은 제외.
    """
    # 러닝 거리의 1.5배 ~ 2배 정도를 커버하는 bounding radius (m)
    # 너무 작으면 후보 경로가 부족, 너무 크면 연산량 폭증
    radius_m = max(500, int(km * 800))

    # OSMnx용 custom_filter
    #  - 포함: footway, path, pedestrian, living_street, residential, service, track, steps, sidewalk, cycleway, alley
    #  - 제외: motorway, trunk, primary 등 자동차 전용/고속도로
    custom_filter = (
        '["highway"~"footway|path|pedestrian|living_street|residential|service|track|steps|sidewalk|cycleway|alley"]'
        '["area"!~"yes"]["motor_vehicle"!~"no"]["service"!~"parking|driveway|private"]'
    )

    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type=network_type,
        custom_filter=custom_filter,
        simplify=True,
    )

    # edge length 보정 (없으면 osmnx 쪽에서 이미 length를 계산해주지만, 안전하게 한 번 더)
    G = ox.add_edge_lengths(G)
    return G


def _nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int:
    """주어진 위경도와 가장 가까운 노드 id."""
    return ox.distance.nearest_nodes(G, lng, lat)


def _path_length(G: nx.MultiDiGraph, path: List[int]) -> float:
    """그래프 상 경로(path)의 길이(m)."""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        # 멀티엣지 중 최단 edge 선택
        best_len = None
        for key in data:
            length = data[key].get("length", None)
            if length is None:
                continue
            if best_len is None or length < best_len:
                best_len = length
        if best_len is None:
            # length 정보가 없으면 대략적인 haversine
            y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
            y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
            best_len = haversine(y1, x1, y2, x2)
        total += best_len
    return total


def _nodes_to_polyline(G: nx.MultiDiGraph, path: List[int]) -> List[Tuple[float, float]]:
    """노드 id 리스트를 (lat, lng) polyline으로 변환."""
    polyline: List[Tuple[float, float]] = []
    for nid in path:
        node = G.nodes[nid]
        polyline.append((float(node["y"]), float(node["x"])))
    return polyline


# ==========================
# 라인 모양/품질 평가 유틸
# ==========================
def polygon_roundness(polyline: List[Tuple[float, float]]) -> float:
    """
    polyline을 2D 평면으로 투영 후, "얼마나 원형/루프 형태에 가까운지"를 0~1 범위로 반환.

    - bounding box가 너무 길쭉하면 (직선형/로드형) roundness가 낮게 나옴.
    - 면적이 거의 없는 경우(완전한 직선)는 0에 가까움.
    """
    if len(polyline) < 4:
        return 0.0

    xs = [p[1] for p in polyline]
    ys = [p[0] for p in polyline]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return 0.0

    # bbox 대비 궤적 길이 비율, aspect ratio 등을 함께 고려
    aspect = min(width, height) / max(width, height)
    perimeter_box = 2 * (width + height)
    perimeter_path = 0.0
    for (y1, x1), (y2, x2) in zip(polyline[:-1], polyline[1:]):
        perimeter_path += math.hypot(x2 - x1, y2 - y1)

    if perimeter_box <= 0:
        return 0.0

    # 루프형일수록 path 길이가 bbox 둘레와 비슷해짐
    ratio = perimeter_path / perimeter_box
    # ratio가 너무 크면 지그재그, 너무 작으면 직선형이므로 1 ~ 3 사이 정도가 이상적
    # 이를 0~1 범위로 squashing
    ratio_score = math.exp(-((ratio - 2.0) ** 2) / 4.0)

    # aspect 역시 1에 가까울수록 정원/정사각 루프
    aspect_score = aspect

    return max(0.0, min(1.0, 0.5 * ratio_score + 0.5 * aspect_score))


def _edge_overlap_fraction(path: List[int]) -> float:
    """
    path 내에서 동일 edge를 몇 번이나 왕복하는지 측정.

    - 간단히 (u, v)와 (v, u)를 같은 edge로 보고,
    - 등장 횟수 > 1 인 edge 비율을 반환.
    """
    if len(path) < 2:
        return 0.0
    edge_counts: Dict[Tuple[int, int], int] = {}
    total_edges = 0
    for u, v in zip(path[:-1], path[1:]):
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        edge_counts[key] = edge_counts.get(key, 0) + 1
        total_edges += 1
    if total_edges == 0:
        return 0.0
    overlapped = sum(1 for c in edge_counts.values() if c > 1)
    return overlapped / total_edges


def _curve_penalty(path: List[int], G: nx.MultiDiGraph) -> float:
    """
    path 상에서 급격한 방향 전환(예: 거의 U턴, 90도 꺾임 등)을 얼마나 많이 하는지.

    - 각 세 점이 이루는 각도의 cos값을 이용해,
    - cos(theta)가 -1에 가까우면 U턴, 0이면 직각, 1이면 직선.
    - 너무 많이 꺾이는 경우가 많으면 penalty 증가.
    """
    if len(path) < 3:
        return 0.0

    def get_xy(nid: int):
        node = G.nodes[nid]
        return float(node["x"]), float(node["y"])

    penalty = 0.0
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        x1, y1 = get_xy(a)
        x2, y2 = get_xy(b)
        x3, y3 = get_xy(c)

        v1x, v1y = x1 - x2, y1 - y2
        v2x, v2y = x3 - x2, y3 - y2
        norm1 = math.hypot(v1x, v1y)
        norm2 = math.hypot(v2x, v2y)
        if norm1 == 0 or norm2 == 0:
            continue
        cos_theta = (v1x * v2x + v1y * v2y) / (norm1 * norm2)
        # 수치 안정성
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)

        # pi(180도)에 가까울수록 U턴, pi/2 근처면 직각, 0이면 직선
        # U턴 및 직각 모두 penalty로 잡되, 살짝 꺾이는 건 용인
        if theta > math.radians(135):  # 거의 U턴
            penalty += 2.0
        elif theta > math.radians(100):  # 꽤 많이 꺾임
            penalty += 1.0
    return penalty


# ==========================
# PSP2 + 로드/디튜어 기반 루트 생성
# ==========================
def generate_area_loop(lat: float, lng: float, km: float):
    """
    PSP2 논문 아이디어를 참고한 보행 루프 생성 알고리즘.

    - 중심점(start)을 기준으로 여러 방향(로드)을 뻗어 보고,
    - 각 방향 끝에서 좌/우 측면(디튜어)을 붙여 PSP2-style 라우트를 생성,
    - shape(루프 형태), overlap(왕복 정도), curve(급격한 턴), length(거리 오차)를 모두 고려해 스코어링.

    Option B:
      - 동일 edge를 많이 왕복하는 "직선형 rod"를 강하게 배제(overlap > 0.6이면 바로 탈락)
      - 스코어링에서 OVERLAP_PENALTY, CURVE_PENALTY, LENGTH_PENALTY 가중치 재조정
    """
    t0 = time.time()
    target_m = km * 1000.0

    # 1. 보행자 그래프 생성
    G = _build_pedestrian_graph(lat, lng, km, network_type="walk")
    undirected = G.to_undirected()

    # 2. 시작 노드 (실제 요청 좌표와 최대한 가깝게 유지)
    start_node = _nearest_node(G, lat, lng)

    # 3. 로드 후보 방향(12방향) 설정
    num_rods = 12
    rod_angles = [2 * math.pi * i / num_rods for i in range(num_rods)]

    # 4. 로드 길이 범위 (왕복 + 디튜어까지 고려해서 대략 타겟의 40~60% 수준)
    min_rod = 0.4 * target_m / 2  # 왕복으로 두 배가 되니 /2
    max_rod = 0.6 * target_m / 2

    # 5. 디튜어 길이 범위 (전체 루프의 20~40% 정도를 디튜어로)
    min_detour = 0.2 * target_m
    max_detour = 0.5 * target_m

    # 후보 엔드포인트 + 디튜어 시작점 캐싱
    candidate_nodes: List[int] = []

    # 라우트 탐색 횟수 제한
    MAX_ROUTE_TRIES = 300

    # [Option B] 스코어링 가중치 (길이 정확도는 유지하되, 루프 형태를 더 강하게 선호)
    ROUNDNESS_WEIGHT = 3.0          # 둥근 루프(원형/타원형)에 가산점
    OVERLAP_PENALTY = 7.0           # 동일 edge 왕복(rod) 강한 감점
    CURVE_PENALTY_WEIGHT = 0.7      # 급격한 꺾임(지그재그) 완화
    LENGTH_PENALTY_WEIGHT = 7.0     # 길이 오차도 중요하지만, 루프형태와 균형

    best_score = -1e18
    best_poly = None
    best_meta = None

    routes_checked = 0
    routes_validated = 0

    # ===== 로드(반직선) 후보 찾기 =====
    # start 주변에서 여러 방향으로 "타겟 거리의 절반 정도"를 가는 경로를 찾고,
    # 그 끝점들을 디튜어 시작점으로 사용.
    for angle in rod_angles:
        # 각 방향마다 랜덤하게 몇 번 시도
        for _ in range(4):
            # 대략적인 목표 로드 길이
            target_rod_len = random.uniform(min_rod, max_rod)

            # angle 방향으로 (가상의 목표점) 찍기
            dy = math.cos(angle) * (target_rod_len / 111000.0)
            dx = math.sin(angle) * (target_rod_len / (111000.0 * math.cos(math.radians(lat))))
            approx_lat = lat + dy
            approx_lng = lng + dx

            try:
                end_node = _nearest_node(G, approx_lat, approx_lng)
            except Exception:
                continue

            # start -> end_node 최단 경로
            try:
                path_rod = nx.shortest_path(G, start_node, end_node, weight="length")
            except Exception:
                continue

            rod_len = _path_length(G, path_rod)
            if rod_len < min_rod or rod_len > max_rod:
                # 로드 길이가 너무 짧거나/길면 패스
                continue

            candidate_nodes.append(end_node)

    # 중복 제거
    candidate_nodes = list(dict.fromkeys(candidate_nodes))

    # ===== 디튜어 + 반환 루프 생성 =====
    # 각 엔드포인트에서 측면으로 꺾어 나가는 디튜어를 만들고, 다시 start로 돌아오는 루프를 구성.
    for endpoint in candidate_nodes:
        if routes_checked >= MAX_ROUTE_TRIES:
            break

        routes_checked += 1

        # 1) start -> endpoint (로드)
        try:
            path_rod = nx.shortest_path(G, start_node, endpoint, weight="length")
        except Exception:
            continue
        rod_len = _path_length(G, path_rod)

        # 2) endpoint에서 "옆으로" 빠져 나가는 디튜어 candidate 노드 찾기
        #    - endpoint에서 일정 거리 범위 내에 있는 노드들 중,
        #      start로 직행하는 방향과 다른 쪽으로 가는 방향을 우선적으로 고른다.
        try:
            neighbors = list(G.neighbors(endpoint))
        except Exception:
            continue

        # endpoint 근처에서 출발하여, target_m - 2*rod_len 정도 되는 길이의 디튜어를 찾는다.
        # (전체: start->endpoint + detour + endpoint->start)
        remaining = target_m - 2 * rod_len
        if remaining < min_detour or remaining > max_detour:
            # rod가 이미 너무 크거나 작아서 적당한 디튜어를 붙이기 어렵다면 패스
            continue

        for neigh in neighbors:
            if routes_checked >= MAX_ROUTE_TRIES:
                break

            # 간단히 endpoint->neigh 로 시작하는 디튜어의 최단경로를 random node까지 확장해본다.
            # detour_end_candidates: Graph 내 임의 노드 일부 샘플
            detour_end_candidates = random.sample(
                list(G.nodes), min(20, G.number_of_nodes())
            )

            for det_end in detour_end_candidates:
                try:
                    path_detour = nx.shortest_path(G, neigh, det_end, weight="length")
                except Exception:
                    continue
                detour_len = _path_length(G, path_detour)

                # 디튜어 길이 필터
                if detour_len < 0.5 * remaining or detour_len > 1.5 * remaining:
                    continue

                # detour 끝에서 다시 start로
                try:
                    path_back = nx.shortest_path(G, det_end, start_node, weight="length")
                except Exception:
                    continue
                back_len = _path_length(G, path_back)

                total_len = rod_len + detour_len + back_len
                length_m = total_len

                # polyline 구성
                full_nodes = path_rod + path_detour + path_back
                # 중복된 연속 노드 제거
                compressed_nodes = []
                for nid in full_nodes:
                    if not compressed_nodes or compressed_nodes[-1] != nid:
                        compressed_nodes.append(nid)

                full_nodes = compressed_nodes
                polyline = _nodes_to_polyline(G, full_nodes)

                err = abs(length_m - target_m)
                roundness = polygon_roundness(polyline)
                overlap = _edge_overlap_fraction(full_nodes)
                curve_penalty = _curve_penalty(full_nodes, undirected)

                # [Option B] 강제 필터: 왕복형(직선형) 경로 제거
                # edge overlap 비율이 0.6을 넘으면 대부분 같은 길을 왕복하는 형태라서 제외
                if overlap > 0.6:
                    continue

                length_ok = err <= 99.0

                # 스코어 계산
                # - roundness: 높을수록 좋음
                # - overlap: 낮을수록 좋음
                # - curve_penalty: 적당한 굴곡은 허용하되 과도한 U턴/지그재그는 감점
                # - length_pen: 타겟과의 상대 오차
                length_pen = (err / target_m) if target_m > 0 else 1.0

                score = (
                    ROUNDNESS_WEIGHT * roundness
                    - OVERLAP_PENALTY * overlap
                    - CURVE_PENALTY_WEIGHT * curve_penalty
                    - LENGTH_PENALTY_WEIGHT * length_pen
                )

                meta = {
                    "len": safe_float(length_m),
                    "err": safe_float(err),
                    "roundness": safe_float(roundness),
                    "overlap": safe_float(overlap),
                    "curve_penalty": safe_float(curve_penalty),
                    "score": safe_float(score),
                    "success": safe_bool(length_ok),
                }

                routes_validated += 1

                if score > best_score:
                    best_score = score
                    best_poly = polyline
                    best_meta = meta

    # ===== 최종 선택 / Fallback 처리 =====
    elapsed = time.time() - t0

    # fallback 여부 판단
    if best_poly is None:
        # 그래프 기반 후보가 하나도 없으면, 기하학적 정사각형 루프 사용
        fallback_poly = _square_fallback_loop(lat, lng, target_m)
        fallback_len = polyline_length_m(fallback_poly)
        fallback_err = abs(fallback_len - target_m)
        fallback_round = polygon_roundness(fallback_poly)

        meta = {
            "len": safe_float(fallback_len),
            "err": safe_float(fallback_err),
            "roundness": safe_float(fallback_round),
            "overlap": 0.0,
            "curve_penalty": 0.0,
            "score": safe_float(-9999.0),
            "success": False,
            "used_fallback": True,
            "routes_checked": safe_int(routes_checked),
            "routes_validated": safe_int(routes_validated),
            "km_requested": safe_float(km),
            "target_m": safe_float(target_m),
            "time_s": safe_float(elapsed),
            "message": "요청 거리의 ±100m 이내에 해당하는 보행 루프를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        }
        return fallback_poly, meta

    # 그래프 기반 루트가 존재하는 경우
    assert best_meta is not None
    best_meta = dict(best_meta)  # 복사

    best_meta.update(
        {
            "used_fallback": False,
            "routes_checked": safe_int(routes_checked),
            "routes_validated": safe_int(routes_validated),
            "km_requested": safe_float(km),
            "target_m": safe_float(target_m),
            "time_s": safe_float(elapsed),
            "message": (
                "최적의 정밀 경로가 도출되었습니다."
                if best_meta.get("success", False)
                else "길이 오차가 다소 있지만, 형태적으로 가장 우수한 루프를 반환합니다."
            ),
        }
    )

    return best_poly, best_meta


# ==========================
# 기하학적 정사각형 fallback
# ==========================
def _square_fallback_loop(
    lat: float,
    lng: float,
    target_m: float,
) -> List[Tuple[float, float]]:
    """
    그래프 기반 루프를 찾지 못했을 때 사용하는 정사각형 fallback 경로.

    - 실제 도로를 따르지 않고, 단순히 지구 곡률을 무시한 근사 좌표를 생성.
    - 실제 서비스에서는 이 fallback 빈도를 줄이는 것이 목표.
    """
    side = target_m / 4.0  # 한 변의 길이 (m)

    # 위도/경도 기준 근사 변환
    dlat = (side / 111000.0)
    dlng = side / (111000.0 * math.cos(math.radians(lat)))

    p1 = (lat, lng)
    p2 = (lat + dlat, lng)
    p3 = (lat + dlat, lng + dlng)
    p4 = (lat, lng + dlng)

    return [p1, p2, p3, p4, p1]
