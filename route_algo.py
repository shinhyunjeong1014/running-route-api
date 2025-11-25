# route_algo.py
import osmnx as ox
import networkx as nx
import math
from functools import lru_cache

###############################################
# 1) 그래프 캐싱
###############################################
@lru_cache(maxsize=32)
def _load_graph(lat_r: float, lng_r: float):
    """
    소수점 3자리(lat/lng) 기준으로 OSMnx 그래프 캐시.
    같은 동네에서는 그래프를 재사용.
    """
    return ox.graph_from_point(
        (lat_r, lng_r),
        dist=1200,          # 반경 약 1.2km
        network_type="walk",
        simplify=True
    )


def get_graph_cached(lat: float, lng: float):
    """
    (lat, lng)에 가장 가까운 보행 네트워크 그래프 + 시작 노드 반환.
    """
    lat_r = round(lat, 3)
    lng_r = round(lng, 3)

    G = _load_graph(lat_r, lng_r)
    s_node = ox.nearest_nodes(G, lng, lat)
    return G, s_node


###############################################
# 2) 경로 길이 계산
###############################################
def path_length(G, path):
    """
    노드 시퀀스(path)에 대한 총 edge length(m) 계산.
    """
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data_dict = G.get_edge_data(u, v)
        if not data_dict:
            continue
        data = list(data_dict.values())[0]
        total += float(data.get("length", 1.0))
    return total


###############################################
# 3) polyline 변환
###############################################
def nodes_to_latlngs(G, path):
    """
    그래프 노드 시퀀스를 [{lat, lng}, ...]로 변환.
    """
    return [{"lat": G.nodes[n]["y"], "lng": G.nodes[n]["x"]} for n in path]


###############################################
# 4) 삼각 루프 기반 경로 생성 (FastLoopRoute v2)
###############################################
def generate_loop_route(G, start, km: float):
    """
    start 노드 기준 삼각형/사다리꼴 루프를 생성.
    - S -> A -> B -> S 구조
    - 길이는 target_km ± 대략 150m 정도.
    """
    target_m = km * 1000.0
    cutoff = target_m * 0.8

    # 1) start에서 일정 거리 범위 안의 후보 노드들
    dist_map = nx.single_source_dijkstra_path_length(
        G, start, weight="length", cutoff=cutoff
    )

    # 2) 25~60% 지점 거리를 가진 노드들을 후보로 사용
    candidates = [
        n for n, d in dist_map.items()
        if 0.25 * target_m < d < 0.60 * target_m
    ]
    if len(candidates) < 2:
        # 후보가 너무 적으면 그냥 앞에서 몇 개 사용
        candidates = list(dist_map.keys())[:50]

    if len(candidates) < 2:
        # 진짜로 후보가 없으면 그냥 시작점만 반환 (극단적 상황)
        return [start]

    # 3) A, B 두 점 선택 (대략 서로 다른 방향이 되도록 멀리 떨어진 것들)
    candidates_sorted = sorted(candidates, key=lambda n: dist_map[n])
    A = candidates_sorted[len(candidates_sorted) // 3]
    B = candidates_sorted[(len(candidates_sorted) * 2) // 3]

    # 4) S -> A -> B -> S 경로
    path1 = nx.shortest_path(G, start, A, weight="length")
    path2 = nx.shortest_path(G, A, B, weight="length")
    path3 = nx.shortest_path(G, B, start, weight="length")

    loop = path1 + path2[1:] + path3[1:]

    # 5) 노이즈 제거 (연속 중복 노드 제거)
    clean = []
    last = None
    for n in loop:
        if n != last:
            clean.append(n)
        last = n

    # 6) 길이 간단 보정 (너무 길거나 짧으면 살짝만 조정)
    clean = _adjust_length_simple(G, clean, target_m)
    return clean


def _adjust_length_simple(G, path, target_m: float, tol_m: float = 200.0):
    """
    아주 단순한 길이 보정:
    - 너무 길면 중간 구간을 조금 단축
    - 너무 짧으면 시작점 근처에 짧은 왕복을 덧붙임
    """
    L = path_length(G, path)

    # 너무 길면: 가운데 구간을 조금 줄여본다
    if L > target_m + tol_m and len(path) > 10:
        i = len(path) // 4
        j = len(path) * 3 // 4
        try:
            sp = nx.shortest_path(G, path[i], path[j], weight="length")
            new_path = path[:i] + sp + path[j + 1 :]
            return new_path
        except Exception:
            return path

    # 너무 짧으면: 시작점 주변 이웃 한두 개 왕복으로 살짝 늘린다
    if L < target_m - tol_m:
        s = path[0]
        neigh = list(G.neighbors(s))
        if neigh:
            k = neigh[0]
            extra = [s, k, s]
            return extra + path

    return path


###############################################
# 5) 좌표 기반 턴/음성 안내 생성
###############################################
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    위경도 좌표 두 점 사이의 해버사인 거리(m).
    """
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _bearing(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """
    A->B 방위각(0~360도).
    """
    dlon = math.radians(b_lng - a_lng)
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.cos(lat2) - math.sin(lat1) * math.sin(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _signed_turn_angle(a: dict, b: dict, c: dict) -> float:
    """
    세 점 a-b-c에 대해 b에서의 회전 각도(부호 포함)를 계산.
    + : 좌회전, - : 우회전
    """
    th1 = _bearing(a["lat"], a["lng"], b["lat"], b["lng"])
    th2 = _bearing(b["lat"], b["lng"], c["lat"], c["lng"])
    # [-180, 180]으로 정규화
    return ((th2 - th1 + 540.0) % 360.0) - 180.0


def _cumulative_distances(polyline):
    dists = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        dists.append(dists[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return dists


def _format_instruction(distance_m: float, turn_type: str) -> str:
    dist_rounded = int(round(distance_m / 10.0) * 10)  # 10m 단위 반올림
    if turn_type == "left":
        return f"{dist_rounded}m 앞 좌회전"
    if turn_type == "right":
        return f"{dist_rounded}m 앞 우회전"
    if turn_type == "straight":
        return f"{dist_rounded}m 직진"
    if turn_type == "uturn":
        return f"{dist_rounded}m 앞 U턴"
    if turn_type == "arrive":
        return "목적지에 도착했습니다"
    return ""


def polyline_to_turns(
    polyline,
    min_step_m: float = 20.0,        # 각도 계산 시 앞/뒤 최소 거리
    min_turn_spacing_m: float = 40.0, # 턴들 사이 최소 간격
    straight_thresh: float = 5.0,    # 이보다 작으면 완전 직진으로 간주
    turn_thresh: float = 15.0,       # 좌/우 회전으로 보는 최소 각도
    uturn_thresh: float = 150.0      # U턴으로 보는 각도
):
    """
    polyline을 기반으로 turn-by-turn 정보를 생성.
    - Out & Back처럼 거의 직선에 가까운 구간에서도 어느 정도 턴이 나오도록
      각도 기준을 완화하고, 거리 기반으로 이전/다음 포인트를 잡는다.
    """
    if not polyline or len(polyline) < 3:
        return []

    n = len(polyline)
    cum = _cumulative_distances(polyline)
    turns = []
    last_turn_idx = 0  # polyline 인덱스 상 마지막 턴 위치

    for i in range(1, n - 1):
        # 1) 이전 포인트 j: i보다 min_step_m 이상 떨어진 지점 중 가장 가까운 인덱스
        j = i - 1
        while j > 0 and (cum[i] - cum[j]) < min_step_m:
            j -= 1

        # 2) 다음 포인트 k: i보다 min_step_m 이상 떨어진 지점 중 가장 가까운 인덱스
        k = i + 1
        while k < n - 1 and (cum[k] - cum[i]) < min_step_m:
            k += 1

        # 유효한 j, k 없으면 스킵
        if j == i or k == i:
            continue

        angle = _signed_turn_angle(polyline[j], polyline[i], polyline[k])
        angle_abs = abs(angle)

        # 거의 직선이면 스킵
        if angle_abs < straight_thresh:
            continue

        # 턴 타입 분류
        if angle_abs >= uturn_thresh:
            t_type = "uturn"
        elif angle_abs >= turn_thresh:
            t_type = "left" if angle > 0 else "right"
        else:
            t_type = "straight"

        # 직전 턴과 너무 가까우면 스킵
        dist_from_last_turn = cum[i] - cum[last_turn_idx]
        if dist_from_last_turn < min_turn_spacing_m:
            continue

        turns.append({
            "lat": polyline[i]["lat"],
            "lng": polyline[i]["lng"],
            "type": t_type,
            "distance": round(dist_from_last_turn, 1),
            "instruction": _format_instruction(dist_from_last_turn, t_type),
        })
        last_turn_idx = i

    # 마지막 도착 안내
    final_dist = cum[-1] - cum[last_turn_idx]
    turns.append({
        "lat": polyline[-1]["lat"],
        "lng": polyline[-1]["lng"],
        "type": "arrive",
        "distance": round(final_dist, 1),
        "instruction": _format_instruction(final_dist, "arrive"),
    })

    return turns


def build_turn_by_turn(polyline, km_requested: float, total_length_m: float = None):
    """
    턴 목록과 요약 메타를 생성.
    """
    turns = polyline_to_turns(polyline)
    if total_length_m is None:
        total_length_m = _cumulative_distances(polyline)[-1] if polyline else 0.0

    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round((total_length_m / 1000.0) * 8.0, 1),
        "turn_count": len([t for t in turns if t.get("type") != "arrive"]),
    }
    return turns, summary
