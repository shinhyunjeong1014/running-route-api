# route_algo.py
import osmnx as ox
import networkx as nx
import math
from functools import lru_cache

###############################################
# 1) 그래프 캐싱 (성능↑)
###############################################
@lru_cache(maxsize=32)
def _load_graph(lat_round, lng_round):
    return ox.graph_from_point(
        (lat_round, lng_round),
        dist=1200,
        network_type="walk",
        simplify=True
    )

def get_graph_cached(lat, lng):
    # 0.001 → 약 110m 단위 캐싱
    lat_r = round(lat, 3)
    lng_r = round(lng, 3)

    G = _load_graph(lat_r, lng_r)
    nearest = ox.nearest_nodes(G, lng, lat)
    return G, nearest


###############################################
# 2) 단순 루프 생성 (빠르고 안정적)
###############################################
def simple_loop_route(G, start, km):
    target_m = km * 1000

    dist_map = nx.single_source_dijkstra_path_length(
        G, start, weight="length", cutoff=target_m * 0.7
    )

    # 40~60% 지점 후보
    mid_candidates = [
        n for n, d in dist_map.items() if 0.3 * target_m < d < 0.6 * target_m
    ]

    if not mid_candidates:
        mid_candidates = list(dist_map.keys())[:50]

    mid = mid_candidates[len(mid_candidates) // 2]

    # start → mid → start
    out_path = nx.shortest_path(G, start, mid, weight="length")
    back_path = nx.shortest_path(G, mid, start, weight="length")

    loop = out_path + back_path[1:]

    # smoothing
    clean = []
    last = None
    for n in loop:
        if n != last:
            clean.append(n)
        last = n

    return clean


###############################################
# 3) polyline 변환
###############################################
def nodes_to_latlngs(G, path):
    return [{"lat": G.nodes[n]["y"], "lng": G.nodes[n]["x"]} for n in path]


###############################################
# 4) 길이 계산
###############################################
def path_length(G, path):
    total = 0
    for u, v in zip(path[:-1], path[1:]):
        data = list(G.get_edge_data(u, v).values())[0]
        total += data.get("length", 1.0)
    return total


###############################################
# 5) 기존 턴 계산 (오류 제거 버전)
###############################################
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return R * 6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _bearing(y1, x1, y2, x2):
    dlon = math.radians(x2 - x1)
    lat1 = math.radians(y1)
    lat2 = math.radians(y2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.cos(lat2) - math.sin(lat1) * math.sin(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def _bearing_from_coords(a, b):
    return _bearing(a["lat"], a["lng"], b["lat"], b["lng"])

def _signed_turn_angle(a, b, c):
    th1 = _bearing_from_coords(a, b)
    th2 = _bearing_from_coords(b, c)
    return ((th2 - th1 + 540) % 360) - 180

def _cumulative_distances(polyline):
    d = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        d.append(d[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return d

def _format_instruction(distance_m, turn_type):
    if turn_type == "left":
        return f"{int(distance_m)}m 앞 좌회전"
    if turn_type == "right":
        return f"{int(distance_m)}m 앞 우회전"
    if turn_type == "straight":
        return f"{int(distance_m)}m 직진"
    if turn_type == "uturn":
        return f"{int(distance_m)}m 앞 U턴"
    return "목적지에 도착했습니다"

def polyline_to_turns(polyline, straight_thresh=10.0, turn_thresh=18.0, uturn_thresh=150.0):
    if len(polyline) < 3:
        return []

    cum = _cumulative_distances(polyline)
    turns = []
    last_idx = 0

    for i in range(1, len(polyline) - 1):
        angle = _signed_turn_angle(polyline[i - 1], polyline[i], polyline[i + 1])
        a_abs = abs(angle)

        if a_abs < straight_thresh:
            continue

        if a_abs >= uturn_thresh:
            t = "uturn"
        elif a_abs >= turn_thresh:
            t = "left" if angle > 0 else "right"
        else:
            t = "straight"

        dist_to_turn = cum[i] - cum[last_idx]
        turns.append({
            "lat": polyline[i]["lat"],
            "lng": polyline[i]["lng"],
            "type": t,
            "distance": round(dist_to_turn, 1),
            "instruction": _format_instruction(dist_to_turn, t)
        })
        last_idx = i

    # arrive
    final_dist = cum[-1] - cum[last_idx]
    turns.append({
        "lat": polyline[-1]["lat"],
        "lng": polyline[-1]["lng"],
        "type": "arrive",
        "distance": round(final_dist, 1),
        "instruction": "목적지에 도착했습니다"
    })

    return turns

def build_turn_by_turn(polyline, km_requested, total_length_m):
    turns = polyline_to_turns(polyline)
    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round((total_length_m / 1000.0) * 8.0, 1),
        "turn_count": len([t for t in turns if t["type"] != "arrive"])
    }
    return turns, summary
