# route_algo.py
import osmnx as ox
import networkx as nx
import math
from functools import lru_cache

###############################################
# 1) 그래프 캐싱
###############################################
@lru_cache(maxsize=32)
def _load_graph(lat_r, lng_r):
    return ox.graph_from_point(
        (lat_r, lng_r),
        dist=1200,
        network_type="walk",
        simplify=True
    )

def get_graph_cached(lat, lng):
    lat_r = round(lat, 3)
    lng_r = round(lng, 3)
    G = _load_graph(lat_r, lng_r)
    s_node = ox.nearest_nodes(G, lng, lat)
    return G, s_node


###############################################
# 2) 거리 계산
###############################################
def path_length(G, path):
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = list(G.get_edge_data(u, v).values())[0]
        total += data.get("length", 1.0)
    return total


###############################################
# 3) polyline 변환
###############################################
def nodes_to_latlngs(G, path):
    return [{"lat": G.nodes[n]["y"], "lng": G.nodes[n]["x"]} for n in path]


###############################################
# 4) 삼각형 기반 Loop Route 생성
###############################################
def generate_loop_route(G, start, km):
    target_m = km * 1000
    cutoff = target_m * 0.8

    # 1) start에서 일정 거리 이상 떨어진 후보군
    dist_map = nx.single_source_dijkstra_path_length(G, start, weight="length", cutoff=cutoff)
    candidates = [n for n, d in dist_map.items() if 0.25 * target_m < d < 0.55 * target_m]

    if len(candidates) < 2:
        candidates = list(dist_map.keys())[:50]

    # 점 2개 선택: A, B
    A = candidates[len(candidates) // 3]
    B = candidates[len(candidates) * 2 // 3]

    # 2) start → A
    path1 = nx.shortest_path(G, start, A, weight="length")
    # 3) A → B
    path2 = nx.shortest_path(G, A, B, weight="length")
    # 4) B → start
    path3 = nx.shortest_path(G, B, start, weight="length")

    loop = path1 + path2[1:] + path3[1:]

    # smoothing
    clean = []
    last = None
    for n in loop:
        if n != last:
            clean.append(n)
        last = n

    # 거리 조정 (±150m)
    L = path_length(G, clean)
    if abs(L - target_m) > 150:
        clean = adjust_length(G, clean, target_m)

    return clean


###############################################
# 5) 거리 보정
###############################################
def adjust_length(G, path, target_m):
    L = path_length(G, path)

    # 너무 길면 단축
    if L > target_m + 150:
        mid1 = path[len(path) // 4]
        mid2 = path[len(path) // 2]
        short = nx.shortest_path(G, mid1, mid2, weight="length")
        new = path[:len(path)//4] + short + path[(len(path)//2)+1:]
        return new

    # 너무 짧으면 패딩(스타트 근처 왕복)
    if L < target_m - 150:
        s = path[0]
        neigh = list(G.neighbors(s))[:3]
        if neigh:
            ext = [s] + neigh + [s]
            return ext + path

    return path


###############################################
# 6) 턴 계산
###############################################
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _bearing(a_lat, a_lng, b_lat, b_lng):
    dlon = math.radians(b_lng - a_lng)
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1)*math.cos(lat2) - math.sin(lat1)*math.sin(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def _signed_turn_angle(a, b, c):
    th1 = _bearing(a["lat"], a["lng"], b["lat"], b["lng"])
    th2 = _bearing(b["lat"], b["lng"], c["lat"], c["lng"])
    return ((th2 - th1 + 540) % 360) - 180

def _cumulative(polyline):
    d = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        d.append(d[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return d

def _instruction(dist, t):
    m = int(dist)
    if t == "left": return f"{m}m 앞 좌회전"
    if t == "right": return f"{m}m 앞 우회전"
    if t == "straight": return f"{m}m 직진"
    if t == "uturn": return f"{m}m 앞 U턴"
    return "목적지에 도착했습니다"

def polyline_to_turns(polyline):
    if len(polyline) < 3:
        return []

    cum = _cumulative(polyline)
    turns = []
    last_i = 0

    for i in range(1, len(polyline) - 1):
        angle = _signed_turn_angle(polyline[i - 1], polyline[i], polyline[i + 1])
        a_abs = abs(angle)

        if a_abs < 10:
            continue
        elif a_abs >= 150:
            t = "uturn"
        elif a_abs >= 20:
            t = "left" if angle > 0 else "right"
        else:
            t = "straight"

        dist = cum[i] - cum[last_i]
        turns.append({
            "lat": polyline[i]["lat"],
            "lng": polyline[i]["lng"],
            "type": t,
            "distance": round(dist, 1),
            "instruction": _instruction(dist, t)
        })
        last_i = i

    final_dist = cum[-1] - cum[last_i]
    turns.append({
        "lat": polyline[-1]["lat"],
        "lng": polyline[-1]["lng"],
        "type": "arrive",
        "distance": final_dist,
        "instruction": "목적지에 도착했습니다"
    })

    return turns

def build_turn_by_turn(polyline, km_requested, total_length_m):
    turns = polyline_to_turns(polyline)
    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round((total_length_m/1000)*8.0, 1),
        "turn_count": len([t for t in turns if t["type"] != "arrive"])
    }
    return turns, summary
