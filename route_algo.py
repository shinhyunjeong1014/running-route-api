# route_algo.py
import random
import networkx as nx
import osmnx as ox
import math

###############################################
# 기본 유틸
###############################################
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _cumulative_distances(polyline):
    d = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        d.append(d[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return d


def nodes_to_latlngs(G, nodes):
    return [{"lat": G.nodes[n]["y"], "lng": G.nodes[n]["x"]} for n in nodes]


###############################################
# 1) Walker 그래프 생성
###############################################
def build_walk_graph(lat, lng, km):
    for dist in [1500, 2500, 4000]:
        try:
            G = ox.graph_from_point(
                (lat, lng),
                dist=dist,
                network_type="walk",
                simplify=True,
                retain_all=False
            )
        except Exception:
            continue
        if len(G) > 0:
            return G

    raise RuntimeError("보행 네트워크 생성 실패")


###############################################
# 2) FastLoopRoute v3 — 안정형 루프 생성
###############################################
def make_fast_loop(G, start_lat, start_lng, km):
    target = km * 1000
    start = ox.nearest_nodes(G, start_lng, start_lat)

    # -----------------------------------------
    # 1단계: 정상 후보
    # -----------------------------------------
    dist_map = nx.single_source_dijkstra_path_length(G, start, cutoff=target * 0.8)
    cand = [n for n, d in dist_map.items() if target * 0.3 <= d <= target * 0.7]

    if len(cand) >= 2:
        try:
            return _make_triangle(G, start, cand)
        except:
            pass

    # -----------------------------------------
    # 2단계: 후보 조건 완화
    # -----------------------------------------
    cand2 = [n for n, d in dist_map.items() if target * 0.15 <= d <= target * 0.9]

    if len(cand2) >= 2:
        try:
            return _make_triangle(G, start, cand2)
        except:
            pass

    # -----------------------------------------
    # 3단계: 원형 루프 생성
    # -----------------------------------------
    try:
        return _make_circle_loop(G, start, km)
    except:
        pass

    # -----------------------------------------
    # 4단계: 최후 fallback — 왕복 루프라도
    # -----------------------------------------
    try:
        return _make_out_back(G, start, target)
    except:
        pass

    raise RuntimeError("루프 생성 실패")


###############################################
# 삼각 루프 생성
###############################################
def _make_triangle(G, start, cand):
    A = random.choice(cand)
    B = random.choice(cand)

    p1 = nx.shortest_path(G, start, A, weight="length")
    p2 = nx.shortest_path(G, A, B, weight="length")
    p3 = nx.shortest_path(G, B, start, weight="length")

    loop = p1 + p2[1:] + p3[1:]
    coords = nodes_to_latlngs(G, loop)
    length = _cumulative_distances(coords)[-1]
    return coords, length


###############################################
# 원형 루프 생성 (start 주변을 반시계로)
###############################################
def _make_circle_loop(G, start, km):
    steps = 10 + int(km * 3)
    cur = start
    loop = [cur]

    for _ in range(steps):
        neigh = list(G.neighbors(cur))
        if not neigh:
            break
        cur = random.choice(neigh)
        loop.append(cur)

    loop.append(start)
    coords = nodes_to_latlngs(G, loop)
    length = _cumulative_distances(coords)[-1]
    return coords, length


###############################################
# 왕복 fallback (start → mid → start)
###############################################
def _make_out_back(G, start, target_m):
    dist_map = nx.single_source_dijkstra_path_length(G, start, cutoff=target_m)
    far = max(dist_map, key=lambda x: dist_map[x])
    mid = far

    p1 = nx.shortest_path(G, start, mid, weight="length")
    p2 = nx.shortest_path(G, mid, start, weight="length")

    loop = p1 + p2[1:]
    coords = nodes_to_latlngs(G, loop)
    length = _cumulative_distances(coords)[-1]
    return coords, length


###############################################
# 외부에서 호출되는 함수
###############################################
def generate_route(lat, lng, km):
    G = build_walk_graph(lat, lng, km)
    polyline, length_m = make_fast_loop(G, lat, lng, km)
    return polyline, length_m
