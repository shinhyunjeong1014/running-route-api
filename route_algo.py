# route_algo.py
import random
import networkx as nx
import osmnx as ox
from itertools import islice
import math

###############################################
# OSM 그래프 구축
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

        if len(G) == 0:
            continue
        return G

    raise RuntimeError("보행 네트워크 생성 실패")


###############################################
# 유틸
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
# 루프 생성 (FastLoopRoute v2)
###############################################
def make_fast_loop(G, start_lat, start_lng, km):
    target = km * 1000
    tol = 150

    # 시작 노드
    start = ox.nearest_nodes(G, start_lng, start_lat)

    # 후보 노드 (중간 지점)
    dist_map = nx.single_source_dijkstra_path_length(G, start, cutoff=target * 0.7)

    cand = [n for n, d in dist_map.items() if 0.3 * target <= d <= 0.7 * target]
    if len(cand) < 2:
        raise RuntimeError("후보 부족")

    # 2개 중간지점을 골라 삼각 루프 구성
    A = random.choice(cand)
    B = random.choice(cand)

    p1 = nx.shortest_path(G, start, A, weight="length")
    p2 = nx.shortest_path(G, A, B, weight="length")
    p3 = nx.shortest_path(G, B, start, weight="length")

    loop = p1 + p2[1:] + p3[1:]

    coords = nodes_to_latlngs(G, loop)
    length_m = _cumulative_distances(coords)[-1]

    return coords, length_m


###############################################
# 거리 조정 보정
###############################################
def fix_length(polyline, target_m, tol=150):
    from random import randint
    # 가벼운 overshoot correction 옵션
    return polyline


###############################################
# 외부에서 불러가는 함수
###############################################
def generate_route(lat, lng, km):
    G = build_walk_graph(lat, lng, km)
    polyline, length_m = make_fast_loop(G, lat, lng, km)
    return polyline, length_m
