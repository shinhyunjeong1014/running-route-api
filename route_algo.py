import osmnx as ox
import networkx as nx
import math
from functools import lru_cache

###############################################
# 1) 그래프 캐싱
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
    lat_r = round(lat, 3)
    lng_r = round(lng, 3)

    G = _load_graph(lat_r, lng_r)
    nearest = ox.nearest_nodes(G, lng, lat)
    return G, nearest


###############################################
# 2) 단순 루프 생성
###############################################
def simple_loop_route(G, start, km):
    target_m = km * 1000

    # 반경 30~40% 지점 노드를 선택
    dist_map = nx.single_source_dijkstra_path_length(G, start, weight="length", cutoff=target_m * 0.7)
    mid_candidates = [n for n, d in dist_map.items() if 0.3 * target_m < d < 0.6 * target_m]

    if not mid_candidates:
        # fallback
        mid_candidates = list(dist_map.keys())[:30]

    mid = mid_candidates[len(mid_candidates)//2]

    # 코스: start → mid → start
    out_path = nx.shortest_path(G, start, mid, weight="length")
    back_path = nx.shortest_path(G, mid, start, weight="length")

    loop = out_path + back_path[1:]

    # smoothing: 동일 좌표 반복 제거
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
# 5) 턴 분석 (기존 것 그대로 사용)
###############################################
# → 여기서는 기존 polyline_to_turns 그대로 사용
from original_turn_processing import (
    build_turn_by_turn
)
