# route_algo.py
import random
import networkx as nx
import osmnx as ox
import math
import logging

logging.basicConfig(level=logging.INFO)
TARGET_RANGE_M = 250 # 목표 거리 ± 250m 허용 (예: 2km 요청 시 1750m ~ 2250m)

###############################################
# 기본 유틸리티
###############################################
def haversine_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이의 거리를 미터(m) 단위로 계산합니다 (Haversine 공식)."""
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _cumulative_distances(polyline):
    """경로(polyline)를 따라 누적 거리를 계산합니다."""
    d = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        d.append(d[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return d


def nodes_to_latlngs(G, nodes):
    """OSMnx 노드 ID 리스트를 {lat, lng} 좌표 리스트로 변환합니다."""
    return [{"lat": G.nodes[n]["y"], "lng": G.nodes[n]["x"]} for n in nodes]


def _path_length(G, path):
    """NetworkX 경로 리스트의 총 길이를 계산합니다."""
    length = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        length += G.get_edge_data(u, v)[0]['length']
    return length


###############################################
# 1) Walker 그래프 생성 (Graph Retrieval)
###############################################
def build_walk_graph(lat, lng, km):
    """주어진 좌표와 목표 거리(km)에 맞는 보행 네트워크 그래프를 생성합니다."""
    # 목표 거리에 따라 검색 반경을 조정 (최대 1.5배)
    base_dist = km * 1000 * 1.5
    search_dists = [base_dist, base_dist * 1.5, base_dist * 2.0, 5000]

    for dist in search_dists:
        try:
            logging.info(f"OSMnx Graph 생성 시도: 반경 {dist:.0f}m")
            # 보행 네트워크, 복잡도 낮추기 위해 simplify=True
            G = ox.graph_from_point(
                (lat, lng),
                dist=dist,
                network_type="walk",
                simplify=True,
                retain_all=False,
                timeout=10,
                memory=None
            )
            G = ox.add_edge_speeds(G)
            G = ox.add_travel_times(G)
            
        except Exception:
            continue
        
        # 최소 노드 수(예: 100개) 이상이거나, 마지막 시도라면 반환
        if len(G.nodes) > 100 or dist == search_dists[-1]:
             logging.info(f"그래프 생성 성공: 노드 수 {len(G.nodes)}, 반경 {dist:.0f}m")
             return G

    raise RuntimeError("보행 네트워크 생성 실패: 주변 지역의 도로 데이터가 부족합니다.")


###############################################
# 3) 경로 길이 보정 (Route Adjustment)
###############################################
def _adjust_route_length(G, path, target_m):
    """
    경로의 길이가 목표 범위(target_m ± TARGET_RANGE_M)에 맞도록 보정합니다.
    (Simple Padding Strategy: 경로의 끝 노드 주변을 왔다 갔다 하도록 추가)
    """
    current_length = _path_length(G, path)
    if current_length >= target_m - TARGET_RANGE_M and current_length <= target_m + TARGET_RANGE_M:
        return path, current_length # 이미 범위 내

    diff_m = target_m - current_length
    
    # 1. 길이가 부족한 경우 (diff_m > TARGET_RANGE_M)
    if diff_m > TARGET_RANGE_M:
        logging.info(f"경로 부족: {current_length:.0f}m. {diff_m:.0f}m 보정 필요.")
        
        # 끝 노드를 출발점으로 설정하여 주변을 맴도는 경로 추가 시도
        end_node = path[-1]
        
        # 끝 노드에서 target_m의 절반 정도까지 갈 수 있는 노드를 찾음
        cutoff_m = min(diff_m * 1.5, target_m * 0.5) 
        try:
            sub_dist_map = nx.single_source_dijkstra_path_length(G, end_node, cutoff=cutoff_m)
            cand_nodes = list(sub_dist_map.keys())
            
            if len(cand_nodes) > 1:
                # 중간 지점 설정 (무작위 또는 가장 먼 노드)
                mid_node = random.choice(cand_nodes[1:])
                
                # 끝 노드 -> 중간 지점 -> 끝 노드 왕복 경로 추가
                p_out = nx.shortest_path(G, end_node, mid_node, weight="length")
                p_back = nx.shortest_path(G, mid_node, end_node, weight="length")
                
                added_path = p_out + p_back[1:]
                
                # 추가된 경로의 길이를 확인하고 반영
                added_length = _path_length(G, added_path)
                
                # 최종적으로 경로를 추가
                new_path = path[:-1] + added_path
                new_length = current_length + added_length
                logging.info(f"패딩 추가: {added_length:.0f}m 추가됨. 최종 길이 {new_length:.0f}m")
                return new_path, new_length

        except Exception as e:
            logging.warning(f"경로 보정(패딩) 실패: {e}. 현재 경로를 그대로 반환.")
            
    # 2. 길이가 너무 긴 경우 (current_length > target_m + TARGET_RANGE_M)
    # 현재는 길이 축소 로직을 넣지 않고, 부족한 경우만 보정하여 실패율을 낮춥니다.
    # 추후 구현: 경로가 길면 끝 부분을 잘라내고 마지막 노드를 시작 노드로 연결
    
    return path, current_length


###############################################
# 2) FastLoopRoute v4 - 다단계 루프 생성
###############################################
def make_loop_route_v4(G, start_lat, start_lng, km):
    """
    FastLoopRoute v4 알고리즘: 다단계 루프 생성 및 거리 보정.
    1. 삼각 루프(3-point)
    2. 다각 루프(4-point)
    3. 왕복 기반 루프 (Out-and-Back Loop)
    4. 원형 루프 (Fallback Random Walk)
    """
    target_m = km * 1000
    start = ox.nearest_nodes(G, start_lng, start_lat)
    
    # 목표 범위 설정
    target_min = target_m - TARGET_RANGE_M
    target_max = target_m + TARGET_RANGE_M

    def try_route(loop_path, method_name):
        """생성된 경로가 목표 범위 내에 있는지 확인하고 보정 후 반환하는 헬퍼 함수"""
        if not loop_path or len(loop_path) < 3:
            return None
        
        # 1. 경로 길이 확인
        initial_length = _path_length(G, loop_path)
        
        # 2. 거리 보정 (범위 충족 시에는 보정 불필요)
        if initial_length < target_min or initial_length > target_max:
            loop_path, final_length = _adjust_route_length(G, loop_path, target_m)
        else:
            final_length = initial_length
            
        # 3. 최종 길이 검사
        if final_length >= target_min and final_length <= target_max:
            logging.info(f"경로 성공: {method_name}, 길이 {final_length:.0f}m")
            return nodes_to_latlngs(G, loop_path), final_length, method_name
        
        logging.warning(f"경로 실패: {method_name}, 최종 길이 {final_length:.0f}m (목표 {target_min:.0f}~{target_max:.0f}m)")
        return None


    # ----------------------------------------------------
    # 1단계: 삼각 루프 (3-Point Loop)
    # ----------------------------------------------------
    logging.info("1단계: 삼각 루프 (3-Point) 시도")
    try:
        # start에서 목표 거리의 1/3 ~ 1/2 지점에 있는 노드를 후보로 선택
        cutoff = target_m * 0.7 
        dist_map = nx.single_source_dijkstra_path_length(G, start, cutoff=cutoff, weight="length")
        
        # A, B 노드는 목표 거리의 1/4 ~ 2/3 지점에 있는 노드를 선호
        cand_min = target_m * 0.25
        cand_max = target_m * 0.66
        cand = [n for n, d in dist_map.items() if cand_min <= d <= cand_max]

        if len(cand) >= 2:
            A, B = random.sample(cand, 2)
            p1 = nx.shortest_path(G, start, A, weight="length")
            p2 = nx.shortest_path(G, A, B, weight="length")
            p3 = nx.shortest_path(G, B, start, weight="length")
            
            loop_path = p1 + p2[1:] + p3[1:]
            result = try_route(loop_path, "Triangle_3P")
            if result: return result
    except Exception as e:
        logging.debug(f"삼각 루프 실패: {e}")

    # ----------------------------------------------------
    # 2단계: 다각 루프 (4-Point Multi-Loop)
    # ----------------------------------------------------
    logging.info("2단계: 다각 루프 (4-Point) 시도")
    try:
        # A, B 노드 후보 재사용 (1단계와 동일 범위)
        if 'cand' in locals() and len(cand) >= 3:
            A, B, C = random.sample(cand, 3)
            
            p1 = nx.shortest_path(G, start, A, weight="length")
            p2 = nx.shortest_path(G, A, B, weight="length")
            p3 = nx.shortest_path(G, B, C, weight="length")
            p4 = nx.shortest_path(G, C, start, weight="length")

            loop_path = p1 + p2[1:] + p3[1:] + p4[1:]
            result = try_route(loop_path, "MultiLoop_4P")
            if result: return result
    except Exception as e:
        logging.debug(f"다각 루프 실패: {e}")

    # ----------------------------------------------------
    # 3단계: 왕복 기반 루프 (Out-and-Back Loop)
    # ----------------------------------------------------
    # 목표 거리의 1/2 지점까지 갔다가 돌아오는 경로를 생성 후 보정
    logging.info("3단계: 왕복 기반 루프 시도")
    try:
        # 목표 지점 찾기: target_m의 40~60% 범위에서 가장 먼 노드를 선택
        half_target = target_m / 2
        cutoff = half_target * 1.5
        dist_map = nx.single_source_dijkstra_path_length(G, start, cutoff=cutoff, weight="length")
        
        # 가장 먼 노드 (Mid-Point) 찾기
        valid_mid_nodes = [n for n, d in dist_map.items() if half_target * 0.8 <= d <= half_target * 1.2]

        if valid_mid_nodes:
            # 가장 먼 노드 선택
            mid = max(valid_mid_nodes, key=lambda n: dist_map[n])
            
            p1 = nx.shortest_path(G, start, mid, weight="length")
            p2 = nx.shortest_path(G, mid, start, weight="length")

            loop_path = p1 + p2[1:]
            result = try_route(loop_path, "Out_and_Back_Loop")
            if result: return result
    except Exception as e:
        logging.debug(f"왕복 루프 실패: {e}")


    # ----------------------------------------------------
    # 4단계: 원형 루프 (Fallback Random Walk) - 강제 루프 생성
    # ----------------------------------------------------
    logging.info("4단계: 원형 루프 (Fallback) 시도")
    try:
        # km에 따라 더 많은 스텝
        steps = 10 + int(km * 5) 
        cur = start
        path_nodes = [cur]
        visited = {cur}

        for i in range(steps):
            neigh = list(G.neighbors(cur))
            # 이미 지나온 노드보다는 새로운 노드를 우선
            unvisited_neigh = [n for n in neigh if n not in visited or n == start and i > steps * 0.7]

            if not unvisited_neigh:
                # 더 이상 갈 곳이 없으면 강제로 다시 시작 노드로 돌아감
                p_back = nx.shortest_path(G, cur, start, weight="length")
                path_nodes.extend(p_back[1:])
                break
            
            # 80% 확률로 새로운 노드 선택, 아니면 무작위로 아무 노드 선택
            if unvisited_neigh and random.random() < 0.8:
                 cur = random.choice(unvisited_neigh)
            else:
                 cur = random.choice(neigh)
                 
            if cur == start and i > steps * 0.7:
                path_nodes.append(cur)
                break
            
            path_nodes.append(cur)
            visited.add(cur)
        
        # 끝 노드가 시작 노드가 아니면 강제로 연결
        if path_nodes[-1] != start:
            p_back = nx.shortest_path(G, path_nodes[-1], start, weight="length")
            path_nodes.extend(p_back[1:])
            
        loop_path = path_nodes
        result = try_route(loop_path, "Random_Walk_Circle")
        if result: return result
        
    except Exception as e:
        logging.error(f"원형 루프(Random Walk) 최종 실패: {e}")


    raise RuntimeError(f"목표 거리 {km}km의 유효한 루프 경로를 생성할 수 없습니다.")


###############################################
# 외부 호출 함수
###############################################
def generate_route(lat, lng, km):
    """경로 생성을 위한 외부 인터페이스"""
    G = build_walk_graph(lat, lng, km)
    # make_loop_route_v4는 (polyline_coords, length_m, algorithm_used)를 반환
    return make_loop_route_v4(G, lat, lng, km)
