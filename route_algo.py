# route_algo.py
import random
import networkx as nx
import osmnx as ox
import math
import logging

logging.basicConfig(level=logging.INFO)
TARGET_RANGE_M = 250  # 목표 거리 ± 250m 허용

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
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            length += edge_data[0].get('length', 0)
    return length


def _polyline_length(polyline):
    """Polyline 좌표 리스트의 총 길이를 계산합니다."""
    length = 0
    for i in range(len(polyline) - 1):
        length += haversine_m(
            polyline[i]["lat"], polyline[i]["lng"],
            polyline[i+1]["lat"], polyline[i+1]["lng"]
        )
    return length


def deduplicate_polyline(polyline):
    """연속된 중복 좌표를 제거합니다."""
    if not polyline:
        return []
    result = [polyline[0]]
    for p in polyline[1:]:
        if p["lat"] != result[-1]["lat"] or p["lng"] != result[-1]["lng"]:
            result.append(p)
    return result


###############################################
# Fallback 경로 생성 알고리즘
###############################################
def generate_circle_loop(center_lat, center_lng, target_m):
    """원형 루프 생성 (Fallback A)"""
    radius_m = target_m / (2 * math.pi)  # 원주 = 2πr
    num_points = max(16, int(target_m / 100))  # 100m당 1개 포인트
    
    polyline = []
    for i in range(num_points + 1):
        angle = (i / num_points) * 2 * math.pi
        # 위도/경도 1도당 대략적인 미터 변환 (위도 37도 기준)
        lat_offset = (radius_m * math.cos(angle)) / 111320
        lng_offset = (radius_m * math.sin(angle)) / (111320 * math.cos(math.radians(center_lat)))
        
        polyline.append({
            "lat": center_lat + lat_offset,
            "lng": center_lng + lng_offset
        })
    
    return polyline


def generate_square_loop(center_lat, center_lng, target_m):
    """사각형 루프 생성 (Fallback B)"""
    side_length_m = target_m / 4  # 정사각형 한 변 길이
    
    # 위도/경도 변환 (위도 37도 기준)
    lat_offset = side_length_m / 2 / 111320
    lng_offset = side_length_m / 2 / (111320 * math.cos(math.radians(center_lat)))
    
    # 사각형 꼭짓점 생성 (시계방향)
    corners = [
        {"lat": center_lat + lat_offset, "lng": center_lng - lng_offset},  # 북서
        {"lat": center_lat + lat_offset, "lng": center_lng + lng_offset},  # 북동
        {"lat": center_lat - lat_offset, "lng": center_lng + lng_offset},  # 남동
        {"lat": center_lat - lat_offset, "lng": center_lng - lng_offset},  # 남서
        {"lat": center_lat + lat_offset, "lng": center_lng - lng_offset},  # 시작점으로 복귀
    ]
    
    # 각 변을 여러 포인트로 세분화 (부드러운 경로 생성)
    polyline = []
    points_per_side = max(8, int(side_length_m / 50))
    
    for i in range(len(corners) - 1):
        start, end = corners[i], corners[i + 1]
        for j in range(points_per_side):
            t = j / points_per_side
            polyline.append({
                "lat": start["lat"] + (end["lat"] - start["lat"]) * t,
                "lng": start["lng"] + (end["lng"] - start["lng"]) * t
            })
    
    polyline.append(corners[0])  # 시작점으로 완전히 복귀
    return polyline


def generate_triangle_loop(center_lat, center_lng, target_m):
    """삼각형 루프 생성 (Fallback C)"""
    side_length_m = target_m / 3
    height_m = side_length_m * math.sqrt(3) / 2
    
    lat_offset = height_m / 2 / 111320
    lng_offset = side_length_m / 2 / (111320 * math.cos(math.radians(center_lat)))
    
    # 삼각형 꼭짓점 (정삼각형)
    corners = [
        {"lat": center_lat + lat_offset, "lng": center_lng},  # 북
        {"lat": center_lat - lat_offset/2, "lng": center_lng + lng_offset},  # 남동
        {"lat": center_lat - lat_offset/2, "lng": center_lng - lng_offset},  # 남서
        {"lat": center_lat + lat_offset, "lng": center_lng},  # 시작점 복귀
    ]
    
    # 각 변 세분화
    polyline = []
    points_per_side = max(8, int(side_length_m / 50))
    
    for i in range(len(corners) - 1):
        start, end = corners[i], corners[i + 1]
        for j in range(points_per_side):
            t = j / points_per_side
            polyline.append({
                "lat": start["lat"] + (end["lat"] - start["lat"]) * t,
                "lng": start["lng"] + (end["lng"] - start["lng"]) * t
            })
    
    polyline.append(corners[0])
    return polyline


###############################################
# OSM 그래프 생성 (Fallback 포함)
###############################################
def build_walk_graph(lat, lng, km):
    """
    보행 네트워크 그래프 생성.
    실패 시 None 반환 (Fallback 경로 생성으로 넘어감)
    """
    base_dist = km * 1000 * 1.5
    search_dists = [base_dist, base_dist * 1.5, base_dist * 2.0, 5000]

    for dist in search_dists:
        try:
            logging.info(f"OSMnx 그래프 생성 시도: 반경 {dist:.0f}m")
            G = ox.graph_from_point(
                (lat, lng),
                dist=dist,
                network_type="walk",
                simplify=True,
                retain_all=False,
                timeout=10
            )
            
            if len(G.nodes) < 30:
                logging.warning(f"노드 수 부족: {len(G.nodes)}개 (최소 30개 필요)")
                continue
            
            G = ox.add_edge_speeds(G)
            G = ox.add_travel_times(G)
            logging.info(f"그래프 생성 성공: 노드 {len(G.nodes)}개, 반경 {dist:.0f}m")
            return G
            
        except Exception as e:
            logging.debug(f"그래프 생성 실패 (반경 {dist:.0f}m): {e}")
            continue
    
    logging.warning("OSM 보행 네트워크 생성 실패 - Fallback 경로 생성으로 전환")
    return None


###############################################
# 경로 길이 보정
###############################################
def _adjust_route_length(G, path, target_m):
    """경로 길이를 목표 범위에 맞게 보정합니다."""
    current_length = _path_length(G, path)
    target_min = target_m - TARGET_RANGE_M
    target_max = target_m + TARGET_RANGE_M
    
    if target_min <= current_length <= target_max:
        return path, current_length
    
    diff_m = target_m - current_length
    
    # 길이 부족 시 패딩 추가
    if diff_m > TARGET_RANGE_M:
        logging.info(f"경로 부족: {current_length:.0f}m. {diff_m:.0f}m 보정 필요")
        end_node = path[-1]
        cutoff_m = min(diff_m * 1.5, target_m * 0.5)
        
        try:
            sub_dist_map = nx.single_source_dijkstra_path_length(G, end_node, cutoff=cutoff_m)
            cand_nodes = list(sub_dist_map.keys())
            
            if len(cand_nodes) > 1:
                mid_node = random.choice(cand_nodes[1:])
                p_out = nx.shortest_path(G, end_node, mid_node, weight="length")
                p_back = nx.shortest_path(G, mid_node, end_node, weight="length")
                added_path = p_out + p_back[1:]
                new_path = path[:-1] + added_path
                new_length = _path_length(G, new_path)
                logging.info(f"패딩 추가: {new_length:.0f}m")
                return new_path, new_length
        except Exception as e:
            logging.warning(f"경로 보정 실패: {e}")
    
    return path, current_length


###############################################
# OSM 기반 루프 생성
###############################################
def make_osm_loop_route(G, start_lat, start_lng, km):
    """OSM 네트워크 기반 루프 경로 생성 (기존 v4 로직)"""
    target_m = km * 1000
    start = ox.nearest_nodes(G, start_lng, start_lat)
    target_min = target_m - TARGET_RANGE_M
    target_max = target_m + TARGET_RANGE_M

    def try_route(loop_path, method_name):
        if not loop_path or len(loop_path) < 3:
            return None
        
        initial_length = _path_length(G, loop_path)
        
        if initial_length < target_min or initial_length > target_max:
            loop_path, final_length = _adjust_route_length(G, loop_path, target_m)
        else:
            final_length = initial_length
        
        if target_min <= final_length <= target_max:
            polyline = nodes_to_latlngs(G, loop_path)
            polyline = deduplicate_polyline(polyline)
            logging.info(f"경로 성공: {method_name}, 길이 {final_length:.0f}m")
            return polyline, final_length, method_name
        
        return None

    # 1. 삼각 루프
    try:
        logging.info("삼각 루프 시도")
        cutoff = target_m * 0.7
        dist_map = nx.single_source_dijkstra_path_length(G, start, cutoff=cutoff, weight="length")
        cand = [n for n, d in dist_map.items() if target_m * 0.25 <= d <= target_m * 0.66]
        
        if len(cand) >= 2:
            A, B = random.sample(cand, 2)
            p1 = nx.shortest_path(G, start, A, weight="length")
            p2 = nx.shortest_path(G, A, B, weight="length")
            p3 = nx.shortest_path(G, B, start, weight="length")
            loop_path = p1 + p2[1:] + p3[1:]
            result = try_route(loop_path, "OSM_Triangle")
            if result:
                return result
    except Exception as e:
        logging.debug(f"삼각 루프 실패: {e}")

    # 2. 사각 루프
    try:
        logging.info("사각 루프 시도")
        if 'cand' in locals() and len(cand) >= 3:
            A, B, C = random.sample(cand, 3)
            p1 = nx.shortest_path(G, start, A, weight="length")
            p2 = nx.shortest_path(G, A, B, weight="length")
            p3 = nx.shortest_path(G, B, C, weight="length")
            p4 = nx.shortest_path(G, C, start, weight="length")
            loop_path = p1 + p2[1:] + p3[1:] + p4[1:]
            result = try_route(loop_path, "OSM_Square")
            if result:
                return result
    except Exception as e:
        logging.debug(f"사각 루프 실패: {e}")

    # 3. 왕복 루프
    try:
        logging.info("왕복 루프 시도")
        half_target = target_m / 2
        cutoff = half_target * 1.5
        dist_map = nx.single_source_dijkstra_path_length(G, start, cutoff=cutoff, weight="length")
        valid_mid = [n for n, d in dist_map.items() if half_target * 0.8 <= d <= half_target * 1.2]
        
        if valid_mid:
            mid = max(valid_mid, key=lambda n: dist_map[n])
            p1 = nx.shortest_path(G, start, mid, weight="length")
            p2 = nx.shortest_path(G, mid, start, weight="length")
            loop_path = p1 + p2[1:]
            result = try_route(loop_path, "OSM_OutAndBack")
            if result:
                return result
    except Exception as e:
        logging.debug(f"왕복 루프 실패: {e}")

    # 4. 랜덤 워크
    try:
        logging.info("랜덤 워크 루프 시도")
        steps = 10 + int(km * 5)
        cur = start
        path_nodes = [cur]
        visited = {cur}
        
        for i in range(steps):
            neigh = list(G.neighbors(cur))
            unvisited = [n for n in neigh if n not in visited or (n == start and i > steps * 0.7)]
            
            if not unvisited:
                p_back = nx.shortest_path(G, cur, start, weight="length")
                path_nodes.extend(p_back[1:])
                break
            
            cur = random.choice(unvisited) if random.random() < 0.8 and unvisited else random.choice(neigh)
            
            if cur == start and i > steps * 0.7:
                path_nodes.append(cur)
                break
            
            path_nodes.append(cur)
            visited.add(cur)
        
        if path_nodes[-1] != start:
            p_back = nx.shortest_path(G, path_nodes[-1], start, weight="length")
            path_nodes.extend(p_back[1:])
        
        result = try_route(path_nodes, "OSM_RandomWalk")
        if result:
            return result
    except Exception as e:
        logging.warning(f"랜덤 워크 실패: {e}")

    return None


###############################################
# 메인 경로 생성 함수
###############################################
def generate_route(lat, lng, km):
    """
    경로 생성 메인 함수 - 절대 실패하지 않음
    OSM 네트워크 사용 불가 시 자동으로 Fallback 경로 생성
    """
    target_m = km * 1000
    target_min = target_m - TARGET_RANGE_M
    target_max = target_m + TARGET_RANGE_M
    
    # 1. OSM 네트워크 시도
    G = build_walk_graph(lat, lng, km)
    
    if G is not None:
        try:
            result = make_osm_loop_route(G, lat, lng, km)
            if result:
                return result
        except Exception as e:
            logging.warning(f"OSM 루프 생성 실패: {e}")
    
    # 2. Fallback 경로 생성 (절대 실패하지 않음)
    logging.info("Fallback 경로 생성 모드 진입")
    
    fallback_methods = [
        ("Circle", generate_circle_loop),
        ("Square", generate_square_loop),
        ("Triangle", generate_triangle_loop),
    ]
    
    for method_name, generator_func in fallback_methods:
        try:
            logging.info(f"Fallback_{method_name} 루프 생성 시도")
            polyline = generator_func(lat, lng, target_m)
            polyline = deduplicate_polyline(polyline)
            length_m = _polyline_length(polyline)
            
            # 길이 보정 (간단한 스케일링)
            if length_m < target_min or length_m > target_max:
                scale_factor = target_m / length_m
                logging.info(f"Fallback 경로 스케일링: {scale_factor:.2f}x")
                
                # 원점 기준으로 스케일링
                center = {"lat": lat, "lng": lng}
                scaled_polyline = []
                for p in polyline:
                    dlat = (p["lat"] - center["lat"]) * scale_factor
                    dlng = (p["lng"] - center["lng"]) * scale_factor
                    scaled_polyline.append({
                        "lat": center["lat"] + dlat,
                        "lng": center["lng"] + dlng
                    })
                polyline = scaled_polyline
                length_m = _polyline_length(polyline)
            
            if target_min <= length_m <= target_max:
                logging.info(f"Fallback_{method_name} 성공: {length_m:.0f}m")
                return polyline, length_m, f"Fallback_{method_name}"
                
        except Exception as e:
            logging.warning(f"Fallback_{method_name} 실패: {e}")
            continue
    
    # 3. 최후의 Fallback (원형 루프 강제 생성)
    logging.warning("모든 방법 실패 - 강제 원형 루프 생성")
    polyline = generate_circle_loop(lat, lng, target_m)
    polyline = deduplicate_polyline(polyline)
    length_m = _polyline_length(polyline)
    return polyline, length_m, "Fallback_Circle_Emergency"
