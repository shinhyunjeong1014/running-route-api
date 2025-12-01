# route_algo.py (A-1 연구용 Full 버전)
#
# 완전 논문 기반 러닝/워킹 루프 생성 모듈
#
# 사용 개념:
# - Random and shortest path generation for running or walking purposes
#   · OSM 보행 그래프 + 최단경로(Dijkstra)
#   · 목표 거리와 오차를 최소화하는 랜덤 왕복 루트
# - RUNAMIC
#   · 시작점 주변 anchor(rod endpoint) 노드 여러 개 선택
#   · start → A → B → C → start 형태의 사이클 생성
# - WSRP24 cycle algorithms
#   · roundness(4πA/P²), self-intersection, edge reuse를 품질 지표로 사용
# - path_traverser
#   · “거리 오차 + 품질 페널티”를 하나의 score로 만들어 최적 루프 선택
#
# 외부 인터페이스 (app.py에서 사용):
#   polyline_tuples, meta = generate_area_loop(lat, lng, km)
#
# polyline_tuples: [(lat, lng), ...]
# meta: {
#   "len": 전체 길이(m),
#   "err": |len - target_m|,
#   "roundness": 0~1,
#   "success": (err <= 99m),
#   "used_fallback": 항상 False (이 버전은 사각형 fallback 사용 X),
#   "valhalla_calls": 0,
#   "kakao_calls": 0,
#   "routes_checked": 시도한 루프 개수,
#   "routes_validated": 실제로 계산된 루프 개수,
#   "km_requested": 요청 km,
#   "target_m": 목표 거리(m),
#   "time_s": 전체 실행 시간,
#   "message": 상태 메시지 문자열
# }

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional
import heapq

import requests

EARTH_RADIUS_M = 6371000.0

# -----------------------------
# 기본 지오메트리 유틸
# -----------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 점 사이의 거리(m)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def polyline_length_m(points: List[Tuple[float, float]]) -> float:
    """폴리라인의 총 길이(m)."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        total += haversine_m(lat1, lon1, lat2, lon2)
    return total


def project_to_local_xy(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    작은 영역(수 km 이내)에서 lat/lon을 평면(m 단위)으로 투영.
    equirectangular projection 사용.
    """
    if not points:
        return []
    lat0 = math.radians(points[0][0])
    lon0 = math.radians(points[0][1])
    out: List[Tuple[float, float]] = []
    for lat, lon in points:
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        x = EARTH_RADIUS_M * (lon_r - lon0) * math.cos(lat0)
        y = EARTH_RADIUS_M * (lat_r - lat0)
        out.append((x, y))
    return out


def polygon_area_and_perimeter(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    points가 만드는 폐곡선의 면적과 둘레를 근사 계산.
    시작/끝이 같지 않아도 자동으로 닫아서 계산.
    """
    if len(points) < 3:
        return 0.0, 0.0
    xy = project_to_local_xy(points)
    if not xy:
        return 0.0, 0.0
    xys = xy + [xy[0]]
    area = 0.0
    for (x1, y1), (x2, y2) in zip(xys, xys[1:]):
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    perim = 0.0
    for (x1, y1), (x2, y2) in zip(xys, xys[1:]):
        dx = x2 - x1
        dy = y2 - y1
        perim += math.hypot(dx, dy)
    return area, perim


def roundness_index(points: List[Tuple[float, float]]) -> float:
    """
    원형에 얼마나 가까운지 지표: 4πA / P² (원 = 1, 선에 가까울수록 0).
    """
    area, perim = polygon_area_and_perimeter(points)
    if perim <= 0.0:
        return 0.0
    return 4.0 * math.pi * area / (perim * perim)


# -----------------------------
# Turn 탐지 (음성 안내용)
# -----------------------------

TURN_ANGLE_THRESHOLD_DEG = 35.0
UTURN_ANGLE_THRESHOLD_DEG = 150.0

def _segment_bearing_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """두 점 사이 방위각 (북 기준, 시계 방향) deg."""
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dl = lon2 - lon1
    x = math.sin(dl) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dl)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    """두 방위각 차이 [-180, 180]."""
    diff = (b - a + 180.0) % 360.0 - 180.0
    return diff


def extract_turns(polyline: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    polyline에서 의미 있는 회전 지점을 추출.
    """
    turns: List[Dict[str, Any]] = []
    n = len(polyline)
    if n < 3:
        return turns

    # 누적 거리
    cum = [0.0]
    for i in range(1, n):
        d = haversine_m(*polyline[i - 1], *polyline[i])
        cum.append(cum[-1] + d)

    for i in range(1, n - 1):
        p_prev = polyline[i - 1]
        p_curr = polyline[i]
        p_next = polyline[i + 1]

        b1 = _segment_bearing_deg(p_prev, p_curr)
        b2 = _segment_bearing_deg(p_curr, p_next)
        diff = _angle_diff_deg(b1, b2)
        ad = abs(diff)

        if ad < TURN_ANGLE_THRESHOLD_DEG:
            continue

        if ad >= UTURN_ANGLE_THRESHOLD_DEG:
            t_type = "uturn"
        elif diff > 0:
            t_type = "right"
        else:
            t_type = "left"

        at = cum[i]
        inst = f"{int(round(at))}m 앞에서 " + (
            "U턴" if t_type == "uturn" else ("우회전" if t_type == "right" else "좌회전")
        )

        turns.append(
            {
                "type": t_type,
                "lat": p_curr[0],
                "lng": p_curr[1],
                "at_dist_m": at,
                "instruction": inst,
            }
        )

    return turns


# -----------------------------
# OSM 그래프 구축 (Overpass API)
# -----------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 보행에 사용할 highway 타입
OSM_HIGHWAY_FILTER = [
    "footway", "path", "cycleway", "pedestrian", "living_street",
    "residential", "track", "service", "steps", "unclassified", "tertiary",
]


class OSMGraph:
    def __init__(self) -> None:
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.adj: Dict[int, List[Tuple[int, float]]] = {}

    def add_node(self, node_id: int, lat: float, lon: float) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = (lat, lon)
            self.adj[node_id] = []

    def add_edge(self, u: int, v: int) -> None:
        if u not in self.nodes or v not in self.nodes:
            return
        lat1, lon1 = self.nodes[u]
        lat2, lon2 = self.nodes[v]
        w = haversine_m(lat1, lon1, lat2, lon2)
        if w <= 0:
            return
        # 무향 그래프 (양방향 보행 가능 가정)
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))


def _meters_to_latlon_delta(lat: float, meters: float) -> Tuple[float, float]:
    """
    주어진 위도에서 meter를 degree 단위 (dlat, dlon)로 변환.
    """
    dlat = meters / 111320.0
    lat_rad = math.radians(lat)
    dlon = meters / (111320.0 * max(math.cos(lat_rad), 1e-6))
    return dlat, dlon


def fetch_osm_pedestrian_graph(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    timeout: float = 60.0,
) -> OSMGraph:
    """
    Overpass API를 사용하여 center 주변 radius_m 이내 보행 네트워크를 그래프로 구축.
    """
    dlat, dlon = _meters_to_latlon_delta(center_lat, radius_m)
    south = center_lat - dlat
    north = center_lat + dlat
    west = center_lon - dlon
    east = center_lon + dlon

    highway_regex = "|".join(OSM_HIGHWAY_FILTER)
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"{highway_regex}"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """

    resp = requests.post(OVERPASS_URL, data=query.encode("utf-8"), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    graph = OSMGraph()
    nodes_tmp: Dict[int, Tuple[float, float]] = {}

    # 노드 임시 저장
    for el in data.get("elements", []):
        if el.get("type") == "node":
            nid = int(el["id"])
            nodes_tmp[nid] = (el["lat"], el["lon"])

    # way를 순회하며 보행 가능한 highway만 그래프에 추가
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        hw = tags.get("highway")
        if not hw or hw not in OSM_HIGHWAY_FILTER:
            continue
        node_ids = el.get("nodes", [])

        # nodes 추가
        for nid in node_ids:
            if nid in nodes_tmp:
                lat, lon = nodes_tmp[nid]
                graph.add_node(nid, lat, lon)

        # 인접 edge 추가
        for u, v in zip(node_ids, node_ids[1:]):
            if u in graph.nodes and v in graph.nodes:
                graph.add_edge(u, v)

    return graph


def _nearest_node(graph: OSMGraph, lat: float, lon: float) -> Optional[int]:
    """
    (lat, lon)에 가장 가까운 그래프 노드 찾기.
    """
    best_id = None
    best_d = float("inf")
    for nid, (nlat, nlon) in graph.nodes.items():
        d = haversine_m(lat, lon, nlat, nlon)
        if d < best_d:
            best_d = d
            best_id = nid
    return best_id


# -----------------------------
# 최단경로 (Dijkstra)
# -----------------------------

def shortest_path(
    graph: OSMGraph,
    src: int,
    dst: int,
    max_dist: Optional[float] = None,
) -> Optional[List[int]]:
    """
    Dijkstra 최단경로.
    max_dist가 주어지면 그 이상 뻗는 경로는 탐색 중단.
    """
    if src == dst:
        return [src]

    dist: Dict[int, float] = {src: 0.0}
    prev: Dict[int, Optional[int]] = {src: None}
    pq: List[Tuple[float, int]] = [(0.0, src)]
    visited: set[int] = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if max_dist is not None and d > max_dist:
            continue
        if u == dst:
            break

        for v, w in graph.adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dst not in dist:
        return None

    # 경로 복원
    path: List[int] = []
    cur: Optional[int] = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def nodes_to_polyline(graph: OSMGraph, path: List[int]) -> List[Tuple[float, float]]:
    return [graph.nodes[nid] for nid in path]


# -----------------------------
# 품질 지표: self-intersection, edge reuse
# -----------------------------

def _segments(points: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    return list(zip(points, points[1:]))


def _ccw(a, b, c) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segment_intersect(p1, p2, p3, p4) -> bool:
    # 인접/공유 점은 intersection으로 보지 않음
    if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
        return False
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


def count_self_intersections(points: List[Tuple[float, float]]) -> int:
    segs = _segments(points)
    n = len(segs)
    cnt = 0
    for i in range(n):
        for j in range(i + 2, n):
            if j == i + 1:
                continue
            if _segment_intersect(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                cnt += 1
    return cnt


def edge_reuse_ratio(points: List[Tuple[float, float]]) -> float:
    segs = _segments(points)
    if not segs:
        return 0.0
    edge_counts: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
    for a, b in segs:
        key = (a, b) if a <= b else (b, a)
        edge_counts[key] = edge_counts.get(key, 0) + 1
    reused = sum(c - 1 for c in edge_counts.values() if c > 1)
    return reused / float(len(segs))


# -----------------------------
# 루프 후보 생성 (RUNAMIC/Random 스타일)
# -----------------------------

MIN_RADIUS_M = 150.0
MAX_RADIUS_M = 1500.0
MAX_LOOP_CANDIDATES = 40    # 생성해볼 루프 수


def _estimate_radius(target_m: float) -> float:
    """
    target_m 기준 대략적인 루프 반경 추정 (원 둘레 2πR ≈ target_m).
    """
    R = target_m / (2.0 * math.pi)
    R = max(MIN_RADIUS_M, min(MAX_RADIUS_M, R))
    return R


def select_anchor_candidates(
    graph: OSMGraph,
    start_lat: float,
    start_lon: float,
    target_m: float,
    max_candidates: int = 40,
) -> List[int]:
    """
    RUNAMIC/WSRP 스타일 anchor 후보 노드 선택.
    - start에서 일정 반경(R) 근처에 있는 노드들을 ring에서 뽑음.
    - 부족하면 전체에서 가까운 순으로 보충.
    """
    R = _estimate_radius(target_m)
    min_r = 0.5 * R
    max_r = 1.2 * R

    nodes_in_ring: List[Tuple[int, float]] = []
    for nid, (lat, lon) in graph.nodes.items():
        d = haversine_m(start_lat, start_lon, lat, lon)
        if min_r <= d <= max_r:
            nodes_in_ring.append((nid, d))

    # ring 안 노드가 너무 적으면 전체 중 가까운 노드 사용
    if len(nodes_in_ring) < 10:
        tmp: List[Tuple[int, float]] = []
        for nid, (lat, lon) in graph.nodes.items():
            d = haversine_m(start_lat, start_lon, lat, lon)
            tmp.append((nid, d))
        tmp.sort(key=lambda x: x[1])
        nodes_in_ring = tmp[:max_candidates]

    random.shuffle(nodes_in_ring)
    return [nid for nid, _ in nodes_in_ring[:max_candidates]]


def _concat_polylines(polys: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """여러 polyline을 이어붙이되, 경계점 중복 제거."""
    out: List[Tuple[float, float]] = []
    for poly in polys:
        if not poly:
            continue
        if not out:
            out.extend(poly)
        else:
            if out[-1] == poly[0]:
                out.extend(poly[1:])
            else:
                out.extend(poly)
    return out


# -----------------------------
# 메인 함수
# -----------------------------

def generate_area_loop(lat: float, lng: float, km: float) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    연구용 Full 버전 루프 생성.
    - OSM 보행 그래프 + Dijkstra + 랜덤 anchor 사이클
    - 여러 루프 후보를 만들어 score가 가장 좋은 루프를 반환
    """
    start_time = time.time()
    target_m = float(km) * 1000.0

    # 1. OSM 그래프 로드 (너무 넓지 않게 radius 제한)
    radius_for_osm = max(800.0, min(3000.0, target_m))
    try:
        graph = fetch_osm_pedestrian_graph(lat, lng, radius_for_osm)
    except Exception as e:
        elapsed = time.time() - start_time
        meta = {
            "len": 0.0,
            "err": float("inf"),
            "roundness": 0.0,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": 0,
            "kakao_calls": 0,
            "routes_checked": 0,
            "routes_validated": 0,
            "km_requested": float(km),
            "target_m": float(target_m),
            "time_s": float(elapsed),
            "message": f"OSM 그래프 로드 실패: {e}",
        }
        return [], meta

    if not graph.nodes:
        elapsed = time.time() - start_time
        meta = {
            "len": 0.0,
            "err": float("inf"),
            "roundness": 0.0,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": 0,
            "kakao_calls": 0,
            "routes_checked": 0,
            "routes_validated": 0,
            "km_requested": float(km),
            "target_m": float(target_m),
            "time_s": float(elapsed),
            "message": "OSM 보행 네트워크에 노드가 없습니다.",
        }
        return [], meta

    start_nid = _nearest_node(graph, lat, lng)
    if start_nid is None:
        elapsed = time.time() - start_time
        meta = {
            "len": 0.0,
            "err": float("inf"),
            "roundness": 0.0,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": 0,
            "kakao_calls": 0,
            "routes_checked": 0,
            "routes_validated": 0,
            "km_requested": float(km),
            "target_m": float(target_m),
            "time_s": float(elapsed),
            "message": "시작점 근처에 보행 가능한 노드를 찾지 못했습니다.",
        }
        return [], meta

    start_lat, start_lon = graph.nodes[start_nid]

    # 2. Anchor 후보 선택
    anchor_candidates = select_anchor_candidates(graph, start_lat, start_lon, target_m, max_candidates=50)
    if len(anchor_candidates) < 2:
        elapsed = time.time() - start_time
        meta = {
            "len": 0.0,
            "err": float("inf"),
            "roundness": 0.0,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": 0,
            "kakao_calls": 0,
            "routes_checked": 0,
            "routes_validated": 0,
            "km_requested": float(km),
            "target_m": float(target_m),
            "time_s": float(elapsed),
            "message": "루프 생성을 위한 anchor 후보 노드가 충분하지 않습니다.",
        }
        return [], meta

    best_polyline: List[Tuple[float, float]] = []
    best_score = float("inf")
    best_err = float("inf")
    best_roundness = 0.0
    routes_checked = 0
    routes_validated = 0

    # 3. 여러 랜덤 루프 후보 생성
    for _ in range(MAX_LOOP_CANDIDATES):
        # anchor 개수: 2 또는 3
        num_anchor = 3 if len(anchor_candidates) >= 3 else 2
        anchors = random.sample(anchor_candidates, num_anchor)
        seq = [start_nid] + anchors + [start_nid]

        legs_nodes: List[List[int]] = []
        failed = False
        for a, b in zip(seq, seq[1:]):
            path = shortest_path(graph, a, b)
            if path is None or len(path) < 2:
                failed = True
                break
            legs_nodes.append(path)
        if failed:
            continue

        legs_polys = [nodes_to_polyline(graph, p) for p in legs_nodes]
        polyline = _concat_polylines(legs_polys)
        if len(polyline) < 4:
            continue

        length_m = polyline_length_m(polyline)
        if length_m <= 0.0:
            continue

        routes_checked += 1
        routes_validated += 1

        # 품질 계산
        err = abs(length_m - target_m)
        err_norm = err / max(target_m, 1.0)
        rnd = roundness_index(polyline)
        self_int = count_self_intersections(polyline)
        reuse = edge_reuse_ratio(polyline)

        # score: 작을수록 좋은 루프
        score = (
            5.0 * err_norm +      # 거리 오차 비중 크게
            1.5 * (1.0 - rnd) +   # 원형에 더 가깝게
            1.0 * self_int +      # self-intersection 적게
            2.0 * reuse           # 같은 edge 재사용 적게
        )

        if score < best_score:
            best_score = score
            best_err = err
            best_roundness = rnd
            best_polyline = polyline

    elapsed = time.time() - start_time

    if not best_polyline:
        meta = {
            "len": 0.0,
            "err": float("inf"),
            "roundness": 0.0,
            "success": False,
            "used_fallback": False,
            "valhalla_calls": 0,
            "kakao_calls": 0,
            "routes_checked": int(routes_checked),
            "routes_validated": int(routes_validated),
            "km_requested": float(km),
            "target_m": float(target_m),
            "time_s": float(elapsed),
            "message": "그래프 기반 루프 생성에 실패했습니다.",
        }
        return [], meta

    success = best_err <= 99.0
    meta = {
        "len": float(polyline_length_m(best_polyline)),
        "err": float(best_err),
        "roundness": float(best_roundness),
        "success": bool(success),
        "used_fallback": False,
        "valhalla_calls": 0,
        "kakao_calls": 0,
        "routes_checked": int(routes_checked),
        "routes_validated": int(routes_validated),
        "km_requested": float(km),
        "target_m": float(target_m),
        "time_s": float(elapsed),
        "message": "허용 오차(±99m) 내 루프" if success else "허용 오차(±99m)를 만족하지 못했지만 최선의 루프를 반환합니다.",
    }

    return best_polyline, meta
