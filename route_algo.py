# ============================================================
# Ultra-Fast Running Route Generator (Option C2, 통합본)
#  - 목표 : time_s ≈ 4~6초 / err ≤ ±45m
#  - 인터페이스: generate_running_route(lat: float, lng: float, km: float)
# ============================================================

from __future__ import annotations
import math
import random
import time
from typing import List, Tuple, Optional, Dict, Any

import networkx as nx

try:
    import osmnx as ox
except Exception:
    ox = None

LatLng = Tuple[float, float]
Polyline = List[LatLng]


# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------
def haversine(a: LatLng, b: LatLng) -> float:
    """
    두 위경도 좌표 사이의 거리(m)를 반환.
    """
    R = 6371000
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    s = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    d = 2 * R * math.asin(math.sqrt(s))
    return d


def nearest_node(G: nx.Graph, point: LatLng) -> int:
    """
    위경도 point와 가장 가까운 노드 ID를 반환.
    """
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다. `pip install osmnx` 필요.")
    # ox.distance.nearest_nodes(G, x, y) 에서 x=경도, y=위도
    return ox.distance.nearest_nodes(G, point[1], point[0])


def _edge_length(G: nx.Graph, u: int, v: int) -> float:
    """
    networkx Multi(Di)Graph에서 edge length를 안전하게 가져오기.
    """
    data = G.get_edge_data(u, v, default=None)
    if not data:
        return 0.0
    # MultiGraph / MultiDiGraph인 경우: {0: {...}, 1: {...}} 형태
    if isinstance(data, dict) and 0 in data:
        return float(data[0].get("length", 0.0))
    # 일반 DiGraph인 경우
    if isinstance(data, dict):
        # 첫 번째 edge 데이터만 사용
        key = next(iter(data))
        if isinstance(data[key], dict) and "length" in data[key]:
            return float(data[key]["length"])
        # 혹시 한 단계만 있는 dict라면
        if "length" in data:
            return float(data.get("length", 0.0))
    return 0.0


# ------------------------------------------------------------
# 파라미터 (Option C2)
# ------------------------------------------------------------
MAX_VIA_TOTAL = 2          # via 후보 개수 제한
MAX_VIA_PAIRS = 2          # via 쌍 제한 (현재 로직에선 1~2개만 사용)
VIA_MIN_FACTOR = 0.6       # via 최소 거리 계수
VIA_MAX_FACTOR = 1.0       # via 최대 거리 계수
GRAPH_RADIUS_FACTOR = 0.35 # 그래프 반경 축소 (속도↑)
LENGTH_TOLERANCE = 45.0    # 요청 거리 허용 오차 (m)


# ------------------------------------------------------------
# 경로 길이 계산
# ------------------------------------------------------------
def path_length(G: nx.Graph, path: List[int]) -> float:
    """
    노드 ID 리스트 path의 실제 거리(m)를 계산.
    """
    dist = 0.0
    for u, v in zip(path[:-1], path[1:]):
        dist += _edge_length(G, u, v)
    return dist


# ------------------------------------------------------------
# polyline 변환
# ------------------------------------------------------------
def path_to_polyline(G: nx.Graph, path: List[int]) -> Polyline:
    """
    노드 ID 리스트를 [(lat, lng), ...] polyline으로 변환.
    """
    result: Polyline = []
    for nid in path:
        x = G.nodes[nid].get("x")
        y = G.nodes[nid].get("y")
        if x is None or y is None:
            continue
        result.append((y, x))  # (lat, lng)
    return result


# ------------------------------------------------------------
# 그래프 로딩
# ------------------------------------------------------------
def _load_graph(center: LatLng, target_m: float) -> nx.Graph:
    """
    요청 거리와 중심점 기준으로 OSMnx 보행 그래프를 로딩.
    반경은 GRAPH_RADIUS_FACTOR * target_m.
    """
    if ox is None:
        raise RuntimeError("osmnx가 설치되어 있지 않습니다. `pip install osmnx` 필요.")

    radius = target_m * GRAPH_RADIUS_FACTOR
    if radius < 400.0:
        radius = 400.0  # 너무 작은 반경은 그래프가 빈약해질 수 있으니 최소값 보정

    G = ox.graph_from_point(
        center,
        dist=radius,
        network_type="walk",
        simplify=True
    )
    return G


# ============================================================
# ★ 핵심: ultra-fast loop route generator
# ============================================================
def generate_running_route(lat: float, lng: float, km: float) -> Dict[str, Any]:
    """
    러닝 루프 경로 생성 (Option C2).
    - 입력: 시작 위도, 경도, km
    - 출력: dict (status, start, polyline, summary, meta)
    """
    t0 = time.time()
    target_m = km * 1000.0
    center = (lat, lng)

    # --------------------------------------------------------
    # 1) 그래프 다운로드 (반경 축소 → 속도 개선)
    # --------------------------------------------------------
    try:
        G = _load_graph(center, target_m)
    except Exception as e:
        t1 = time.time()
        return {
            "status": "error",
            "message": f"그래프 로딩 실패: {e}",
            "meta": {
                "time_s": t1 - t0
            }
        }

    if len(G.nodes) < 2:
        t1 = time.time()
        return {
            "status": "error",
            "message": "그래프 노드 수가 너무 적습니다.",
            "meta": {
                "time_s": t1 - t0
            }
        }

    # --------------------------------------------------------
    # 2) 가장 가까운 시작 노드
    # --------------------------------------------------------
    try:
        start = nearest_node(G, center)
    except Exception as e:
        t1 = time.time()
        return {
            "status": "error",
            "message": f"nearest_node 계산 실패: {e}",
            "meta": {
                "time_s": t1 - t0
            }
        }

    # --------------------------------------------------------
    # 3) via 후보 선택 – 반경 강력 제한
    # --------------------------------------------------------
    rough_r = target_m / 2.0
    min_r = rough_r * VIA_MIN_FACTOR
    max_r = rough_r * VIA_MAX_FACTOR

    candidates: List[Tuple[int, float]] = []
    for nid in G.nodes:
        ny = G.nodes[nid].get("y")
        nx_ = G.nodes[nid].get("x")
        if ny is None or nx_ is None:
            continue
        p = (ny, nx_)
        d = haversine(center, p)
        if min_r <= d <= max_r:
            candidates.append((nid, d))

    # 거리 기준 정렬 후, 상한선 이상이면 랜덤 샘플링
    candidates = sorted(candidates, key=lambda x: x[1])
    if len(candidates) > MAX_VIA_TOTAL:
        candidates = random.sample(candidates, MAX_VIA_TOTAL)

    via_ids = [c[0] for c in candidates]

    if len(via_ids) < 1:
        # via 실패 → 단순 왕복
        return _fallback_return(G, start, target_m, t0)

    # via pair 구성 (최대 1~2개)
    via_pairs: List[Tuple[int, int]] = []
    if len(via_ids) == 1:
        via_pairs = [(via_ids[0], via_ids[0])]
    else:
        via_pairs = [(via_ids[0], via_ids[1])]

    # --------------------------------------------------------
    # 4) 경로 조합 생성 및 평가
    # --------------------------------------------------------
    best: Optional[Dict[str, Any]] = None

    for a, b in via_pairs[:MAX_VIA_PAIRS]:
        try:
            path1 = nx.shortest_path(G, start, a, weight="length")
            path2 = nx.shortest_path(G, a, b, weight="length")
            path3 = nx.shortest_path(G, b, start, weight="length")
        except Exception:
            continue

        full = path1 + path2[1:] + path3[1:]
        L = path_length(G, full)
        err = abs(L - target_m)

        if best is None or err < best["err"]:
            best = {
                "path": full,
                "length": L,
                "err": err,
                "a": a,
                "b": b,
            }

    if best is None:
        # via 경로 구성 실패 → fallback
        return _fallback_return(G, start, target_m, t0)

    # --------------------------------------------------------
    # 5) 오차 허용 범위 내면 성공
    # --------------------------------------------------------
    if best["err"] <= LENGTH_TOLERANCE:
        return _build_response(
            G,
            start,
            best["path"],
            best["length"],
            km,
            used_fallback=False,
            t0=t0,
        )

    # --------------------------------------------------------
    # 6) 그래도 오차 크면, 그래도 best 경로는 주되 approx로 표시
    # --------------------------------------------------------
    return _build_response(
        G,
        start,
        best["path"],
        best["length"],
        km,
        used_fallback=True,
        t0=t0,
    )


# ============================================================
# Fallback – 간단 왕복 루트
# ============================================================
def _fallback_return(G: nx.Graph, start: int, target_m: float, t0: float) -> Dict[str, Any]:
    """
    via 기반 루프를 못 만들었을 때 사용하는 단순 왕복 경로.
    """
    nodes = list(G.nodes)
    if len(nodes) < 2:
        t1 = time.time()
        return {
            "status": "error",
            "message": "그래프가 너무 작아서 fallback도 실패했습니다.",
            "meta": {
                "time_s": t1 - t0,
            },
        }

    # 그냥 중간쯤 노드를 하나 골라 왕복
    far = nodes[len(nodes) // 2]
    try:
        p1 = nx.shortest_path(G, start, far, weight="length")
        p2 = list(reversed(p1))
        full = p1 + p2[1:]
        L = path_length(G, full)
    except Exception as e:
        t1 = time.time()
        return {
            "status": "error",
            "message": f"Fallback shortest_path 실패: {e}",
            "meta": {
                "time_s": t1 - t0,
            },
        }

    return _build_response(
        G,
        start,
        full,
        L,
        target_m / 1000.0,
        used_fallback=True,
        t0=t0,
    )


# ============================================================
# JSON Response Builder
# ============================================================
def _build_response(
    G: nx.Graph,
    start: int,
    path: List[int],
    L: float,
    km: float,
    used_fallback: bool,
    t0: float,
) -> Dict[str, Any]:
    """
    최종 JSON 응답 형태를 만드는 함수.
    (turn_algo는 app.py에서 polyline을 넘겨서 따로 붙이는 구조라면,
     여기서는 polyline/summary/meta까지만 책임)
    """
    t1 = time.time()
    polyline = path_to_polyline(G, path)

    if not polyline:
        return {
            "status": "error",
            "message": "polyline 생성 실패",
            "meta": {
                "time_s": t1 - t0,
            },
        }

    status = "ok" if not used_fallback else "approx"

    return {
        "status": status,
        "start": {
            "lat": polyline[0][0],
            "lng": polyline[0][1],
        },
        "polyline": [
            {"lat": p[0], "lng": p[1]} for p in polyline
        ],
        "summary": {
            "length_m": round(L, 1),
            "km_requested": km,
            "estimated_time_min": round((L / 1000.0) / 8.0 * 60.0, 1),
        },
        "meta": {
            "len": L,
            "err": abs(L - km * 1000.0),
            "time_s": t1 - t0,
            "used_fallback": used_fallback,
        },
    }