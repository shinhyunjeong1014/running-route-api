# route_algo.py
import random
import networkx as nx
import osmnx as ox
from typing import Dict, List
from itertools import islice
import math
from statistics import mean

# --------------------------------------------------------
# 1) 그래프 생성 (+안전가중치 비용)
# --------------------------------------------------------
def build_walk_graph(lat: float, lng: float, km: float):
    for dist_m in [1500, 2500, 4000, 6000, 8000]:
        try:
            G = ox.graph_from_point(
                (lat, lng),
                dist=dist_m,
                network_type="walk",
                simplify=True,
                retain_all=False,
            )
        except Exception:
            continue
        if len(G) == 0:
            continue

        try:
            s_node = ox.nearest_nodes(G, lng, lat)
        except Exception:
            s_node = min(G.nodes, key=lambda n: (G.nodes[n]["y"]-lat)**2 + (G.nodes[n]["x"]-lng)**2)

        comps = list(nx.connected_components(G.to_undirected()))
        comp_nodes = max(comps, key=len)
        if s_node not in comp_nodes:
            comp_nodes = min(
                comps,
                key=lambda c: min((G.nodes[n]["x"]-lng)**2 + (G.nodes[n]["y"]-lat)**2 for n in c)
            )
        G = G.subgraph(comp_nodes).copy()

        if len(G) > 0:
            break

    # 엣지 비용 정의(가로등/인도 보너스)
    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 1.0))
        lit = str(data.get("lit", "")).lower() in ("yes", "24/7", "automatic", "true")
        sidewalk = str(data.get("sidewalk", "")).lower()
        has_sidewalk = sidewalk not in ("", "no", "none", "0")

        bonus = (0.25 if lit else 0.0) + (0.25 if has_sidewalk else 0.0)
        data["length"]    = length
        data["cost_base"] = length / (1.0 + bonus) + random.random() * 0.5
        data["cost_alt"]  = length / (1.0 + bonus) + random.random() * 0.8

    return G

# --------------------------------------------------------
# 2) 유틸
# --------------------------------------------------------
def path_length(G, path: List[int]) -> float:
    total = 0.0
    if not path or len(path) < 2:
        return 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if data is None:
            try:
                sp = nx.shortest_path(G, u, v, weight="length", method="dijkstra")
            except Exception:
                return float("inf")
            for a, b in zip(sp[:-1], sp[1:]):
                d2 = min(G.get_edge_data(a, b).values(), key=lambda d: d.get("length", 1.0))
                total += float(d2.get("length", 1.0))
        else:
            d = min(data.values(), key=lambda d: d.get("length", 1.0))
            total += float(d.get("length", 1.0))
    return total

def nodes_to_latlngs(G, path: List[int]):
    return [{"lat": G.nodes[n]["y"], "lng": G.nodes[n]["x"]} for n in path]

def _edge_set_undirected(path: List[int]):
    return {frozenset((u, v)) for u, v in zip(path[:-1], path[1:])}

def _jaccard(a, b):
    inter = len(a & b)
    return 0.0 if inter == 0 else inter / len(a | b)

# --------------------------------------------------------
# 3) 보정/페널티
# --------------------------------------------------------
def repair_path(G, path):
    if not path or len(path) < 2:
        return path
    new_path = [path[0]]
    for v in path[1:]:
        u = new_path[-1]
        if G.has_edge(u, v) or G.has_edge(v, u):
            new_path.append(v)
        else:
            try:
                sp = nx.shortest_path(G, u, v, weight="length", method="dijkstra")
            except Exception:
                return None
            new_path.extend(sp[1:])
    return new_path

def penalize_p1_for_return(G, p1, factor_edges=40.0, factor_nodes=10.0):
    """
    p1 구간에 페널티를 줘서 귀환 경로를 다른 길로 유도.
    - 엣지(멀티엣지 포함): cost_alt × factor_edges
    - p1에 포함된 노드의 in/out 엣지: cost_alt × factor_nodes
    """
    Gp = G.copy()

    # ✅ 엣지 페널티: MultiDiGraph에서는 get_edge_data(u, v)로 key별 attr dict 접근
    for u, v in zip(p1[:-1], p1[1:]):
        ed = Gp.get_edge_data(u, v)  # dict: {key: attrdict, ...} or None
        if ed:
            for k, data in ed.items():
                data["cost_alt"] = float(data.get("cost_alt", data.get("cost_base", 1.0))) * factor_edges

    # ✅ 노드 페널티: out_edges / in_edges는 data=True만 사용 (keys는 불필요)
    for n in p1:
        for _, v, data in Gp.out_edges(n, data=True):
            data["cost_alt"] = float(data.get("cost_alt", data.get("cost_base", 1.0))) * factor_nodes
        for u, _, data in Gp.in_edges(n, data=True):
            data["cost_alt"] = float(data.get("cost_alt", data.get("cost_base", 1.0))) * factor_nodes

    return Gp

def ensure_closed_loop(G, nodes, s_node, weight="cost_alt"):
    if not nodes:
        return None
    if nodes[0] == nodes[-1]:
        return nodes
    try:
        back = nx.shortest_path(G, nodes[-1], s_node, weight=weight, method="dijkstra")
    except Exception:
        return None
    return nodes + back[1:]

# --------------------------------------------------------
# 4) 미세 최적화 (±30m)
# --------------------------------------------------------
def two_opt_once(G, path: List[int], target_m: float):
    n = len(path)
    if n < 6:
        return None
    base_obj = abs(path_length(G, path) - target_m)
    for i in range(1, n - 3):
        for j in range(i + 1, n - 1):
            if j - i < 2:
                continue
            cand = path[:i] + path[i:j][::-1] + path[j:]
            cand = repair_path(G, cand)
            if cand is None:
                continue
            if cand[0] != cand[-1]:
                cand = cand + [cand[0]]
            obj = abs(path_length(G, cand) - target_m)
            if obj + 1e-6 < base_obj:
                return cand
    return None

def or_opt_once(G, path: List[int], target_m: float, k_max: int = 3):
    n = len(path)
    if n < 6:
        return None
    base_obj = abs(path_length(G, path) - target_m)
    for k in range(1, k_max + 1):
        for i in range(1, n - 1 - k):
            seg = path[i : i + k]
            base = path[:i] + path[i + k :]
            for j in range(1, len(base)):
                cand = base[:j] + seg + base[j:]
                cand = repair_path(G, cand)
                if cand is None:
                    continue
                if cand[0] != cand[-1]:
                    cand = cand + [cand[0]]
                obj = abs(path_length(G, cand) - target_m)
                if obj + 1e-6 < base_obj:
                    return cand
    return None

def refine_length_by_local_search(G, path: List[int], target_m: float,
                                  tol_m: float = 30.0, max_outer: int = 20):
    cur = repair_path(G, path) or path[:]
    if cur[0] != cur[-1]:
        cur = cur + [cur[0]]
    for _ in range(max_outer):
        if abs(path_length(G, cur) - target_m) <= tol_m:
            break
        improved = False
        cand = two_opt_once(G, cur, target_m)
        if cand is not None:
            cur = cand; improved = True
        if not improved:
            cand = or_opt_once(G, cur, target_m, k_max=3)
            if cand is not None:
                cur = cand; improved = True
        if not improved:
            break
    return cur

# --------------------------------------------------------
# 5) 후보 선정/삼각 fallback
# --------------------------------------------------------
def _pick_mid_candidates(G, s_node: int, target_m: float, top_k: int = 200):
    mid_low, mid_high = 0.45 * target_m, 0.75 * target_m
    cutoff = mid_high * 1.2
    try:
        dist_map = nx.single_source_dijkstra_path_length(G, s_node, cutoff=cutoff, weight="length")
    except Exception:
        dist_map = {}
    cand = [n for n, d in dist_map.items() if mid_low <= d <= mid_high]
    if not cand:
        mid_low2, mid_high2 = 0.30 * target_m, 0.90 * target_m
        dist_map2 = nx.single_source_dijkstra_path_length(G, s_node, cutoff=mid_high2*1.2, weight="length")
        cand = [n for n, d in dist_map2.items() if mid_low2 <= d <= mid_high2]
    random.shuffle(cand)
    return cand[:top_k]

def _bearing(y1, x1, y2, x2):
    dlon = math.radians(x2 - x1)
    lat1 = math.radians(y1); lat2 = math.radians(y2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1)*math.cos(lat2) - math.sin(lat1)*math.sin(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def _tri_loop_fallback(G, s_node: int, target_m: float):
    low, high = 0.30 * target_m, 0.60 * target_m
    dist_map = nx.single_source_dijkstra_path_length(G, s_node, cutoff=high*1.3, weight="length")
    cand = [n for n, d in dist_map.items() if low <= d <= high]
    if len(cand) < 2:
        dist_map = nx.single_source_dijkstra_path_length(G, s_node, cutoff=0.9*target_m, weight="length")
        cand = [n for n, d in dist_map.items() if 0.2*target_m <= d <= 0.9*target_m]
        if len(cand) < 2:
            raise RuntimeError("fallback용 후보가 부족합니다.")
    sx, sy = G.nodes[s_node]["x"], G.nodes[s_node]["y"]
    random.shuffle(cand)
    a = cand[0]; a_brg = _bearing(sy, sx, G.nodes[a]["y"], G.nodes[a]["x"])
    b = None
    for n in cand[1:]:
        brg = _bearing(sy, sx, G.nodes[n]["y"], G.nodes[n]["x"])
        diff = abs((brg - a_brg + 540) % 360 - 180)
        if diff >= 110:
            b = n; break
    if b is None: b = cand[1]
    p_sa = nx.shortest_path(G, s_node, a, weight="cost_base", method="dijkstra")
    p_ab = nx.shortest_path(G, a, b,       weight="cost_alt",  method="dijkstra")
    p_bs = nx.shortest_path(G, b, s_node,  weight="cost_alt",  method="dijkstra")
    return p_sa + p_ab[1:] + p_bs[1:]

# ====== [추가] 형상/길이 보정 유틸 ======
def _polyline_turn_sum(G, path):
    """경로의 총 꺾임(도). 각도 변화 누적."""
    if len(path) < 3: return 0.0
    def brg(n1, n2):
        y1,x1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
        y2,x2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
        dlon = math.radians(x2 - x1)
        lat1 = math.radians(y1); lat2 = math.radians(y2)
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1)*math.cos(lat2) - math.sin(lat1)*math.sin(lat2)*math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    turns = 0.0
    for a,b,c in zip(path[:-2], path[1:-1], path[2:]):
        th1 = brg(a,b); th2 = brg(b,c)
        diff = abs((th2 - th1 + 540) % 360 - 180)
        turns += diff
    return turns

def _looks_out_and_back(G, p1, p2):
    """왕복처럼 보이면 True (겹침률 높거나 꺾임 매우 적음)."""
    overlap = _jaccard(_edge_set_undirected(p1), _edge_set_undirected(p2))
    if overlap >= 0.6:
        return True
    loop = p1 + p2[1:]
    if _polyline_turn_sum(G, loop) < 90.0:  # 거의 일자
        return True
    return False

def _shortcut_if_too_long(G, path, excess_m, trials=20):
    """길이가 너무 길면 임의 구간을 최단경로로 단축."""
    base = path[:]
    L0 = path_length(G, base)
    target = L0 - excess_m
    best = base; best_err = abs(L0 - target)
    N = len(base)
    import random
    for _ in range(trials):
        i = random.randrange(1, N-4)
        j = random.randrange(i+2, N-1)
        try:
            sp = nx.shortest_path(G, base[i], base[j], weight="cost_base", method="dijkstra")
        except Exception:
            continue
        cand = base[:i] + sp + base[j:]
        cand = repair_path(G, cand) or cand
        if cand[0] != cand[-1]: cand = cand + [cand[0]]
        L = path_length(G, cand)
        err = abs(L - target)
        if err < best_err:
            best, best_err = cand, err
            if err < 50: break
    return best

def _pad_if_too_short(G, path, deficit_m):
    """짧으면 시작점 근처 짧은 스퍼 왕복으로 패딩."""
    return _pad_distance_out_and_back(G, path, path_length(G, path)+deficit_m, overshoot_ok=deficit_m+50.0)

def adjust_length_to_tolerance(G, path, target_m, tol_m=400.0):
    """±tol_m 안으로 들어오도록 대략 보정 → 이후 미세 최적화."""
    L = path_length(G, path)
    if L > target_m + tol_m:
        path = _shortcut_if_too_long(G, path, L - (target_m + tol_m))
    elif L < target_m - tol_m:
        path = _pad_if_too_short(G, path, (target_m - tol_m) - L)
    return ensure_closed_loop(G, path, path[0]) or path

# --------------------------------------------------------
# 6) 메인: 루프 생성
# --------------------------------------------------------
def make_loop_route(G, s_node: int, target_km: float):
    TARGET_TOL_M = 400.0
    target_m = target_km * 1000.0
    good = _pick_mid_candidates(G, s_node, target_m, top_k=220)

    def finalize(loop):
        # 1) 대략 보정(±400m 진입)
        loop = adjust_length_to_tolerance(G, loop, target_m, tol_m=TARGET_TOL_M)
        # 2) 미세 보정(±30m 시도; 실패해도 최소 ±400m는 유지)
        loop = refine_length_by_local_search(G, loop, target_m, tol_m=30.0, max_outer=30)
        L = path_length(G, loop)
        # 최종 제약: ±400m
        if abs(L - target_m) <= TARGET_TOL_M:
            return loop, L
        return None, None

    # 후보가 없으면 삼각 fallback
    if not good:
        loop = _tri_loop_fallback(G, s_node, target_m)
        ans = finalize(loop)
        if ans[0]: return ans
        # 마지막 시도: 그냥 반환(±400 실패하지만 최소 닫힘)
        loop = ensure_closed_loop(G, loop, s_node) or loop
        return loop, path_length(G, loop)

    best = None; best_err = 1e12

    for t in good[:200]:
        # s->t
        try:
            p1 = nx.shortest_path(G, s_node, t, weight="cost_base", method="dijkstra")
        except Exception:
            continue

        # 귀환 후보
        Gp = penalize_p1_for_return(G, p1, factor_edges=40.0, factor_nodes=10.0)
        try:
            p2_list = list(islice(nx.shortest_simple_paths(Gp, t, s_node, weight="cost_alt"), 25))
        except Exception:
            continue

        # 왕복처럼 보이는 후보는 패스
        p2_list = [p2 for p2 in p2_list if not _looks_out_and_back(G, p1, p2)]
        if not p2_list:
            continue

        # 겹침률 순으로 정렬해 다양한 루프 우선
        e1 = _edge_set_undirected(p1)
        p2_list.sort(key=lambda p2: _jaccard(e1, _edge_set_undirected(p2)))

        for p2 in p2_list[:6]:
            loop = p1 + p2[1:]
            loop = ensure_closed_loop(G, loop, s_node, weight="cost_alt")
            if loop is None:
                continue

            # 길이 제약/미세 보정
            cand, L = finalize(loop)
            if cand:
                return cand, L  # ±400m 통과 → 바로 채택

            # 베스트(목표와의 오차 최소)를 기록 (혹시 마지막에라도 리턴)
            Lraw = path_length(G, loop)
            err = abs(Lraw - target_m)
            if err < best_err:
                best_err, best = err, loop

    # 모두 실패 → 삼각 fallback
    loop = _tri_loop_fallback(G, s_node, target_m)
    cand, L = finalize(loop)
    if cand:
        return cand, L
    # 그래도 실패하면 닫힌 루프라도
    loop = ensure_closed_loop(G, loop, s_node) or loop
    return loop, path_length(G, loop)


# --------------------------------------------------------
# 7) 턴 분석/거리/안내문 생성
# --------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """단순 해버사인 거리(m)."""
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _bearing_from_coords(a: Dict[str, float], b: Dict[str, float]) -> float:
    return _bearing(a["lat"], a["lng"], b["lat"], b["lng"])


def _signed_turn_angle(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
    th1 = _bearing_from_coords(a, b)
    th2 = _bearing_from_coords(b, c)
    return ((th2 - th1 + 540) % 360) - 180  # [-180, 180], +: left, -: right


def _cumulative_distances(polyline: List[Dict[str, float]]) -> List[float]:
    dists = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        dists.append(dists[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return dists


def _format_instruction(distance_m: float, turn_type: str) -> str:
    dist_rounded = int(round(distance_m / 10.0) * 10)
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


def polyline_to_turns(polyline: List[Dict[str, float]],
                      straight_thresh: float = 15.0,
                      turn_thresh: float = 30.0,
                      uturn_thresh: float = 150.0) -> List[Dict[str, float]]:
    """
    polyline을 이용해 턴 포인트, 남은 거리, 안내문을 생성.

    - straight_thresh: 이보다 작은 각도 변화는 무시.
    - turn_thresh: 좌/우로 분류되는 최소 각도 변화.
    - uturn_thresh: U턴으로 간주하는 각도.
    """
    if not polyline or len(polyline) < 2:
        return []

    cumulative = _cumulative_distances(polyline)
    turns = []
    last_turn_idx = 0

    for i in range(1, len(polyline) - 1):
        angle = _signed_turn_angle(polyline[i - 1], polyline[i], polyline[i + 1])
        angle_abs = abs(angle)
        if angle_abs < straight_thresh:
            continue

        if angle_abs >= uturn_thresh:
            t_type = "uturn"
        elif angle_abs >= turn_thresh:
            t_type = "left" if angle > 0 else "right"
        else:
            t_type = "straight"

        dist_to_turn = cumulative[i] - cumulative[last_turn_idx]
        turns.append({
            "lat": polyline[i]["lat"],
            "lng": polyline[i]["lng"],
            "type": t_type,
            "distance": round(dist_to_turn, 1),
            "instruction": _format_instruction(dist_to_turn, t_type),
        })
        last_turn_idx = i

    # 도착 안내 (마지막 점)
    final_dist = cumulative[-1] - cumulative[last_turn_idx]
    turns.append({
        "lat": polyline[-1]["lat"],
        "lng": polyline[-1]["lng"],
        "type": "arrive",
        "distance": round(final_dist, 1),
        "instruction": _format_instruction(final_dist, "arrive"),
    })

    return turns


def build_turn_by_turn(polyline: List[Dict[str, float]],
                       km_requested: float,
                       pace_min_per_km: float = 8.0,
                       total_length_m: float = None):
    """턴 목록과 요약 메타를 생성."""
    turns = polyline_to_turns(polyline)
    length_m = total_length_m if total_length_m is not None else (_cumulative_distances(polyline)[-1] if polyline else 0.0)
    summary = {
        "length_m": round(length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round((length_m / 1000.0) * pace_min_per_km, 1),
        "turn_count": len([t for t in turns if t.get("type") != "arrive"]),
    }
    return turns, summary
