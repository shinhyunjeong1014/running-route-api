############################################################
#                 Running Route Generator v3
#       Cycle + Rod Hybrid Algorithm (K-Walk Optimized)
############################################################

import math
import random
import time
from typing import List, Tuple, Dict, Any

import networkx as nx
import osmnx as ox

# ==========================
# 거리 계산 (Haversine)
# ==========================
import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 점 사이의 대원거리 (미터)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def polyline_length_m(polyline):
    """polyline = [(lat, lng), ...] 배열의 총 길이 계산."""
    if not polyline or len(polyline) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        total += haversine(lat1, lon1, lat2, lon2)
    return float(total)

############################################################
# JSON Safe
############################################################

def safe_float(x):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
    return x

def safe_list(v):
    if isinstance(v, list):
        return [safe_list(x) for x in v]
    if isinstance(v, dict):
        return safe_dict(v)
    if isinstance(v, float):
        return safe_float(v)
    return v

def safe_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = safe_list(v)
        elif isinstance(v, dict):
            out[k] = safe_dict(v)
        elif isinstance(v, float):
            out[k] = safe_float(v)
        else:
            out[k] = v
    return out


############################################################
# Haversine Distance
############################################################

def haversine(a, b, x, y):
    R = 6371000.0
    dphi = math.radians(x - a)
    dlambda = math.radians(y - b)
    phi1 = math.radians(a)
    phi2 = math.radians(x)
    h = (math.sin(dphi/2)**2
         + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2)
    return 2*R*math.asin(math.sqrt(h))

def polyline_length(poly):
    if len(poly) < 2:
        return 0.0
    L = 0
    for (a, b), (c, d) in zip(poly[:-1], poly[1:]):
        L += haversine(a, b, c, d)
    return L


############################################################
# Roundness
############################################################

def _to_xy(poly):
    if not poly:
        return []
    R = 6371000.0
    ref_lat = poly[0][0]
    ref_lon = poly[0][1]
    cos_lat = math.cos(math.radians(ref_lat))
    out = []
    for lat, lon in poly:
        x = (lon - ref_lon) * cos_lat * R
        y = (lat - ref_lat) * R
        out.append((x, y))
    return out

def roundness(poly):
    if len(poly) < 4:
        return 0.0
    xy = _to_xy(poly)
    area = 0.0
    P = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i+1)%n]
        area += x1*y2 - x2*y1
        P += math.hypot(x2-x1, y2-y1)
    area = abs(area)/2
    if area == 0 or P == 0:
        return 0.0
    return 4*math.pi*area/(P*P)


############################################################
# Overlap / Curve Penalty
############################################################

def overlap_fraction(nodes):
    if len(nodes) < 2:
        return 0.0
    cnt = {}
    for u, v in zip(nodes[:-1], nodes[1:]):
        e = (u, v) if u <= v else (v, u)
        cnt[e] = cnt.get(e, 0) + 1
    over = sum(1 for c in cnt.values() if c > 1)
    return over / len(cnt) if cnt else 0.0

def curve_penalty(nodes, G):
    if len(nodes) < 3:
        return 0.0
    s = 0.0
    for i in range(1, len(nodes)-1):
        a, b, c = nodes[i-1], nodes[i], nodes[i+1]
        if a not in G or b not in G or c not in G:
            continue
        ya, xa = G.nodes[a]["y"], G.nodes[a]["x"]
        yb, xb = G.nodes[b]["y"], G.nodes[b]["x"]
        yc, xc = G.nodes[c]["y"], G.nodes[c]["x"]

        v1x = xb - xa
        v1y = yb - ya
        v2x = xc - xb
        v2y = yc - yb

        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0 or n2 == 0:
            continue

        dot = (v1x*v2x + v1y*v2y)/(n1*n2)
        dot = max(-1, min(1, dot))
        theta = math.degrees(math.acos(dot))
        if theta < 60:
            s += (60 - theta)/60
    return s


############################################################
# Path Length on Graph
############################################################

def graph_path_length(G, nodes):
    L = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if v in G[u]:
            best = min(G[u][v].values(), key=lambda x: x.get("length", 1))
            L += best.get("length", 0)
    return L


############################################################
# Convert Nodes → Polyline
############################################################

def nodes_to_polyline(G, nodes):
    return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in nodes]


############################################################
# Step 1 — Pedestrian Graph Builder
############################################################

def build_graph(lat, lng, km):
    dist = min(max(km*800 + 600, 900), 4200)

    custom = (
        '["highway"~"footway|path|sidewalk|cycleway|steps|pedestrian|track|'
        'service|residential|living_street|alley"]'
    )

    G = ox.graph_from_point(
        (lat, lng),
        dist=dist,
        network_type="walk",
        custom_filter=custom,
        simplify=True,
        retain_all=False
    )

    G.remove_nodes_from(list(nx.isolates(G)))
    if not G.nodes:
        raise ValueError("Empty pedestrian graph")

    return G


############################################################
# Step 2 — Rod Candidates (Start → Far Node)
############################################################

def get_rod_candidates(G, start_node, target):
    UG = ox.utils_graph.get_undirected(G)
    dist = nx.single_source_dijkstra_path_length(
        UG, start_node, cutoff=target*0.9, weight="length"
    )
    cand = [(n, d) for n, d in dist.items() if target*0.25 <= d <= target*0.75]
    cand.sort(key=lambda x: x[1])
    return [c[0] for c in cand[:60]]


############################################################
# Step 3 — Cycle Near Rod Node
############################################################

def get_cycle_near(G, rod_node, target):
    # BFS local ball for cycle search
    ball = nx.single_source_shortest_path_length(G, rod_node, cutoff=8)
    H = G.subgraph(ball.keys()).copy()
    if H.number_of_nodes() < 4:
        return []
    Hs = nx.Graph(H)
    cyc = nx.cycle_basis(Hs)
    out = []
    for c in cyc:
        if len(c) < 3:
            continue
        closed = c + [c[0]]
        L = graph_path_length(Hs, closed)
        if target*0.15 <= L <= target*1.8:
            out.append(closed)
    return out


############################################################
# Step 4 — Build Hybrid Loop (Rod + Cycle*N + Rod)
############################################################

def build_hybrid(G, start_node, rod_node, cycle_nodes, target):
    UG = ox.utils_graph.get_undirected(G)

    try:
        rod_fwd = nx.shortest_path(UG, start_node, rod_node, weight="length")
    except:
        return None

    rod_len = graph_path_length(UG, rod_fwd)
    if rod_len <= 0:
        return None

    best = None
    best_meta = None
    tol = target * 0.05

    for rep in range(1, 7):   # cycle repetition 1~6
        nodes = []
        nodes.extend(rod_fwd)

        for _ in range(rep):
            nodes.extend(cycle_nodes[1:])  # close cycle

        back = list(reversed(rod_fwd))
        nodes.extend(back[1:])

        poly = nodes_to_polyline(G, nodes)
        L = polyline_length(poly)
        err = abs(L - target)
        length_ok = err <= tol

        rnd = roundness(poly)
        ov = overlap_fraction(nodes)
        curv = curve_penalty(nodes, G)

        # scoring
        score = (
            rnd*3.0
            - ov*2.0
            - curv*0.25
            - (err/target)*8.0
        )

        if (best is None) or (score > best_meta["score"]):
            best = poly
            best_meta = {
                "len": L,
                "err": err,
                "roundness": rnd,
                "overlap": ov,
                "curve_penalty": curv,
                "score": score,
                "length_ok": length_ok,
            }

        if length_ok:
            break

    return best, best_meta


############################################################
# Step 5 — Fallback
############################################################

def fallback(lat, lng, km):
    target = km*1000
    side = target/4
    dlat = side / 111000
    dlon = dlat / math.cos(math.radians(lat))

    A = (lat+dlat, lng)
    B = (lat, lng+dlon)
    C = (lat-dlat, lng)
    D = (lat, lng-dlon)
    poly = [A, B, C, D, A]

    L = polyline_length(poly)
    r = roundness(poly)
    return poly, L, r


############################################################
# Main
############################################################

def generate_area_loop(lat, lng, km):
    start_time = time.time()
    target = km*1000
    tol = target*0.05

    meta = dict(
        len=None, err=None, roundness=None, overlap=None, curve_penalty=None,
        score=None, success=False, length_ok=False, used_fallback=False,
        valhalla_calls=0, kakao_calls=0, routes_checked=0, routes_validated=0,
        km_requested=km, target_m=target, time_s=None, message=""
    )

    # 1) Build graph
    try:
        G = build_graph(lat, lng, km)
        start_node = ox.distance.nearest_nodes(G, X=lng, Y=lat)
    except Exception as e:
        poly, L, r = fallback(lat, lng, km)
        err = abs(L - target)
        meta.update(dict(
            len=L, err=err, roundness=r, overlap=0.0, curve_penalty=0.0,
            score=r, success=False, length_ok=(err <= tol),
            used_fallback=True,
            message=f"Graph fail: {e}"
        ))
        meta["time_s"] = time.time() - start_time
        return safe_list(poly), safe_dict(meta)

    UG = ox.utils_graph.get_undirected(G)

    # 2) Rod candidates
    rods = get_rod_candidates(G, start_node, target)
    if not rods:
        poly, L, r = fallback(lat, lng, km)
        err = abs(L-target)
        meta.update(dict(
            len=L, err=err, roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True, message="No rod candidates"
        ))
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    best = None
    best_meta = None

    # 3) Try rods + cycles
    for rod in rods:
        # cycles near rod
        cycles = get_cycle_near(G, rod, target)
        if not cycles:
            continue

        for cyc in cycles:
            out = build_hybrid(G, start_node, rod, cyc, target)
            meta["routes_checked"] += 1
            if out is None:
                continue

            poly, m = out
            if m["length_ok"]:
                meta["routes_validated"] += 1

            # Update best
            if (best is None) or (m["score"] > best_meta["score"]):
                best = poly
                best_meta = m

            if m["length_ok"]:
                break

    if best is None:
        poly, L, r = fallback(lat, lng, km)
        err = abs(L-target)
        meta.update(dict(
            len=L, err=err, roundness=r, overlap=0, curve_penalty=0,
            score=r, used_fallback=True, message="No valid loops"
        ))
        meta["time_s"] = time.time()-start_time
        return safe_list(poly), safe_dict(meta)

    # 4) Force insert real start coordinate AT FIRST
    best = [(lat, lng)] + best

    L = polyline_length(best)
    err = abs(L-target)
    rnd = roundness(best)

    meta.update(best_meta)
    meta.update(dict(
        len=L, err=err, roundness=rnd,
        success=(err <= tol),
        length_ok=(err <= tol),
        used_fallback=False,
        message="Optimal hybrid rod+cycle route generated"
    ))
    meta["time_s"] = time.time()-start_time

    return safe_list(best), safe_dict(meta)

