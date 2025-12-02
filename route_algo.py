"""
Route generation algorithm for pedestrian running/walking loops.

Design goals
-----------
1. Respect requested distance as tightly as possible (±5% if feasible).
2. Generate visually "loop-like" routes using real pedestrian ways:
   - Avoid straight out-and-back overlaps (large shared segments).
   - Prefer closed shapes with some area ("roundness").
3. Always return routes whose polyline starts/ends at the requested (lat, lng).

Implementation overview
-----------------------
- Build a pedestrian graph from OpenStreetMap with osmnx (network_type="walk")
  and a custom filter that excludes motorways, trunks, etc.
- From the start node, run Dijkstra to get nodes in a distance band
  near half the requested length (for loop construction).
- For each candidate "pivot" node p:
    forward:  shortest path start -> p   (weight="length")
    backward: shortest path  p -> start  (weight="length_penalized"),
              where edges used in the forward path get a multiplicative
              penalty, so the backward path tends to use different streets.
  This yields a loop start -> ... -> p -> ... -> start.
- For each loop we compute:
    L          : total length (meters)
    length_err : |L - target_m| / target_m
    overlap    : fraction of edges that are reused
    roundness  : 4πA / P^2  in [0, 1], where A, P are area & perimeter
    uturn      : fraction of strong back-angles (> 150°)
- A scalar score combines these metrics. The best route whose length_err ≤ 0.05
  is preferred; if none exists we choose the best overall or fall back
  to a geometric rectangular loop around the start.

The public API is:

    generate_area_loop(lat: float, lng: float, km: float)
        -> (polyline: List[Tuple[float, float]], meta: Dict)

    polyline_length_m(polyline: List[Tuple[float, float]]) -> float

These are the only symbols used by app.py.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

try:
    import osmnx as ox  # type: ignore
except Exception:  # pragma: no cover - osmnx must be installed in runtime
    ox = None  # type: ignore


LatLng = Tuple[float, float]
NodeId = int


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def polyline_length_m(points: Sequence[LatLng]) -> float:
    """Length of a polyline in meters using haversine metric."""
    if not points or len(points) < 2:
        return 0.0
    total = 0.0
    prev = points[0]
    for pt in points[1:]:
        total += _haversine_m(prev[0], prev[1], pt[0], pt[1])
        prev = pt
    return total


def _to_local_xy(points: Sequence[LatLng], origin: Optional[LatLng] = None) -> List[Tuple[float, float]]:
    """Approximate projection to local tangent plane (meters)."""
    if not points:
        return []
    if origin is None:
        origin = points[0]
    lat0, lon0 = origin
    R = 6371000.0
    cos_lat0 = math.cos(math.radians(lat0))
    res: List[Tuple[float, float]] = []
    for lat, lon in points:
        dx = math.radians(lon - lon0) * R * cos_lat0
        dy = math.radians(lat - lat0) * R
        res.append((dx, dy))
    return res


def _roundness(points: Sequence[LatLng]) -> float:
    """
    4πA / P^2, where A is polygon area and P is perimeter.
    Values: 1 for a perfect circle, ~0 for a degenerate line.
    """
    if len(points) < 4:
        return 0.0

    # Ensure closed polygon in local XY
    xy = _to_local_xy(points)
    if xy[0] != xy[-1]:
        xy = xy + [xy[0]]

    # Perimeter
    perim = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
        dx = x2 - x1
        dy = y2 - y1
        perim += math.hypot(dx, dy)
    if perim <= 0:
        return 0.0

    # Shoelace formula for area
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
        area2 += x1 * y2 - x2 * y1
    area = abs(area2) / 2.0

    return max(0.0, min(1.0, 4.0 * math.pi * area / (perim * perim)))


def _uturn_fraction(points: Sequence[LatLng]) -> float:
    """
    Fraction of vertices that form a strong back-angle (≈ U-turn).
    We consider angles > 150 degrees as U-turn-ish.
    """
    n = len(points)
    if n < 3:
        return 0.0

    xy = _to_local_xy(points)
    uturns = 0
    usable = 0
    for i in range(1, n - 1):
        x0, y0 = xy[i - 1]
        x1, y1 = xy[i]
        x2, y2 = xy[i + 1]

        v1x, v1y = x0 - x1, y0 - y1
        v2x, v2y = x2 - x1, y2 - y1
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        usable += 1
        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        angle_deg = math.degrees(math.acos(dot))
        if angle_deg > 150.0:
            uturns += 1

    if usable == 0:
        return 0.0
    return uturns / usable


# -----------------------------------------------------------------------------
# OSM graph construction
# -----------------------------------------------------------------------------


@dataclass
class GraphBuildParams:
    dist_m: float
    network_type: str = "walk"
    custom_filter: Optional[str] = None


def _build_pedestrian_graph(lat: float, lng: float, km: float) -> nx.Graph:
    """
    Build an undirected pedestrian graph around (lat, lng).

    - Uses OSMnx network_type="walk".
    - Applies a restrictive custom_filter to keep only walkable ways.
    - Ensures each edge has a 'length' attribute in meters.
    """
    if ox is None:
        raise RuntimeError("osmnx is not available in this environment")

    # Radius selection: a bit larger than half the requested perimeter
    target_m = max(300.0, km * 1000.0)
    dist_m = target_m * 0.7 + 400.0

    cf = (
        '["highway"~"footway|path|sidewalk|cycleway|steps|pedestrian|track|'
        'service|residential|living_street|unclassified|tertiary|secondary|primary"]'
        '["motor_vehicle"!~"yes"]["access"!~"no|private"]'
    )

    G = ox.graph_from_point(
        (lat, lng),
        dist=dist_m,
        dist_type="network",
        network_type="walk",
        simplify=True,
        retain_all=True,
        custom_filter=cf,
    )

    # Convert to undirected to allow free movement in both directions
    G_u = ox.utils_graph.get_undirected(G)

    # Ensure each edge has a 'length'
    nodes = G_u.nodes
    for u, v, data in G_u.edges(data=True):
        if "length" not in data:
            lat1 = nodes[u].get("y")
            lon1 = nodes[u].get("x")
            lat2 = nodes[v].get("y")
            lon2 = nodes[v].get("x")
            if None not in (lat1, lon1, lat2, lon2):
                data["length"] = _haversine_m(lat1, lon1, lat2, lon2)
            else:
                data["length"] = float(data.get("distance", 1.0))

        # Also keep a penalizable copy
        if "length_penalized" not in data:
            data["length_penalized"] = float(data["length"])

    return G_u


def _nearest_node(G: nx.Graph, lat: float, lng: float) -> NodeId:
    """Get nearest graph node to (lat, lng)."""
    # osmnx >= 1.6: nearest_nodes(x, y)
    return ox.distance.nearest_nodes(G, X=[lng], Y=[lat])[0]  # type: ignore


# -----------------------------------------------------------------------------
# Route scoring
# -----------------------------------------------------------------------------


@dataclass
class RouteScore:
    length_m: float
    length_err_ratio: float
    overlap_ratio: float
    roundness: float
    uturn_ratio: float
    score: float


def _edge_overlap_ratio(path_nodes: Sequence[NodeId]) -> float:
    """
    Fraction of traversed *edges* that are reused at least once.
    0 means every edge is unique; 1 means the whole route is perfectly out-and-back.
    """
    if len(path_nodes) < 2:
        return 0.0

    seen = set()
    repeated = 0
    total = 0

    for u, v in zip(path_nodes, path_nodes[1:]):
        if u == v:
            continue
        e = (u, v) if u <= v else (v, u)  # undirected edge id
        total += 1
        if e in seen:
            repeated += 1
        else:
            seen.add(e)

    if total == 0:
        return 0.0
    return repeated / total


def _score_route(
    polyline: Sequence[LatLng],
    path_nodes: Sequence[NodeId],
    target_m: float,
) -> RouteScore:
    L = polyline_length_m(polyline)
    if L <= 0.0:
        return RouteScore(0.0, 1.0, 1.0, 0.0, 0.0, -1e9)

    length_err_ratio = abs(L - target_m) / target_m
    overlap_ratio = _edge_overlap_ratio(path_nodes)
    roundness = _roundness(polyline)
    uturn_ratio = _uturn_fraction(polyline)

    # Scoring weights (empirically tuned for "quality first")
    # Lower is worse, higher is better.
    score = 0.0
    # Strongly prioritize length accuracy
    score -= 5.0 * length_err_ratio  # ±5% -> -0.25
    # Avoid overlapping edges (out-and-back)
    score -= 3.0 * overlap_ratio
    # Avoid visual U-turns
    score -= 2.0 * uturn_ratio
    # Reward roundness above a small baseline (0.2)
    score += 2.0 * max(0.0, roundness - 0.2)

    return RouteScore(L, length_err_ratio, overlap_ratio, roundness, uturn_ratio, score)


# -----------------------------------------------------------------------------
# Route construction (forward+penalized-backward loops)
# -----------------------------------------------------------------------------


def _single_source_distances(
    G: nx.Graph, source: NodeId, cutoff: float
) -> Dict[NodeId, float]:
    """Dijkstra distances from source up to cutoff meters."""
    return nx.single_source_dijkstra_path_length(G, source, cutoff=cutoff, weight="length")


def _build_penalized_lengths(G: nx.Graph, used_edges: Iterable[Tuple[NodeId, NodeId]], penalty: float) -> None:
    """
    For edges in used_edges, multiply their 'length_penalized'.
    Operates in-place on G.
    """
    edge_set = set()
    for u, v in used_edges:
        if u == v:
            continue
        e = (u, v) if u <= v else (v, u)
        edge_set.add(e)

    for u, v, data in G.edges(data=True):
        e = (u, v) if u <= v else (v, u)
        base = float(data.get("length", 1.0))
        if e in edge_set:
            data["length_penalized"] = base * penalty
        else:
            data["length_penalized"] = base


def _nodes_to_polyline(G: nx.Graph, path_nodes: Sequence[NodeId]) -> List[LatLng]:
    pts: List[LatLng] = []
    nodes = G.nodes
    for nid in path_nodes:
        info = nodes[nid]
        lat = float(info.get("y"))
        lng = float(info.get("x"))
        pts.append((lat, lng))
    return pts


def _generate_loops(
    G: nx.Graph,
    start_node: NodeId,
    target_m: float,
    max_candidates: int = 40,
    backtrack_penalty: float = 4.0,
) -> List[Tuple[List[LatLng], List[NodeId], RouteScore]]:
    """
    Generate candidate loops by:
    - picking pivot nodes at ~half the target distance from start
    - forward shortest path start->pivot (length)
    - backward shortest path pivot->start with penalized edges
    """
    # 1) Precompute distances from start (forward tree)
    max_one_way = target_m * 0.7
    dist = _single_source_distances(G, start_node, cutoff=max_one_way)

    # Candidate pivots roughly around half perimeter
    min_d = target_m * 0.35
    max_d = target_m * 0.65
    candidates = [n for n, d in dist.items() if min_d <= d <= max_d]

    if not candidates:
        # fallback: just pick farthest reachable nodes up to max_one_way
        sorted_nodes = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [n for n, d in sorted_nodes[:max_candidates] if d > target_m * 0.2]

    random.shuffle(candidates)
    candidates = candidates[:max_candidates]

    results: List[Tuple[List[LatLng], List[NodeId], RouteScore]] = []

    for pivot in candidates:
        try:
            # Forward path
            path_f = nx.shortest_path(G, start_node, pivot, weight="length")
        except nx.NetworkXNoPath:
            continue

        # Penalize edges in the forward path
        forward_edges = list(zip(path_f, path_f[1:]))
        _build_penalized_lengths(G, forward_edges, penalty=backtrack_penalty)

        try:
            path_b = nx.shortest_path(G, pivot, start_node, weight="length_penalized")
        except nx.NetworkXNoPath:
            continue

        # Build full node path (avoid duplicate pivot)
        full_nodes = list(path_f) + list(path_b[1:])
        if len(full_nodes) < 3:
            continue

        polyline = _nodes_to_polyline(G, full_nodes)

        # Ensure polyline starts/ends exactly at the first graph node position;
        # app.py will separately inject the exact (lat,lng) from the request.
        score = _score_route(polyline, full_nodes, target_m)
        results.append((polyline, full_nodes, score))

    return results


# -----------------------------------------------------------------------------
# Fallback geometric loop
# -----------------------------------------------------------------------------


def _build_geometric_box(lat: float, lng: float, target_m: float) -> List[LatLng]:
    """
    Simple rectangular loop around (lat, lng) used as final fallback.
    Side length ~ target_m / 4.
    """
    # side length in meters
    a = target_m / 4.0
    R = 6371000.0
    dlat = (a / R) * (180.0 / math.pi)
    dlon = dlat / max(0.1, math.cos(math.radians(lat)))

    return [
        (lat + dlat, lng),
        (lat + dlat, lng + dlon),
        (lat - dlat, lng + dlon),
        (lat - dlat, lng),
        (lat + dlat, lng),
    ]


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[LatLng], Dict]:
    """
    Main API used by app.py.

    Returns:
        polyline: List[(lat, lng)] WITHOUT the duplicated start/end
                  injection; app.py uses the raw list.
        meta:     Diagnostic information.
    """
    t0 = time.time()
    target_m = max(300.0, km * 1000.0)
    km_requested = km

    meta: Dict = {
        "len": 0.0,
        "err": float("inf"),
        "roundness": 0.0,
        "overlap": 0.0,
        "uturn": 0.0,
        "score": -1e9,
        "success": False,
        "length_ok": False,
        "used_fallback": False,
        "valhalla_calls": 0,
        "kakao_calls": 0,
        "routes_checked": 0,
        "routes_validated": 0,
        "km_requested": km_requested,
        "target_m": target_m,
        "time_s": 0.0,
        "message": "",
    }

    # Build OSM pedestrian graph
    try:
        G = _build_pedestrian_graph(lat, lng, km)
    except Exception as e:  # pragma: no cover - defensive
        # As a last resort, geometric loop
        box = _build_geometric_box(lat, lng, target_m)
        meta.update(
            len=polyline_length_m(box),
            err=abs(polyline_length_m(box) - target_m),
            roundness=_roundness(box),
            overlap=1.0,
            uturn=_uturn_fraction(box),
            score=-1e6,
            success=False,
            length_ok=False,
            used_fallback=True,
            time_s=time.time() - t0,
            message=f"OSM graph build failed: {e}; geometric fallback used.",
        )
        return box, meta

    start_node = _nearest_node(G, lat, lng)

    # Generate candidate loops
    loops = _generate_loops(G, start_node, target_m)
    meta["routes_checked"] = len(loops)

    if not loops:
        box = _build_geometric_box(lat, lng, target_m)
        sbox = _score_route(box, [], target_m)
        meta.update(
            len=sbox.length_m,
            err=abs(sbox.length_m - target_m),
            roundness=sbox.roundness,
            overlap=sbox.overlap_ratio,
            uturn=sbox.uturn_ratio,
            score=sbox.score,
            success=False,
            length_ok=False,
            used_fallback=True,
            time_s=time.time() - t0,
            message="보행 루프를 찾지 못해 기하학적 사각형 루프를 사용했습니다.",
        )
        return box, meta

    # Choose best among candidates with a two-stage selection:
    # 1) strict: length_err ≤ 5%
    # 2) otherwise: best overall
    best_strict = None
    best_loose = None
    for polyline, nodes, score in loops:
        meta["routes_validated"] += 1

        if best_loose is None or score.score > best_loose[2].score:
            best_loose = (polyline, nodes, score)

        if score.length_err_ratio <= 0.05:
            if best_strict is None or score.score > best_strict[2].score:
                best_strict = (polyline, nodes, score)

    chosen = best_strict if best_strict is not None else best_loose
    polyline, nodes, sc = chosen  # type: ignore[assignment]

    # Inject the exact requested (lat, lng) as start/end points so that
    # the polyline visually begins and ends at the user's location.
    if polyline:
        if polyline[0] != (lat, lng):
            polyline = [(lat, lng)] + list(polyline)
        if polyline[-1] != (lat, lng):
            polyline = list(polyline) + [(lat, lng)]
        # Recompute score with the updated polyline (nodes stay the same).
        sc = _score_route(polyline, nodes, target_m)

    meta.update(
        len=sc.length_m,
        err=abs(sc.length_m - target_m),
        roundness=sc.roundness,
        overlap=sc.overlap_ratio,
        uturn=sc.uturn_ratio,
        score=sc.score,
        success=True,
        length_ok=sc.length_err_ratio <= 0.05,
        used_fallback=False,
        time_s=time.time() - t0,
        message=(
            "요청 거리의 ±5% 이내에서 가장 품질이 높은 보행 루프를 반환합니다."
            if sc.length_err_ratio <= 0.05
            else "±5% 이내 후보는 없었지만, 가장 품질이 높은 루프를 반환합니다."
        ),
    )

    return polyline, meta