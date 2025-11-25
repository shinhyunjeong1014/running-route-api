# turn_algo.py
import math
from route_algo import haversine_m, _cumulative_distances

###############################################
#  Polyline Simplification
###############################################
def simplify_polyline(polyline, min_dist=20.0):
    if not polyline:
        return polyline
    simp = [polyline[0]]
    last = polyline[0]
    for p in polyline[1:]:
        d = haversine_m(last["lat"], last["lng"], p["lat"], p["lng"])
        if d >= min_dist:
            simp.append(p)
            last = p
    if simp[-1] != polyline[-1]:
        simp.append(polyline[-1])
    return simp


###############################################
# Angles / Bearings
###############################################
def bearing(a, b):
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lng"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lng"])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1)*math.cos(lat2) - math.sin(lat1)*math.sin(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def signed_angle(a, b, c):
    th1 = bearing(a, b)
    th2 = bearing(b, c)
    return ((th2 - th1 + 540) % 360) - 180


###############################################
# Text formatting for instruction
###############################################
def format_instruction(distance_m, turn_type):
    d = int(round(distance_m / 10.0) * 10)
    if turn_type == "left":
        return f"{d}m 앞 좌회전"
    if turn_type == "right":
        return f"{d}m 앞 우회전"
    if turn_type == "straight":
        return f"{d}m 직진"
    if turn_type == "uturn":
        return f"{d}m 앞 U턴"
    if turn_type == "arrive":
        return "목적지에 도착했습니다"
    return ""


###############################################
# Main turn detection
###############################################
def polyline_turns_new(polyline):
    if not polyline or len(polyline) < 3:
        return []
    
    simp = simplify_polyline(polyline, min_dist=20.0)
    cum = _cumulative_distances(simp)

    turns = []
    last_turn_idx = 0

    for i in range(1, len(simp) - 1):
        a, b, c = simp[i-1], simp[i], simp[i+1]
        ang = signed_angle(a, b, c)
        ang_abs = abs(ang)

        if ang_abs < 5:
            continue

        if ang_abs >= 150:
            t_type = "uturn"
        elif ang_abs >= 25:
            t_type = "left" if ang > 0 else "right"
        else:
            t_type = "straight"

        dist_to_turn = cum[i] - cum[last_turn_idx]
        if dist_to_turn < 35:
            continue

        turns.append({
            "lat": b["lat"],
            "lng": b["lng"],
            "type": t_type,
            "distance": round(dist_to_turn, 1),
            "instruction": format_instruction(dist_to_turn, t_type),
        })
        last_turn_idx = i

    final_dist = cum[-1] - cum[last_turn_idx]
    turns.append({
        "lat": simp[-1]["lat"],
        "lng": simp[-1]["lng"],
        "type": "arrive",
        "distance": round(final_dist, 1),
        "instruction": format_instruction(final_dist, "arrive"),
    })

    return turns


def build_turn_by_turn(polyline, km_requested: float, total_length_m=None):
    if total_length_m is None:
        total_length_m = _cumulative_distances(polyline)[-1]

    turns = polyline_turns_new(polyline)

    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round((total_length_m/1000)*8.0, 1),
        "turn_count": len([t for t in turns if t["type"] != "arrive"])
    }

    return turns, summary
