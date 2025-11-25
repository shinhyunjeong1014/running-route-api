# turn_algo.py
import math
import logging
from route_algo import haversine_m, _cumulative_distances

# 회전 감지 임계값
ANGLE_UTURN = 150
ANGLE_TURN = 30  # 25→30도로 조정 (Fallback 경로에서 더 자연스러운 감지)
MIN_DIST_TURN = 30  # 35→30m로 조정
MIN_DIST_SIMPLIFY = 12.0  # 15→12m로 조정 (더 세밀한 경로 표현)
STRAIGHT_INTERVAL = 250  # 300→250m로 조정

###############################################
# Polyline Simplification
###############################################
def simplify_polyline(polyline):
    """최소 거리 임계값을 이용한 경로 간소화"""
    if not polyline or len(polyline) < 3:
        return polyline
    
    simp = [polyline[0]]
    last = polyline[0]
    
    for p in polyline[1:]:
        d = haversine_m(last["lat"], last["lng"], p["lat"], p["lng"])
        if d >= MIN_DIST_SIMPLIFY:
            simp.append(p)
            last = p
    
    if simp[-1] != polyline[-1]:
        simp.append(polyline[-1])
    
    logging.debug(f"경로 간소화: {len(polyline)}개 → {len(simp)}개")
    return simp


###############################################
# Angles / Bearings
###############################################
def bearing(a, b):
    """두 지점 사이의 방위각 계산 (0~360도)"""
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lng"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lng"])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.cos(lat2) - math.sin(lat1) * math.sin(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def signed_angle(a, b, c):
    """b 지점에서의 회전각 계산 (-180~+180도)"""
    th1 = bearing(a, b)
    th2 = bearing(b, c)
    return ((th2 - th1 + 540) % 360) - 180


###############################################
# Instruction Formatting
###############################################
def format_instruction(distance_m, turn_type):
    """안내 문구 생성"""
    d = int(round(distance_m / 10.0) * 10)
    
    instructions = {
        "left": f"{d}m 앞에서 좌회전하세요",
        "right": f"{d}m 앞에서 우회전하세요",
        "straight": f"{d}m 직진하세요",
        "uturn": f"{d}m 앞에서 U턴하세요",
        "arrive": "목적지에 도착했습니다. 러닝을 완료했습니다.",
        "start": "러닝을 시작합니다. 직진하세요."
    }
    
    return instructions.get(turn_type, "")


###############################################
# Turn Detection
###############################################
def polyline_turns(polyline):
    """Turn-by-turn 안내 생성"""
    if not polyline or len(polyline) < 3:
        return []
    
    simp = simplify_polyline(polyline)
    cum = _cumulative_distances(simp)
    turns = []
    last_turn_dist = 0.0
    
    # 시작 안내
    turns.append({
        "lat": simp[0]["lat"],
        "lng": simp[0]["lng"],
        "type": "start",
        "distance": 0.0,
        "instruction": format_instruction(0, "start"),
    })
    
    # 회전 감지
    for i in range(1, len(simp) - 1):
        a, b, c = simp[i-1], simp[i], simp[i+1]
        ang = signed_angle(a, b, c)
        ang_abs = abs(ang)
        dist_to_turn = cum[i] - last_turn_dist
        t_type = None
        
        # 회전 유형 판별
        if ang_abs >= ANGLE_UTURN:
            t_type = "uturn"
        elif ang_abs >= ANGLE_TURN:
            t_type = "left" if ang > 0 else "right"
        elif dist_to_turn >= STRAIGHT_INTERVAL:
            t_type = "straight"
        
        # 유효한 안내 추가
        if t_type and dist_to_turn >= MIN_DIST_TURN:
            turns.append({
                "lat": b["lat"],
                "lng": b["lng"],
                "type": t_type,
                "angle": round(ang, 1),
                "distance": round(dist_to_turn, 1),
                "instruction": format_instruction(dist_to_turn, t_type),
            })
            last_turn_dist = cum[i]
    
    # 도착 안내
    final_dist = cum[-1] - last_turn_dist
    turns.append({
        "lat": simp[-1]["lat"],
        "lng": simp[-1]["lng"],
        "type": "arrive",
        "distance": round(final_dist, 1),
        "instruction": format_instruction(final_dist, "arrive"),
    })
    
    return turns


###############################################
# Main Function
###############################################
def build_turn_by_turn(polyline, km_requested: float, total_length_m: float):
    """Turn-by-turn 안내 및 요약 정보 생성"""
    turns = polyline_turns(polyline)
    
    # 러닝 속도: 8km/h (8분/km 페이스)
    RUNNING_SPEED_KMH = 8.0
    estimated_time_min = (total_length_m / 1000) / (RUNNING_SPEED_KMH / 60)
    
    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round(estimated_time_min, 1),
        "turn_count": len([t for t in turns if t["type"] in ["left", "right", "uturn", "straight"]])
    }
    
    return turns, summary
