"""Bounded open-loop UE routes; these are NOT obstacle-avoiding planners."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Segment:
    name: str
    seconds: float
    speed: float
    curvature: float


def route(profile):
    if profile == "short":
        segments = [Segment(name, seconds, .05, curvature) for name, seconds, curvature in (
            ("straight_1", 30, 0), ("left_1", 45, .8), ("straight_2", 20, 0),
            ("right_1", 45, -.8), ("straight_3", 30, 0),
            ("right_2", 20, -.8), ("left_2", 20, .8))]
    elif profile == "long":
        # Alternate left/right stadium-shaped laps, ideally returning to the
        # same start pose after each lap. Drift and obstacles still require
        # monitoring; the runtime also enforces a 15 m reference-position bound.
        segments, remaining, lap = [], 1800., 0
        while remaining > 1e-6:
            sign = 1 if lap % 2 == 0 else -1
            for index, (seconds, curvature) in enumerate(((40, 0), (math.pi / .04, sign * .8),
                                                         (40, 0), (math.pi / .04, sign * .8))):
                duration = min(seconds, remaining)
                if duration <= 1e-6:
                    break
                segments.append(Segment(f"lap_{lap+1}_leg_{index+1}", duration, .05, curvature))
                remaining -= duration
            lap += 1
    else:
        raise ValueError("profile must be short or long")
    assert all(0 < item.seconds <= 90 and abs(item.speed) <= .1 and abs(item.curvature) <= 2
               for item in segments)
    return segments


def describe(profile):
    segments = route(profile)
    return {"profile": profile, "duration_seconds": sum(s.seconds for s in segments),
            "nominal_distance_m": sum(abs(s.speed) * s.seconds for s in segments),
            "control": "timed_open_loop_no_obstacle_avoidance",
            "segments": [asdict(s) for s in segments]}
