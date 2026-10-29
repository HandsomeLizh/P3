"""Bound output discontinuities, including repeated corrections of one stamp."""
import numpy as np
from .core import motion


class OutputContinuity:
    def __init__(self, max_speed=4., max_angular_speed=2., max_step=6., max_angle=1.,
                 correction_translation=.5, correction_rotation=.15, correction_vertical=None):
        if correction_vertical is None:correction_vertical=.4*correction_translation
        values=[max_speed,max_angular_speed,max_step,max_angle,correction_translation,
                correction_rotation,correction_vertical]
        if any(not np.isfinite(v) or v<=0 for v in values):
            raise ValueError("Output continuity limits must be finite and positive")
        self.max_speed,self.max_angular_speed=max_speed,max_angular_speed
        self.max_step,self.max_angle=max_step,max_angle
        self.correction_translation,self.correction_rotation=correction_translation,correction_rotation
        self.correction_vertical=correction_vertical
        self.latest=None
        self.stamp_origin=None

    def check(self, stamp, transform, speed_limit=None, vertical_rate_limit=None):
        """Optional measured bounds (e.g. wheel speed) only ever tighten the static limits."""
        for limit in (speed_limit,vertical_rate_limit):
            if limit is not None and (not np.isfinite(limit) or limit<0):
                raise ValueError("Measured output speed limits must be finite and nonnegative")
        if self.latest is None:return "qualified"
        old_stamp,old=self.latest
        dt=stamp-old_stamp
        if dt<0:return "out_of_order"
        if dt==0:
            # Compare to the first accepted result of this timestamp so a
            # sequence of small corrections cannot bypass the total bound.
            _,distance,angle=motion(self.stamp_origin,transform)
            if distance>self.correction_translation or angle>self.correction_rotation:
                return "same_stamp_discontinuity"
        else:
            max_speed=self.max_speed if speed_limit is None else min(self.max_speed,speed_limit)
            _,distance,angle=motion(old,transform)
            if (distance>min(self.max_step,max_speed*dt+self.correction_translation) or
                angle>min(self.max_angle,self.max_angular_speed*dt+self.correction_rotation)):
                return "output_discontinuity"
            if (vertical_rate_limit is not None and abs(transform[2,3]-old[2,3])>
                    min(self.max_step,vertical_rate_limit*dt+self.correction_vertical)):
                return "vertical_discontinuity"
        return "qualified"

    def accept(self, stamp, transform):
        if self.latest is None or stamp>self.latest[0]:self.stamp_origin=transform.copy()
        self.latest=(stamp,transform.copy())
