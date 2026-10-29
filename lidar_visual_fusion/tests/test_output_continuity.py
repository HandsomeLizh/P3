from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/t3_lidar_visual_fusion'))
from t3_lidar_visual_fusion.output_continuity import OutputContinuity
from t3_lidar_visual_fusion.continuous_frame import ContinuousFrame

def pose(x):
    p=np.eye(4);p[0,3]=x;return p

g=OutputContinuity();g.accept(1.,pose(0.))
assert g.check(1.03,pose(.006))=='qualified'
g.accept(1.03,pose(.006))
assert g.check(10.499,pose(21.6648))=='output_discontinuity'
assert g.check(1000.,pose(21.6648))=='output_discontinuity'
assert g.check(2.33,pose(.266))=='qualified'
g.accept(2.33,pose(.266))
assert g.check(2.33,pose(.466))=='qualified'
g.accept(2.33,pose(.466))
assert g.check(2.33,pose(.866))=='same_stamp_discontinuity'
assert g.check(1.0,pose(0.))=='out_of_order'
def lift(x,z):
    p=pose(x);p[2,3]=z;return p

# Measured wheel-speed bounds tighten the static limits but never loosen them.
w=OutputContinuity(max_speed=1.,max_step=6.,correction_translation=.25,correction_vertical=.05)
w.accept(1.,pose(0.))
assert w.check(2.,pose(.9))=='qualified'                                   # static limit only
assert w.check(2.,pose(.9),speed_limit=.15)=='output_discontinuity'        # wheels say <=0.15 m/s
assert w.check(2.,pose(.3),speed_limit=.15)=='qualified'
assert w.check(2.,pose(.9),speed_limit=5.)=='qualified'                    # cannot loosen: min() with static
assert w.check(2.,lift(.05,.14),speed_limit=.15,vertical_rate_limit=.08)=='vertical_discontinuity'
assert w.check(2.,lift(.05,.11),speed_limit=.15,vertical_rate_limit=.08)=='qualified'
assert w.check(2.,lift(.05,.14))=='qualified'                              # no wheel data: static only
try:w.check(2.,pose(.1),speed_limit=-1.);raise AssertionError('negative limit accepted')
except ValueError:pass
# The observed switch jumps: ~0.14-1.1 m of z in ~1 s while wheels report ~0.02-0.11 m/s.
for dz,wheel in ((.143,.054),(.231,.162),(.993,.174)):
    assert w.check(2.,lift(.05,dz),speed_limit=wheel,
        vertical_rate_limit=wheel*.574+.02) in ('vertical_discontinuity','output_discontinuity')
source=ContinuousFrame(recovery_frames=1,recovery_max_step=6.,recovery_max_angle=1.)
source.begin_epoch('odom');source.alignment=np.eye(4)
assert source.accept(1.,pose(0.)).transform is not None
source.close('failed_registration')
assert source.accept(20.,pose(21.)).reason=='recovery_discontinuity'
assert source.accept(21.,pose(.5)).transform is not None
print('PASS: long-gap output/source jump rejected; same-stamp cumulative bound; nearby recovery; wheel-bounded speed and vertical gates')
