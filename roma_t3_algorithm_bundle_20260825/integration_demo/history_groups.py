"""Group legacy Env_X ROS samples without trusting the broken header clock."""
INPUTS = ("/Car/T5/Cam_Left/image_raw/color", "/Car/T5/Cam_Right/image_raw/color", "/Car/T5/OS1/points")


class BatchGroups:
    def __init__(self, max_span_ns=2_000_000_000):
        self.pending = {}
        self.max_span_ns = max_span_ns
        self.dropped = 0

    def add(self, topic, recorded_ns, data):
        if topic not in INPUTS:
            return None
        if self.pending and (topic in self.pending or
                             recorded_ns - min(value[0] for value in self.pending.values()) > self.max_span_ns):
            self.dropped += len(self.pending)
            self.pending.clear()
        self.pending[topic] = (recorded_ns, data)
        if len(self.pending) != 3:
            return None
        result, self.pending = self.pending, {}
        return result
