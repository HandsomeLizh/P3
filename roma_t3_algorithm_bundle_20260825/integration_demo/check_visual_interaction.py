"""Exercise only our owned map window; no sensor replay or vehicle commands."""
import argparse
import ctypes as c
import json
import math
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    output = Path(state['output'])
    if not output.resolve().is_relative_to(Path('/home/yanfa/P3')):
        raise ValueError('Only test P3-owned visual state')
    pid = state['services']['rviz']['pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'/integration_demo/rviz_window/build/t3_visual_window' not in command:
        raise ValueError('Unexpected viewer process')
    window = None
    for value in re.findall('0x[0-9a-f]+', subprocess.check_output(['xprop', '-root', '_NET_CLIENT_LIST'], text=True)):
        owner = subprocess.check_output(['xprop', '-id', value, '_NET_WM_PID'], text=True)
        if re.search(r'=\s*'+str(pid)+r'\s*$', owner):
            window = value
            break
    if window is None:
        raise ValueError('No owned viewer window found')
    info = subprocess.check_output(['xwininfo', '-id', window], text=True)
    ox = int(re.search(r'Absolute upper-left X:\s*(-?\d+)', info)[1])
    oy = int(re.search(r'Absolute upper-left Y:\s*(-?\d+)', info)[1])
    x = c.CDLL('libX11.so.6')
    xt = c.CDLL('libXtst.so.6')
    x.XOpenDisplay.argtypes, x.XOpenDisplay.restype = [c.c_char_p], c.c_void_p
    x.XDefaultRootWindow.argtypes, x.XDefaultRootWindow.restype = [c.c_void_p], c.c_ulong
    x.XQueryPointer.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_ulong), c.POINTER(c.c_ulong),
                                c.POINTER(c.c_int), c.POINTER(c.c_int), c.POINTER(c.c_int),
                                c.POINTER(c.c_int), c.POINTER(c.c_uint)]
    x.XFlush.argtypes = x.XCloseDisplay.argtypes = [c.c_void_p]
    xt.XTestFakeMotionEvent.argtypes = [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_ulong]
    xt.XTestFakeButtonEvent.argtypes = [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]
    display = x.XOpenDisplay(None)
    if not display:
        raise RuntimeError('No X11 display')
    root, child, px, py, wx, wy, mask = c.c_ulong(), c.c_ulong(), c.c_int(), c.c_int(), c.c_int(), c.c_int(), c.c_uint()
    x.XQueryPointer(display, x.XDefaultRootWindow(display), c.byref(root), c.byref(child),
                    c.byref(px), c.byref(py), c.byref(wx), c.byref(wy), c.byref(mask))
    if mask.value & (256 | 512 | 1024):
        x.XCloseDisplay(display)
        raise RuntimeError('User is holding a mouse button; test skipped')
    def motion(xx, yy):
        xt.XTestFakeMotionEvent(display, -1, xx, yy, 0)
        x.XFlush(display)
    def button(pressed):
        xt.XTestFakeButtonEvent(display, 1, int(pressed), 0)
        x.XFlush(display)
    runtime = output/'visual_runtime'
    heartbeat = runtime/'render_status.json'
    records = []
    try:
        motion(ox+440, oy+440)
        start = time.monotonic()
        before = json.loads(heartbeat.read_text())['frames']
        button(True)
        for i in range(120):
            motion(ox+440+int(90*math.sin(i/20)), oy+440+int(25*math.sin(i/12)))
            time.sleep(.05)
            status = json.loads(heartbeat.read_text())
            records.append({'wall_elapsed_sec': time.monotonic()-start, **status,
                            'heartbeat_age_sec': time.time()-heartbeat.stat().st_mtime})
        button(False)
        time.sleep(.6)
        elapsed = time.monotonic()-start
        latest = json.loads(heartbeat.read_text())
        # Restore the full-map view after testing rotation.
        motion(ox+45, oy+18)
        button(True)
        button(False)
    finally:
        button(False)
        motion(px.value, py.value)
        x.XCloseDisplay(display)
    monitor = json.loads((runtime/'monitor_window.json').read_text())
    tree = subprocess.check_output(['xwininfo', '-id', hex(monitor['window'])], text=True)
    parent_tree = subprocess.check_output(['xwininfo', '-id', window, '-tree'], text=True)
    attached = hex(monitor['window']) in parent_tree
    result = {'window': window, 'viewer_pid': pid, 'duration_sec': elapsed,
              'frame_updates': latest['frames']-before, 'update_hz': (latest['frames']-before)/elapsed,
              'maximum_heartbeat_age_sec': max(row['heartbeat_age_sec'] for row in records),
              'last_recent_update_gap_ms': latest.get('maximum_recent_update_gap_ms'),
              'right_panel_still_attached': attached,
              'runs_bag': False, 'publishes_vehicle_commands': False, 'samples': records}
    result['passed'] = result['update_hz'] >= 25 and result['maximum_heartbeat_age_sec'] < 1.5 and attached
    report = output/('interaction_check_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}, indent=2))
    print('Report:', report)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
