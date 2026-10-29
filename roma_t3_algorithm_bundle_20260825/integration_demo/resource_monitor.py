"""Small read-only /proc sampler; all recorded output stays in the demo folder."""
import json
from pathlib import Path
import shutil
import threading
import time


def snapshot(state):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[name] = int(value.split()[0]) * 1024
    groups = {item['pid']: name for name, item in state['services'].items()}
    processes = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            stat = (item / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            group = int(fields[2])
            if group not in groups or fields[0] == 'Z':
                continue
            status = (item / 'status').read_text().splitlines()
            rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith('VmRSS:')), 0)
            processes.append({'pid': int(item.name), 'service': groups[group], 'rss_bytes': rss})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return {'time': time.time(), 'system_memory_bytes': memory, 'processes': processes,
            'free_disk_bytes': shutil.disk_usage(state['output']).free}


class ResourceMonitor:
    def __init__(self, state):
        self.state = state
        self.path = Path(state['output']) / 'resources.jsonl'
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.samples = 0
        self.minimum_available = None
        self.maximum_rss = {}
        self.resource_error = None

    def start(self):
        self.thread.start()

    def run(self):
        try:
            with self.path.open('a', buffering=1) as stream:
                while not self.stop_event.is_set():
                    item = snapshot(self.state)
                    stream.write(json.dumps(item) + '\n')
                    self.samples += 1
                    available = item['system_memory_bytes']['MemAvailable']
                    self.minimum_available = min(available, self.minimum_available or available)
                    for process in item['processes']:
                        service = process['service']
                        self.maximum_rss[service] = max(self.maximum_rss.get(service, 0), process['rss_bytes'])
                    if available < 8 * 1024**3 or item['free_disk_bytes'] < 10 * 1024**3:
                        self.resource_error = 'Available memory < 8 GiB or disk < 10 GiB'
                    self.stop_event.wait(2)
        except Exception as error:
            self.resource_error = f'Resource sampling failed: {error}'

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        return {'samples': self.samples, 'minimum_available_bytes': self.minimum_available,
                'maximum_single_process_rss_bytes_by_service': self.maximum_rss,
                'resource_error': self.resource_error,
                'note': '2-second samples, not exact transient peaks; RSS includes shared pages'}
