#!/usr/bin/env python

import json
import psutil
import time

class DiagnosticsLogger(object):
    def __init__(self, interval=5):
        self.interval = interval
        self.chrome_procs = None

    def run(self):
        while not self.chrome_procs:
            time.sleep(0.2)
            self.refresh_chrome_procs()
        while True:
            print(json.dumps({
                'time': time.time(),
                'cpu_times': self.cpu_times(),
                'cpu_percent': psutil.cpu_percent(percpu=True),
                'chrome_reset': self.chrome_reset,
            }), flush=True)
            self.chrome_reset = False
            time.sleep(self.interval)

    def refresh_chrome_procs(self):
        def is_chrome(proc):
            try:
                return proc.name() == 'chrome'
            except psutil.ZombieProcess:
                return False
        self.chrome_procs = [p for p in psutil.process_iter() if is_chrome(p)]
        self.last_cpu_times = [p.cpu_times() for p in self.chrome_procs]
        self.chrome_reset = True

    def cpu_times(self):
        try:
            cpu_times = [p.cpu_times() for p in self.chrome_procs]
        except psutil.NoSuchProcess:
            # Chrome died and restarted, get new pids and try again
            self.refresh_chrome_procs()
            cpu_times = [p.cpu_times() for p in self.chrome_procs]
        cpu_times_diff = {p.pid: {'user': (t[0] - l[0]) / self.interval, 'sys': (t[1] - l[1]) / self.interval}
                for (p, t, l) in zip(self.chrome_procs, cpu_times, self.last_cpu_times)}
        self.last_cpu_times = cpu_times
        return cpu_times_diff

if __name__ == '__main__':
    DiagnosticsLogger().run()

