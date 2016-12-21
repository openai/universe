#!/usr/bin/env python

import json
import psutil
import time

class DiagnosticsLogger(object):
    def __init__(self, interval=5):
        self.interval = interval
        self.last_cpu_times = {}  # pid -> (user, sys)

    def run(self):
        while True:
            cpu_times, chrome_reset = self.cpu_times()
            print(json.dumps({
                'time': time.time(),
                'cpu_times': cpu_times,
                'cpu_percent': psutil.cpu_percent(percpu=True),
                'chrome_reset': chrome_reset,
            }), flush=True)
            self.chrome_reset = False
            time.sleep(self.interval)

    def get_chrome_procs(self):
        def is_chrome(proc):
            try:
                return proc.name() == 'chrome'
            except psutil.ZombieProcess:
                return False
        return [p for p in psutil.process_iter() if is_chrome(p)]

    def cpu_times(self):
        ''' return {pid: {'user': 0.0, 'sys': 0.0}}, chrome_reset '''
        chrome_procs = self.get_chrome_procs()
        new_pids = {p.pid for p in chrome_procs}
        old_pids = {pid for pid in self.last_cpu_times}
        try:
            cpu_times = {p.pid: p.cpu_times() for p in chrome_procs}
        except psutil.NoSuchProcess:
            # Chrome restarted since fetching the new pids above. Better luck next time.
            return {}, True
        if new_pids != old_pids:
            # We don't know when the Chrome procs were restarted, so don't
            # return elapsed time until next run.
            self.last_cpu_times = cpu_times
            return {}, True
        # Same chrome pids as last run: measure the elapsed cpu times
        ordered_old_times = (self.last_cpu_times[p.pid] for p in chrome_procs)
        ordered_new_times = (cpu_times[p.pid] for p in chrome_procs)
        cpu_times_diff = {p.pid: {'user': (t[0] - l[0]) / self.interval, 'sys': (t[1] - l[1]) / self.interval}
                for (p, t, l) in zip(chrome_procs, ordered_new_times, ordered_old_times)}
        self.last_cpu_times = cpu_times
        return cpu_times_diff, False

if __name__ == '__main__':
    DiagnosticsLogger().run()

