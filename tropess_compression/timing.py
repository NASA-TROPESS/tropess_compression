from time import process_time
from timeit import default_timer
import logging

class RuntimeLogging(object):

    def __init__(self, description, logger=logging.getLogger(), log_level=logging.DEBUG):

        self.description = description
        self.logger = logger
        self.log_level = log_level

        self.cpu_start = 0
        self.wall_start = 0

        self.cpu_end = 0
        self.wall_end = 0

    def runtime_string(self, seconds):
        s = seconds

        h = int(s // 3600)
        s = s - (h * 3600)

        m = int(s // 60)
        s = s - (m * 60)

        return f"{h:02d}:{m:02d}:{s:05.2f}"

    def start(self):
        self.cpu_start = self.cpu_end = process_time()
        self.wall_start = self.wall_end = default_timer()

        return self.cpu_start, self.wall_start

    def stop(self):
        self.cpu_end = process_time()
        self.wall_end = default_timer()

        cpu_elapsed = self.cpu_end - self.cpu_start
        wall_elapsed = self.wall_end - self.wall_start

        cpu_runtime = self.runtime_string(cpu_elapsed)
        wall_runtime = self.runtime_string(wall_elapsed)

        self.logger.log(self.log_level, f"{self.description} - sys/user cpu time: {cpu_runtime} ({cpu_elapsed:02.3f} seconds)")
        self.logger.log(self.log_level, f"{self.description} -   wall clock time: {wall_runtime} ({wall_elapsed:02.3f} seconds)")

    def __enter__(self):
        return self.start()

    def __exit__(self, type, value, traceback):
        self.stop()
