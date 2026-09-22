from psutil import Process


def get_current_memory(pid=None):
    return Process(pid).memory_full_info()


class MemoryMonitorContext(object):
    def __init__(self, pid=None):
        self._pid = pid

        self.reset()

    def __enter__(self):
        self._enter_memory = self._get_used_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_memory = self._get_used_memory()
        return self

    def _get_used_memory(self) -> float:
        mem = get_current_memory(self._pid)
        return mem.uss

    def reset(self):
        self._enter_memory = -1
        self._exit_memory = -1

    def delta(self) -> float:
        if self._enter_memory < 0:
            raise RuntimeError("Please using context manager to wrap your code.")

        if self._exit_memory < 0:
            return self._get_used_memory() - self._enter_memory

        return self._exit_memory - self._enter_memory
