import time


class AverageMeter:
    """
    用于统计某个标量的平均值。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class SimpleLogger:
    """
    简单日志工具。
    """

    def __init__(self):
        self.start_time = time.time()

    def log(self, message: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:8.1f}s] {message}")