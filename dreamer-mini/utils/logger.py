import csv
import os


class CSVLogger:
    """
    最小版 CSV 日志器。
    把每一步的 metrics 记录到 csv，方便后续画图。
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.initialized = False
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, metrics: dict):
        """
        追加一行 metrics 到 csv。
        """
        write_header = not self.initialized and not os.path.exists(self.filepath)

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))

            if write_header:
                writer.writeheader()

            writer.writerow(metrics)

        self.initialized = True