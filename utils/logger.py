import csv, os
from typing import Dict, List
from collections import defaultdict


class Logger:
    def __init__(self, log_dir: str = "logs", print_every: int = 100):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "training.csv")
        self.print_every = print_every
        self._ep = 0
        self._file = None
        self._writer = None
        self.history: Dict[str, List] = defaultdict(list)

    def log(self, metrics: Dict[str, float]):
        self._ep += 1
        for k, v in metrics.items():
            self.history[k].append(v)

        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=["episode"] + list(metrics.keys())
            )
            self._writer.writeheader()

        self._writer.writerow({"episode": self._ep, **metrics})
        self._file.flush()

        if self._ep % self.print_every == 0:
            parts = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"[Ep {self._ep:5d}] {parts}")

    def close(self):
        if self._file:
            self._file.close()