"""
utils/episode_logger.py

EpisodeLogger — ghi log chi tiết từng episode ra file CSV.

Mỗi hàng CSV là 1 episode với đầy đủ thông tin:
  Cột chung:
    algo           : tên thuật toán ("DQN", "OSPF", ...)
    episode        : số thứ tự episode
    src, dst       : cặp node nguồn/đích
    path           : đường đi (dạng "0→1→3→7")
    hops           : số hop
    total_delay_ms : tổng delay (ms)
    dropped        : 1 nếu bị drop, 0 nếu không
    reward         : reward tổng episode (để trống nếu không có — demo/OSPF)
    loss           : average loss episode (ep_loss / avg_steps).
                        None → để trống (OSPF/demo/eval).
    avg_utilization : utilization băng thông trung bình trên path
    avg_queue_util  : queue_util trung bình trên path
    total_dropped_data: tổng data bị drop trên cả đường đi

  Cột per-link (lặp lại cho mỗi link, indexed từ 0 đến max_hops-1):
    link{i}_nodes          : "u→v"
    link{i}_delay          : delay vật lý (ms)
    link{i}_bandwidth      : băng thông (Mbps)
    link{i}_load           : load hiện tại (Mbps)
    link{i}_utilization    : load / bandwidth
    link{i}_queue_size_cur : kích thước hàng đợi (packets)
    link{i}_queue_used_cur : hàng đợi đang dùng (packets)
    link{i}_queue_util     : queue_used_cur / queue_size_cur
    link{i}_dropped_data   : data bị drop tích lũy trên link (packets)

Ghi chú:
  - Header được tạo lần đầu tiên log_episode() được gọi.
  - Số cột per-link cố định = max_hops để header nhất quán.
  - Cột "algo" phân biệt DQN / OSPF / ... trong cùng 1 file (demo).
"""

import csv
import os
from typing import Any, Dict, List, Optional


class EpisodeLogger:

    LINK_FIELDS = [
        "nodes",
        "delay",
        "bandwidth",
        "load",
        "utilization",
        "queue_size_cur",
        "queue_used_cur",
        "queue_util",
        "dropped_data",   # lượng data bị drop tích lũy trên link (packets)
    ]

    def __init__(self, log_dir: str, filename: str, max_hops: int = 8):
        """
        Args:
            log_dir  : thư mục lưu file CSV.
            filename : tên file (vd. "train_episodes.csv").
            max_hops : số link tối đa mỗi path — xác định số cột per-link.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.filepath  = os.path.join(log_dir, filename)
        self.max_hops  = max_hops
        self._file     = None
        self._writer   = None
        self._fieldnames: List[str] = []

    # ------------------------------------------------------------------ #

    def _build_fieldnames(self) -> List[str]:
        base = [
            "algo",
            "episode",
            "src",
            "dst",
            "path",
            "hops",
            "total_delay_ms",
            "dropped",
            "reward",
            "loss",
            "avg_utilization",
            "avg_queue_util",
            "total_dropped_data",
        ]
        for i in range(self.max_hops):
            for f in self.LINK_FIELDS:
                base.append(f"link{i}_{f}")
        return base

    def _init_writer(self):
        self._fieldnames = self._build_fieldnames()
        self._file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            extrasaction="ignore",
        )
        self._writer.writeheader()

    # ------------------------------------------------------------------ #

    def log_episode(
        self,
        algo:    str,
        episode: int,
        info:    Dict[str, Any],
        reward:  Optional[float] = None,
        loss:    Optional[float] = None,
    ):
        """
        Ghi một episode vào file CSV.

        Args:
            algo    : tên thuật toán — "DQN", "OSPF", ...
                      Cột này xuất hiện đầu tiên trong mỗi hàng.
            episode : số thứ tự episode (bắt đầu từ 1).
            info    : dict info từ env._info() hoặc tự tạo. Cần có:
                        src, dst, path, hops, total_delay, dropped,
                        avg_utilization, avg_queue_util, link_details.
            reward  : tổng reward episode. None → để trống (OSPF/demo).
            loss    : average loss episode (ep_loss / avg_steps).
                      None → để trống (OSPF/demo/eval).
        """
        if self._writer is None:
            self._init_writer()

        link_details: List[Dict] = info.get("link_details", [])
        path     = info.get("path", [])
        path_str = "→".join(map(str, path))

        row: Dict[str, Any] = {
            "algo":            algo,
            "episode":         episode,
            "src":             info.get("src", ""),
            "dst":             info.get("dst", ""),
            "path":            path_str,
            "hops":            info.get("hops", max(0, len(path) - 1)),
            "total_delay_ms":  f"{info.get('total_delay', 0.0):.4f}",
            "dropped":         int(info.get("dropped", False)),
            "reward":          f"{reward:.4f}" if reward is not None else "",
            "loss":            f"{loss:.6f}"   if loss   is not None else "",
            "avg_utilization":     f"{info.get('avg_utilization', 0.0):.4f}",
            "avg_queue_util":      f"{info.get('avg_queue_util', 0.0):.4f}",
            "total_dropped_data":  f"{info.get('total_dropped_data', 0.0):.4f}",
        }

        # Per-link columns (padded đến max_hops với chuỗi rỗng)
        for i in range(self.max_hops):
            if i < len(link_details):
                lk = link_details[i]
                row[f"link{i}_nodes"]          = f"{lk['src_node']}→{lk['dst_node']}"
                row[f"link{i}_delay"]          = f"{lk['delay']:.4f}"
                row[f"link{i}_bandwidth"]      = f"{lk['bandwidth']:.4f}"
                row[f"link{i}_load"]           = f"{lk['load']:.4f}"
                row[f"link{i}_utilization"]    = f"{lk['utilization']:.4f}"
                row[f"link{i}_queue_size_cur"] = f"{lk['queue_size_cur']:.0f}"
                row[f"link{i}_queue_used_cur"] = f"{lk['queue_used_cur']:.4f}"
                row[f"link{i}_queue_util"]     = f"{lk['queue_util']:.4f}"
                row[f"link{i}_dropped_data"]  = f"{lk.get('dropped_data', 0.0):.4f}"
            else:
                for f in self.LINK_FIELDS:
                    row[f"link{i}_{f}"] = ""

        self._writer.writerow(row)
        self._file.flush()

    # ------------------------------------------------------------------ #

    def close(self):
        if self._file:
            self._file.close()
            self._file = None