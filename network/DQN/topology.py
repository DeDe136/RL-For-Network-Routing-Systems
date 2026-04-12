"""
network/DQN/topology.py

Topology mạng 8 node với MDP traffic model theo spec.

Mỗi link có 6 thông số (LinkAttr):
  delay, bandwidth, queue_size_cur, queue_used_cur, queue_used, load

link_state_vector() trả về (num_edges, 5):
  [delay_norm, bw_norm, queue_size_cur_norm, utilization, queue_util]

═══════════════════════════════════════════════════════════════════
  send_traffic(path, volume) — MDP traffic model
═══════════════════════════════════════════════════════════════════

Tại mỗi bước (step), agent chọn next_hop và env gọi send_traffic
CHỈ cho đoạn link hiện tại [prev_node → cur_node].

Logic send_traffic cho 1 link (u→v) với volume_mbps:

  1. Tìm queue_used của link trước (prev→u) trong path.
     Nếu không có link trước → queue_used_prev = 0.

  2. Cộng dồn vào queue_used_cur của link (u→v):
       queue_used_cur += queue_used_prev + volume  (chỉ bước đầu)
       queue_used_cur += queue_used_prev           (các bước sau — volume đã đi vào bước trước)
     Sau đó gán queue_used của link trước về 0.

  3. Drop check:
       if queue_used_cur > queue_size_cur → dropped = True

  4. Cập nhật load và queue_used:
       remain_bw = bandwidth - load
       if remain_bw <= queue_used_cur:
           load += remain_bw
           queue_used  = load
           queue_used_cur -= remain_bw
       else:
           load += queue_used_cur
           queue_used  = load
           queue_used_cur = 0

  ➜ Ý nghĩa: remain_bw là phần băng thông trống còn lại. Nếu queue
    lớn hơn phần trống thì chỉ truyền được phần trống, phần còn lại
    nằm lại trong queue_used_cur chờ bước sau. Nếu ngược lại, truyền
    hết queue và load tăng đúng bằng lượng đã truyền.

═══════════════════════════════════════════════════════════════════
  reduce_load(path) — leaky bucket trên path
═══════════════════════════════════════════════════════════════════

Sau mỗi step, reduce_load xử lý tuần tự từng link trong path (trái→phải):

  Với mỗi link (u→v) trong path:
  A) Nếu load(u→v) != 0 và tồn tại link tiếp theo (v→w):
       queue_used_cur(v→w) += load(u→v)
       load(u→v) = 0
     Nếu không có link tiếp theo:
       load(u→v) = 0

  B) Nếu queue_used_cur(u→v) == 0:
       load(u→v) = 0
     Nếu queue_used_cur(u→v) != 0:
       load(u→v) = 0
       remain_bw = bandwidth(u→v) - load(u→v)  (= bandwidth vì load=0)
       if remain_bw <= queue_used_cur(u→v):
           load += remain_bw
           queue_used_cur -= remain_bw
       else:
           load += queue_used_cur
           queue_used_cur = 0

  Các link ngoài path: decay load theo cấp số nhân (leaky bucket).
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional

from network.DQN.link import LinkAttr


class NetworkTopology:
    """
    Đồ thị mạng 8 node.

        0 --- 1 --- 3 --- 5
        |  \\  |     |  /  |
        2 --- 4 --- 6 --- 7
    """

    NUM_NODES = 8

    EDGE_LIST: List[Tuple[int, int]] = [
        (0, 1), (0, 2), (1, 2), (1, 3), (1, 4),
        (2, 4), (3, 5), (3, 4), (4, 5), (4, 6),
        (5, 7), (6, 7), (2, 6),
    ]

    # Khoảng random cho tham số vật lý
    DELAY_RANGE          = (1.0,  10.0)   # ms
    BANDWIDTH_RANGE      = (20.0, 200.0)  # Mbps
    QUEUE_SIZE_CUR_RANGE = (20,   100)    # packets

    # Chuẩn hóa để đưa vào link_state_vector
    DELAY_NORM_MAX          = 10.0
    BANDWIDTH_NORM_MAX      = 200.0
    QUEUE_SIZE_CUR_NORM_MAX = 100.0

    def __init__(self, seed: int = None,
                 rng: Optional[np.random.Generator] = None):
        self.graph = nx.DiGraph()
        self._link_attrs: Dict[Tuple[int, int], LinkAttr] = {}
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self._build_structure()
        self.randomize_links()

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def _build_structure(self):
        self.graph.add_nodes_from(range(self.NUM_NODES))
        for src, dst in self.EDGE_LIST:
            for u, v in [(src, dst), (dst, src)]:
                self.graph.add_edge(u, v)

    def randomize_links(self):
        """
        Sinh ngẫu nhiên delay, bandwidth, queue_size_cur cho mọi link.
        Hai chiều của một cạnh vật lý chia sẻ cùng tham số.
        Gọi mỗi episode reset() để tạo topology mới.
        """
        self._link_attrs.clear()
        for u, v in self.EDGE_LIST:
            delay  = float(self._rng.uniform(*self.DELAY_RANGE))
            bw     = float(self._rng.uniform(*self.BANDWIDTH_RANGE))
            qsize  = int(self._rng.integers(*self.QUEUE_SIZE_CUR_RANGE))
            for a, b in [(u, v), (v, u)]:
                self._link_attrs[(a, b)] = LinkAttr(
                    delay=delay,
                    bandwidth=bw,
                    queue_size_cur=qsize,
                )

    # ------------------------------------------------------------------ #
    #  State access                                                        #
    # ------------------------------------------------------------------ #

    def link(self, u: int, v: int) -> LinkAttr:
        return self._link_attrs[(u, v)]

    def neighbors(self, node: int) -> List[int]:
        return list(self.graph.successors(node))

    def has_link(self, u: int, v: int) -> bool:
        return (u, v) in self._link_attrs

    def num_edges(self) -> int:
        return len(self._link_attrs)

    @property
    def adj_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(self.graph, weight=None).astype(np.float32)

    def link_state_vector(self) -> np.ndarray:
        """
        Shape (num_edges, 5):
          [delay_norm, bw_norm, queue_size_cur_norm, utilization, queue_util]

        Cột 0-2: tham số vật lý (thay đổi mỗi episode).
        Cột 3-4: trạng thái lưu lượng (thay đổi mỗi step).
        """
        rows = []
        for (u, v), attr in sorted(self._link_attrs.items()):
            rows.append([
                attr.delay          / self.DELAY_NORM_MAX,
                attr.bandwidth      / self.BANDWIDTH_NORM_MAX,
                attr.queue_size_cur / self.QUEUE_SIZE_CUR_NORM_MAX,
                attr.utilization,
                attr.queue_util,
            ])
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  send_traffic — MDP model                                            #
    # ------------------------------------------------------------------ #

    def send_traffic(
        self,
        path:        List[int],
        volume_mbps: float,
        is_first_hop: bool,
    ) -> Dict:
        """
        Xử lý traffic cho đoạn link CUỐI CÙNG trong path.
        Gọi mỗi step khi agent di chuyển từ path[-2] → path[-1].

        Args:
            path        : path đầy đủ tính đến step này, ví dụ [0, 1, 2].
            volume_mbps : lưu lượng của flow (Mbps / packets).
            is_first_hop: True nếu đây là hop đầu tiên (cộng volume vào
                          queue_used_cur), False nếu không (chỉ cộng
                          queue_used từ link trước).

        Returns:
            dict: total_delay (ms), dropped (bool), hops (int).
        """
        if len(path) < 2:
            return {"total_delay": 0.0, "dropped": False, "hops": 0}

        u = path[-2]   # node trước
        v = path[-1]   # node hiện tại (vừa di chuyển đến)
        attr = self._link_attrs[(u, v)]

        # ── Bước 1: lấy queue_used từ link trước ─────────────────────
        if len(path) >= 3:
            prev = path[-3]
            prev_attr      = self._link_attrs[(prev, u)]
            queue_used_prev = prev_attr.queue_used
            prev_attr.queue_used = 0.0          # reset sau khi lấy
        else:
            queue_used_prev = 0.0

        # ── Bước 2: cộng vào queue_used_cur ──────────────────────────
        if is_first_hop:
            # Hop đầu: cộng cả volume lẫn queue_used từ link trước
            attr.queue_used_cur += queue_used_prev + volume_mbps
        else:
            # Hop sau: volume đã được đưa vào queue từ bước trước
            # chỉ cộng queue_used được chuyển từ link trước
            attr.queue_used_cur += queue_used_prev

        # ── Bước 3: drop check ───────────────────────────────────────
        dropped = attr.queue_used_cur > attr.queue_size_cur

        # ── Bước 4: cập nhật load và queue_used ──────────────────────
        remain_bw = attr.bandwidth - attr.load   # băng thông còn trống
        if remain_bw <= attr.queue_used_cur:
            attr.load          += remain_bw
            attr.queue_used     = attr.load
            attr.queue_used_cur -= remain_bw
        else:
            attr.load          += attr.queue_used_cur
            attr.queue_used     = attr.load
            attr.queue_used_cur = 0.0

        return {
            "total_delay": attr.delay,
            "dropped":     dropped,
            "hops":        1,
        }

    # ------------------------------------------------------------------ #
    #  reduce_load — leaky bucket trên path                               #
    # ------------------------------------------------------------------ #

    def reduce_load(self, path: List[int], decay: float = 0.85) -> bool:
        """
        Sau mỗi step, xử lý tuần tự từng link trong path (trái→phải),
        sau đó decay load các link ngoài path.

        Args:
            path : path đầy đủ tính đến step này.
            decay: hệ số giảm load cho link ngoài path [0,1].
        """
        path_links = set()
        dropped = False

        if len(path) >= 2:
            for idx in range(len(path) - 1):
                u = path[idx]
                v = path[idx + 1]
                attr = self._link_attrs[(u, v)]
                path_links.add((u, v))

                # Tìm link tiếp theo trong path
                if idx + 2 < len(path):
                    w         = path[idx + 2]
                    next_attr = self._link_attrs[(v, w)]
                    has_next  = True
                else:
                    next_attr = None
                    has_next  = False

                # ── Phần A: chuyển load sang link kế tiếp ─────────────
                if attr.load != 0:
                    if has_next:
                        next_attr.queue_used_cur += attr.load
                        dropped = next_attr.queue_used_cur > next_attr.queue_size_cur
                    attr.load = 0.0

                # ── Phần B: xử lý queue_used_cur còn lại ──────────────
                if attr.queue_used_cur == 0.0:
                    attr.load = 0.0
                else:
                    attr.load  = 0.0
                    remain_bw  = attr.bandwidth   # load=0 nên remain=bw
                    if remain_bw <= attr.queue_used_cur:
                        attr.load           += remain_bw
                        attr.queue_used_cur -= remain_bw
                    else:
                        attr.load           += attr.queue_used_cur
                        attr.queue_used_cur  = 0.0

        # Decay link ngoài path (background leaky bucket)
        for (u, v), attr in self._link_attrs.items():
            if (u, v) not in path_links:
                attr.load           = attr.load           * decay
                attr.queue_used_cur = attr.queue_used_cur * decay
                attr.queue_used     = attr.queue_used     * decay
        
        return dropped

    # ------------------------------------------------------------------ #
    #  Background traffic                                                  #
    # ------------------------------------------------------------------ #

    def step_background(self, intensity: float = 0.3):
        """
        Thêm background traffic ngẫu nhiên lên 30–60% link.
        Gọi sau reset() để tạo trạng thái khởi đầu đa dạng.
        """
        all_links = list(self._link_attrs.keys())
        n = int(self._rng.integers(
            len(all_links) // 3,
            max(len(all_links) // 3 + 1, len(all_links) * 2 // 3)
        ))
        for idx in self._rng.choice(len(all_links), size=n, replace=False):
            u, v = all_links[idx]
            attr  = self._link_attrs[(u, v)]
            extra = float(self._rng.uniform(0, intensity * attr.bandwidth))
            attr.load           = min(attr.bandwidth,      attr.load + extra)
            attr.queue_used_cur = min(attr.queue_size_cur, attr.queue_used_cur + extra)

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, rng: Optional[np.random.Generator] = None):
        """Sinh lại tham số link ngẫu nhiên và xóa trạng thái traffic."""
        if rng is not None:
            self._rng = rng
        self.randomize_links()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def shortest_path(self, src: int, dst: int) -> List[int]:
        try:
            return nx.shortest_path(self.graph, src, dst, weight="delay")
        except nx.NetworkXNoPath:
            return []

    def summary(self) -> Dict:
        utils  = [a.utilization for a in self._link_attrs.values()]
        queues = [a.queue_util  for a in self._link_attrs.values()]
        return {
            "avg_utilization": float(np.mean(utils))  if utils else 0.0,
            "max_utilization": float(np.max(utils))   if utils else 0.0,
            "avg_queue_util":  float(np.mean(queues)) if queues else 0.0,
            "congested_links": sum(1 for a in self._link_attrs.values()
                                   if a.is_congested),
        }