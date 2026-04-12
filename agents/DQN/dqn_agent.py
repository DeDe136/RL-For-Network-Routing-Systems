"""
agents/DQN/dqn_agent.py

Vanilla DQN Agent cho bài toán định tuyến mạng 8 node.

Đặc điểm:
  - Experience Replay: phá vỡ correlation giữa các transition liên tiếp.
  - Target Network: ổn định quá trình học bằng cách giữ target cố định
    trong target_update_freq bước.
  - Action Masking: chỉ cho phép chọn next-hop có link thực sự tồn tại.
    Áp dụng cả lúc select_action (inference) và lúc tính target Q
    (training) — tránh agent học dựa vào Q-value của action không hợp lệ.
  - Gradient Clipping: tránh gradient exploding khi loss lớn.

Hạn chế đã biết của Vanilla DQN (so với Double DQN):
  - Tính target Q bằng target_net.max() → cùng network chọn action
    VÀ tính value → có xu hướng overestimate Q-value.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any

from agents.base_agent import BaseAgent
from agents.DQN.q_network import QNetwork
from agents.DQN.replay_buffer import ReplayBuffer
from env.DQN.spaces import obs_to_flat, NUM_NODES


class DQNAgent(BaseAgent):
    """
    Agent DQN với kỹ thuật Experience Replay và Target Network.

    Kế thừa từ BaseAgent, ghi đè các phương thức:
        - select_action
        - update
        - save / load

    Các phương thức bổ trợ:
        - set_neighbor_mask: cập nhật mặt nạ hàng xóm từ topology.
        - remember: lưu transition vào replay buffer.
        - best_path: tìm đường đi tốt nhất theo chính sách hiện tại.
    """
    def __init__(self, n_states: int, n_actions: int,
                 config: Dict[str, Any]):
        """
        Khởi tạo DQN Agent.

        Tham số:
            n_states (int): Không dùng trực tiếp, giữ để tương thích BaseAgent.
            n_actions (int): Số lượng hành động (số nút mạng, mặc định 8).
            config (Dict): Từ điển chứa siêu tham số.
        """
        super().__init__(n_states, n_actions, config)

        self.input_dim          = config.get("input_dim",          132)
        self.hidden_dims        = config.get("hidden_dims",        [128, 128])
        self.lr                 = config.get("lr",                 1e-3)
        self.gamma              = config.get("gamma",              0.99)
        self.epsilon            = config.get("epsilon",            1.0)
        self.eps_min            = config.get("eps_min",            0.01)
        self.eps_decay          = config.get("eps_decay",          0.9995)
        self.batch_size         = config.get("batch_size",         64)
        self.buffer_capacity    = config.get("buffer_capacity",    10_000)
        self.target_update_freq = config.get("target_update_freq", 100)
        self.learn_start        = config.get("learn_start",        self.batch_size)

        # Thiết bị tính toán (CPU/GPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Mạng Q chính và mạng mục tiêu
        self.q_net      = QNetwork(self.input_dim, n_actions,
                                   self.hidden_dims).to(self.device)
        self.target_net = QNetwork(self.input_dim, n_actions,
                                   self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Mạng mục tiêu chỉ dùng để suy luận

        # Bộ tối ưu và hàm mất mát
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn    = nn.MSELoss()

        # Bộ nhớ trải nghiệm
        self.memory     = ReplayBuffer(self.buffer_capacity)

        # Mặt nạ hàng xóm: ma trận kề (8x8), 1 = có kết nối, 0 = không
        self.neighbor_mask = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
        self.steps_done    = 0

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def set_neighbor_mask(self, adj_matrix: np.ndarray) -> None:
        """Cập nhật mask từ adj_matrix của topology (8×8, binary)."""
        self.neighbor_mask = adj_matrix.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: Dict) -> int:
        """
        ε-greedy với action masking.

        Exploration (training, xác suất ε):
            Chọn ngẫu nhiên đều trong valid neighbors.
        Exploitation:
            Tính Q-values từ q_net, gán -inf cho invalid actions,
            lấy argmax. Tie-breaking: random trong tất cả action
            cùng đạt max Q.

        Args:
            obs: dict observation từ env, có key "current_node",
                 "dst_node", "link_states".
        """
        current_node = obs["current_node"]
        valid = np.where(self.neighbor_mask[current_node] > 0)[0]
        if len(valid) == 0:
            return 0  # fallback, không nên xảy ra

        # Exploration
        if self.training and np.random.random() < self.epsilon:
            return int(np.random.choice(valid))

        # Exploitation: masked Q-values + tie-breaking
        q_vals       = self._masked_q_values(obs_to_flat(obs), current_node)
        max_q        = np.max(q_vals[valid])
        best_actions = valid[q_vals[valid] == max_q]
        return int(np.random.choice(best_actions))

    def remember(self, obs: Dict, action: int, reward: float,
                 next_obs: Dict, done: bool) -> None:
        """Lưu một transition vào replay buffer."""
        self.memory.push(
            obs_to_flat(obs), action, reward,
            obs_to_flat(next_obs), done,
        )

    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        Vanilla DQN update.

        Target:
            y = r + γ · max_{a'∈valid} Q_target(s', a')   nếu not done
            y = r                                           nếu done

        Lưu ý: target_net chọn action có max Q trong valid neighbors
        (có action masking) — tránh Q-value của action không hợp lệ
        kéo target lên cao.

        Hạn chế (vanilla): cùng target_net vừa chọn action vừa tính
        value → overestimation. Double DQN dùng q_net chọn action.
        """
        if len(self.memory) < self.learn_start:
            return {"loss": None}

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a) hiện tại
        q_values = self.q_net(states_t).gather(1, actions_t)

        # Target Q với action masking trên next state
        with torch.no_grad():
            next_q_all = self.target_net(next_states_t)      # (B, 8)

            # Xây mask (B, 8): decode current_node từ next_states[:, 0]
            mask = self._build_mask_batch(next_states_t)     # (B, 8) bool
            next_q_all[~mask] = float("-inf")

            # Vanilla DQN: target_net.max() vừa chọn action vừa tính value
            next_q_values = next_q_all.max(dim=1, keepdim=True)[0]  # (B,1)

            target_q = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.training:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        # Sync target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _masked_q_values(self, state_flat: np.ndarray,
                         current_node: int) -> np.ndarray:
        """Q-values từ q_net với invalid actions = -inf."""
        t = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(t).cpu().numpy().flatten()
        mask = self.neighbor_mask[current_node]
        return np.where(mask > 0, q_vals, -np.inf)

    def _build_mask_batch(self,
                          next_states_t: torch.Tensor) -> torch.Tensor:
        """
        Xây mask (B, 8) bool cho cả batch.
        Decode current_node từ next_states[:, 0] × (NUM_NODES-1).
        """
        B    = next_states_t.shape[0]
        mask = torch.zeros(B, NUM_NODES, dtype=torch.bool,
                           device=self.device)
        nodes = (next_states_t[:, 0] * (NUM_NODES - 1)).round().long()
        for i in range(B):
            valid = np.where(
                self.neighbor_mask[int(nodes[i].item())] > 0
            )[0]
            mask[i, valid] = True
        return mask

    def best_path(self, src: int, dst: int, topo,
                  volume_mbps: float = 10.0,
                  max_hops: int = 8) -> List[int]:
        """
        Greedy rollout từ src → dst theo MDP traffic model.

        Mỗi bước:
          1. Đọc link_states hiện tại (utilization, queue_util thay đổi thực tế).
          2. Chọn next-hop theo greedy masked Q (tie-breaking random).
          3. Gọi topo.send_traffic(path, volume, is_first_hop) — cập nhật
             queue_used_cur, load, queue_used theo MDP spec.
          4. Gọi topo.reduce_load(path) — chuyển load sang link tiếp theo
             và decay link ngoài path (leaky bucket).

        Link_states phản ánh trạng thái mạng thực sau mỗi hop, agent
        phải đọc chúng thay vì chỉ nhớ (src, dst).

        Args:
            src, dst    : cặp node nguồn/đích.
            topo        : NetworkTopology instance.
            volume_mbps : lưu lượng gửi qua mỗi link.
            max_hops    : giới hạn hop để tránh vòng lặp vô hạn.

        Returns:
            List[int]: path từ src đến dst (bao gồm src).
        """
        was_training = self.training
        self.eval_mode()

        path    = [src]
        visited = {src}
        cur     = src
        hops    = 0

        for _ in range(max_hops):
            if cur == dst:
                break

            # Đọc link_states sau khi traffic bước trước đã cập nhật
            obs = {
                "current_node": cur,
                "dst_node":     dst,
                "link_states":  topo.link_state_vector(),
            }
            valid = [a for a in
                     np.where(self.neighbor_mask[cur] > 0)[0]
                     if a not in visited]
            valid = np.array(valid)
            if len(valid) == 0:
                break

            q_vals    = self._masked_q_values(obs_to_flat(obs), cur)
            max_q        = np.max(q_vals[valid])
            best_actions = valid[q_vals[valid] == max_q]
            nxt       = int(np.random.choice(best_actions))

            path.append(nxt)
            visited.add(nxt)
            hops += 1

            # send_traffic theo MDP model:
            # is_first_hop=True chỉ ở hop đầu tiên (cộng volume vào queue)
            is_first_hop = (hops == 1)
            topo.send_traffic(
                path        = path,
                volume_mbps = volume_mbps,
                is_first_hop= is_first_hop,
            )

            # reduce_load: leaky bucket trên path + decay ngoài path
            topo.reduce_load(path)

            cur = nxt

        self.training = was_training
        return path

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps_done": self.steps_done,
        }, filepath)
        print(f"DQN saved → {filepath}")

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt.get("epsilon",    self.eps_min)
        self.steps_done = ckpt.get("steps_done", 0)
        print(f"DQN loaded ← {filepath}")

    def network_summary(self) -> str:
        total = sum(p.numel() for p in self.q_net.parameters())
        return (f"DQN | params={total:,} | "
                f"input={self.input_dim} | hidden={self.hidden_dims} | "
                f"output={self.n_actions} | device={self.device} | "
                f"ε={self.epsilon:.4f}")