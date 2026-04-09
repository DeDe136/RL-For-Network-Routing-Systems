"""
agents/DQN/dqn_agent.py

Deep Q‑Network Agent cho bài toán định tuyến mạng, kế thừa từ BaseAgent.
Có cơ chế che (mask) các hành động không hợp lệ (không có kết nối trực tiếp).
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Optional, Any

from agents.base_agent import BaseAgent
from agents.DQN.q_network import QNetwork
from agents.DQN.replay_buffer import ReplayBuffer
from env.Q_Learning.spaces import obs_to_flat, NUM_NODES


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

    def __init__(self, n_states: int, n_actions: int, config: Dict[str, Any]):
        """
        Khởi tạo DQN Agent.

        Tham số:
            n_states (int): Không dùng trực tiếp, giữ để tương thích BaseAgent.
            n_actions (int): Số lượng hành động (số nút mạng, mặc định 8).
            config (Dict): Từ điển chứa siêu tham số.
        """
        super().__init__(n_states, n_actions, config)

        # Trích xuất siêu tham số từ config
        # input_dim = 2 (node info) + 26 cạnh * 5 đặc trưng = 132
        self.input_dim = config.get("input_dim", 132)
        self.hidden_dims = config.get("hidden_dims", [128, 128])
        self.lr = config.get("lr", 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.eps_min = config.get("eps_min", 0.01)
        self.eps_decay = config.get("eps_decay", 0.9995)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_capacity = config.get("buffer_capacity", 10000)
        self.target_update_freq = config.get("target_update_freq", 100)

        # Thiết bị tính toán (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mạng Q chính và mạng mục tiêu
        self.q_net = QNetwork(self.input_dim, self.n_actions, self.hidden_dims).to(self.device)
        self.target_net = QNetwork(self.input_dim, self.n_actions, self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Mạng mục tiêu chỉ dùng để suy luận

        # Bộ tối ưu và hàm mất mát
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

        # Bộ nhớ trải nghiệm
        self.memory = ReplayBuffer(self.buffer_capacity)

        # Mặt nạ hàng xóm: ma trận kề (8x8), 1 = có kết nối, 0 = không
        self.neighbor_mask = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
        self.steps_done = 0   # Đếm số bước đã thực hiện (dùng để cập nhật target network)

    # ------------------------------------------------------------------ #
    #  Phương thức ghi đè từ BaseAgent                                    #
    # ------------------------------------------------------------------ #

    def select_action(self, state: Any) -> int:
        """
        Chọn hành động theo chính sách epsilon‑greedy, có áp dụng mặt nạ hàng xóm.
        `state` là dictionary observation từ môi trường.

        Tham số:
            state (Dict): Observation chứa "current_node", "dst_node", "link_states".

        Trả về:
            int: Hành động được chọn (nút tiếp theo).
        """
        explore = self.training and (np.random.random() < self.epsilon)
        return self._select_action_impl(state, explore=explore)

    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        Thực hiện một bước học: lấy batch từ replay buffer và cập nhật mạng Q.

        Trả về:
            Dict: Chứa giá trị loss (nếu có), ví dụ {"loss": 0.023}.
        """
        if len(self.memory) < self.batch_size:
            return {"loss": None}

        # Lấy mẫu một batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Chuyển sang tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Giá trị Q hiện tại cho các hành động đã chọn
        q_values = self.q_net(states).gather(1, actions)

        # Tính giá trị mục tiêu sử dụng mạng mục tiêu
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Tính loss và lan truyền ngược
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Có thể thêm gradient clipping nếu cần
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Giảm epsilon theo thời gian
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        # Cập nhật mạng mục tiêu định kỳ
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item()}

    # ------------------------------------------------------------------ #
    #  Phương thức đặc thù của DQN                                        #
    # ------------------------------------------------------------------ #

    def set_neighbor_mask(self, adj_matrix: np.ndarray) -> None:
        """
        Cập nhật mặt nạ hàng xóm từ ma trận kề của topology.

        Tham số:
            adj_matrix (np.ndarray): Ma trận nhị phân (8x8), 1 nếu có liên kết.
        """
        self.neighbor_mask = adj_matrix.astype(np.float32)

    def remember(self, obs: Dict, action: int, reward: float,
                 next_obs: Dict, done: bool) -> None:
        """
        Lưu một transition vào replay buffer.

        Tham số:
            obs: Observation hiện tại.
            action: Hành động đã chọn.
            reward: Phần thưởng nhận được.
            next_obs: Observation kế tiếp.
            done: Cờ kết thúc episode.
        """
        state_flat = self._preprocess_state(obs)
        next_state_flat = self._preprocess_state(next_obs)
        self.memory.push(state_flat, action, reward, next_state_flat, done)

    def best_path(self, src: int, dst: int, topo) -> List[int]:
        """
        Tìm đường đi tham lam (greedy) từ src đến dst sử dụng mạng Q hiện tại.
        Dùng để kiểm tra chính sách sau huấn luyện.

        Tham số:
            src (int): Nút nguồn.
            dst (int): Nút đích.
            topo: Đối tượng NetworkTopology (để lấy vector trạng thái liên kết).

        Trả về:
            List[int]: Danh sách các nút trên đường đi.
        """
        # Tạo một môi trường giả để sinh observation
        from env.Q_Learning.routing_env import NetworkRoutingEnv
        dummy_env = NetworkRoutingEnv()
        dummy_env.topo = topo
        obs, _ = dummy_env.reset()
        # Ghi đè src, dst
        obs["current_node"] = src
        obs["dst_node"] = dst
        dummy_env._src = src
        dummy_env._dst = dst
        dummy_env._current_node = src

        path = [src]
        max_hops = 20
        for _ in range(max_hops):
            if path[-1] == dst:
                break
            # Cập nhật trạng thái liên kết (phòng khi topo thay đổi)
            obs["link_states"] = topo.link_state_vector()
            # Chuyển sang chế độ đánh giá (không exploration)
            was_training = self.training
            self.eval_mode()
            action = self.select_action(obs)
            self.training = was_training
            # Tránh lặp vô hạn hoặc hành động không hợp lệ
            if action == path[-1] or not topo.has_link(path[-1], action):
                break
            path.append(action)
            obs["current_node"] = action
        return path

    # ------------------------------------------------------------------ #
    #  Các hàm trợ giúp nội bộ                                            #
    # ------------------------------------------------------------------ #

    def _preprocess_state(self, obs: Dict) -> np.ndarray:
        """
        Chuyển đổi observation dictionary thành vector phẳng cho mạng nơ‑ron.

        Tham số:
            obs (Dict): Observation từ môi trường.

        Trả về:
            np.ndarray: Vector trạng thái đã làm phẳng.
        """
        return obs_to_flat(obs)

    def _masked_q_values(self, state_flat: np.ndarray, current_node: int) -> np.ndarray:
        """
        Tính giá trị Q và áp dụng mặt nạ cho các hành động không phải hàng xóm.

        Tham số:
            state_flat: Vector trạng thái phẳng.
            current_node: Nút hiện tại (dùng để lấy hàng tương ứng trong mặt nạ).

        Trả về:
            np.ndarray: Mảng Q‑values với các hành động không hợp lệ được gán -inf.
        """
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(state_tensor).cpu().numpy().flatten()
        mask = self.neighbor_mask[current_node]
        # Gán -inf cho các hành động không có kết nối
        q_vals = np.where(mask > 0, q_vals, -np.inf)
        return q_vals

    def _select_action_impl(self, obs: Dict, explore: bool) -> int:
        """
        Logic lựa chọn hành động, tách riêng để dùng lại.

        Tham số:
            obs: Observation hiện tại.
            explore (bool): Nếu True thì dùng epsilon‑greedy, ngược lại dùng greedy.

        Trả về:
            int: Hành động được chọn.
        """
        state_flat = self._preprocess_state(obs)
        current_node = obs["current_node"]

        if explore:
            # Khám phá: chọn ngẫu nhiên trong các hàng xóm hợp lệ
            valid_actions = np.where(self.neighbor_mask[current_node] > 0)[0]
            if len(valid_actions) == 0:
                return 0  # fallback (không nên xảy ra)
            return int(np.random.choice(valid_actions))
        else:
            # Khai thác: chọn hành động có Q cao nhất (đã được che)
            q_vals = self._masked_q_values(state_flat, current_node)
            return int(np.argmax(q_vals))

    # ------------------------------------------------------------------ #
    #  Lưu và nạp mô hình                                                 #
    # ------------------------------------------------------------------ #

    def save(self, filepath: str) -> None:
        """Lưu state dict của mạng Q."""
        torch.save(self.q_net.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Nạp state dict cho mạng Q và mạng mục tiêu."""
        state_dict = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def q_table_summary(self) -> str:
        """DQN không dùng Q‑table nên trả về thông báo."""
        return "DQN does not have a Q‑table."