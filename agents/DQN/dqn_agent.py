"""
agents/DQN/dqn_agent.py
 
D3QN Agent — Double DQN + Dueling DQN + Prioritized Experience Replay (PER).
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Ba cải tiến và lý do kết hợp
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
1. Double DQN (van Hasselt et al. 2016) — chống overestimation:
   Vanilla DQN: target = r + γ · max_a Q_target(s', a)
     → target_net vừa CHỌN action vừa TÍNH value → overestimate.
   Double DQN : target = r + γ · Q_target(s', argmax_a Q_online(s', a))
     → q_net CHỌN action tốt nhất, target_net CHỈ tính value của action đó.
     → Tách biệt 2 vai trò → bias giảm đáng kể.
 
2. Dueling DQN (Wang et al. 2016) — học hiệu quả hơn:
   Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
     V(s) : giá trị trạng thái (state value) — học từ MỌI action.
     A(s,a): lợi thế hành động (advantage) — học từ action được chọn.
   Lợi ích:
     - V(s) cập nhật dù action nào được thực hiện → học nhanh hơn.
     - Quan trọng với routing: khi 2 đường đều tốt, V(s) vẫn cập nhật
       ngay cả khi agent chọn ngẫu nhiên một trong 2.
 
3. Prioritized Experience Replay — học từ sai lầm quan trọng:
   P(i) = p_i^α / Σ p_k^α ,  p_i = |δ_i| + ε  (TD-error)
     → Transition có TD-error lớn được sample nhiều hơn.
     → IS weights bù trừ bias do sampling không đều.
   Lợi ích:
     - Tập trung vào những tình huống agent còn sai nhiều.
     - Đặc biệt tốt với routing: tình huống congestion cao thường
       có reward âm lớn → TD-error lớn → được học nhiều hơn.
 
Kết hợp cả 3 tạo nên D3QN:
  - Ổn định hơn (Double DQN)
  - Học hiệu quả từ mọi trạng thái (Dueling)
  - Ưu tiên học từ tình huống quan trọng (PER)
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Interface (backward-compatible với Vanilla DQN):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
  DQNAgent.__init__(n_states, n_actions, config)
  agent.select_action(obs)        → int
  agent.remember(obs, a, r, obs', done)
  agent.update()                  → {"loss": float, "epsilon": float}
  agent.best_path(src, dst, topo) → List[int]
  agent.save(path) / agent.load(path)
  agent.network_summary()         → str
 
Hyperparameters mới trong config:
  use_double  : bool  — bật/tắt Double DQN (default True)
  use_dueling : bool  — bật/tắt Dueling architecture (default True)
  use_per     : bool  — bật/tắt Prioritized Replay (default True)
  per_alpha   : float — exponent ưu tiên hoá [0=uniform, 1=full PER] (0.6)
  per_beta_start : float — IS weight ban đầu (0.4)
  per_beta_frames: int  — số frame để β tăng từ start → 1.0 (100000)
  per_eps     : float — epsilon nhỏ tránh priority=0 (1e-6)
"""
 
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any
 
from agents.base_agent import BaseAgent
from agents.DQN.q_network import QNetwork, DuelingQNetwork
from agents.DQN.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from env.DQN.spaces import obs_to_flat, NUM_NODES
 
 
class DQNAgent(BaseAgent):
 
    def __init__(self, n_states: int, n_actions: int,
                 config: Dict[str, Any]):
        super().__init__(n_states, n_actions, config)
 
        # ── Hyperparameters cơ bản ────────────────────────────────────
        self.input_dim          = config.get("input_dim",          158)
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
        self.purge_threshold    = config.get("purge_threshold",    0.95)
 
        # ── Flags bật/tắt từng cải tiến ──────────────────────────────
        self.use_double  = config.get("use_double",  True)
        self.use_dueling = config.get("use_dueling", True)
        self.use_per     = config.get("use_per",     True)
 
        # ── PER hyperparameters ───────────────────────────────────────
        self.per_alpha       = config.get("per_alpha",        0.6)
        self.per_beta_start  = config.get("per_beta_start",   0.4)
        self.per_beta_frames = config.get("per_beta_frames",  100_000)
        self.per_eps         = config.get("per_eps",          1e-6)
        self._per_frame      = 0   # đếm số update steps để tăng β
 
        # Thiết bị tính toán (CPU/GPU)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device("xpu")
            print("🚀 Đang dùng Intel XPU (GPU Intel)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Đang dùng NVIDIA CUDA")
        else:
            self.device = torch.device("cpu")
            print("📟 Đang dùng CPU")
 
        # ── Networks (Dueling hoặc Standard) ─────────────────────────
        NetworkClass = DuelingQNetwork if self.use_dueling else QNetwork
        self.q_net      = NetworkClass(self.input_dim, n_actions,
                                       self.hidden_dims).to(self.device)
        self.target_net = NetworkClass(self.input_dim, n_actions,
                                       self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
 
        # ── Optimizer và Loss ─────────────────────────────────────────
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        # MSELoss với reduction="none" để nhân IS weights per-sample
        self.loss_fn   = nn.MSELoss(reduction="none")
 
        # ── Replay Buffer (PER hoặc Uniform) ─────────────────────────
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity  = self.buffer_capacity,
                alpha     = self.per_alpha,
                per_eps   = self.per_eps,
            )
        else:
            # ReplayBuffer FIFO thuần: không có purge_threshold
            self.memory = ReplayBuffer(self.buffer_capacity)
 
        # ── Neighbor mask ─────────────────────────────────────────────
        self.neighbor_mask = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
        self.steps_done    = 0
 
    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #
 
    def set_neighbor_mask(self, adj_matrix: np.ndarray) -> None:
        """Cập nhật mask từ adj_matrix của topology (8×8, binary)."""
        self.neighbor_mask = adj_matrix.astype(np.float32)
 
    # ------------------------------------------------------------------ #
    #  β schedule cho PER                                                  #
    # ------------------------------------------------------------------ #
 
    def _current_beta(self) -> float:
        """
        β tăng tuyến tính từ per_beta_start → 1.0 trong per_beta_frames bước.
        β = 1.0 → IS weights hoàn toàn unbiased.
        """
        progress = min(1.0, self._per_frame / max(1, self.per_beta_frames))
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)
 
    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #
 
    def select_action(self, obs: Dict) -> int:
        """
        ε-greedy với action masking.
 
        Exploration (training, xác suất ε): chọn ngẫu nhiên valid neighbor.
        Exploitation: Dueling/Standard Q-values, invalid = -inf,
                      tie-breaking random trong tất cả action cùng max Q.
        """
        current_node = obs["current_node"]
        valid = np.where(self.neighbor_mask[current_node] > 0)[0]
        if len(valid) == 0:
            return 0   # fallback, không nên xảy ra
 
        if self.training and np.random.random() < self.epsilon:
            return int(np.random.choice(valid))
 
        q_vals       = self._masked_q_values(obs_to_flat(obs), current_node)
        max_q        = np.max(q_vals[valid])
        best_actions = valid[q_vals[valid] == max_q]
        return int(np.random.choice(best_actions))
 
    def remember(self, obs: Dict, action: int, reward: float,
                 next_obs: Dict, done: bool) -> None:
        """Lưu transition vào replay buffer (PER hoặc Uniform)."""
        self.memory.push(
            obs_to_flat(obs), action, reward,
            obs_to_flat(next_obs), done,
        )
 
    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        D3QN update step: Double DQN target + IS-weighted loss + priority update.
 
        Target (Double DQN):
            a*  = argmax_{a ∈ valid} Q_online(s', a)   ← q_net CHỌN
            y   = r + γ · Q_target(s', a*)              ← target_net TÍNH
          vs Vanilla DQN:
            y   = r + γ · max_a Q_target(s', a)   (cùng target_net làm cả 2)
 
        Loss với IS weights (PER):
            L = mean(w_i · (Q(s,a) - y_i)^2)
          vs Vanilla:
            L = mean((Q(s,a) - y_i)^2)
 
        Sau update → cập nhật priorities với |TD-error| mới.
 
        Returns:
            dict với keys: loss, epsilon, beta (PER), mean_priority (PER).
        """
        if len(self.memory) < self.learn_start:
            return {"loss": None}
 
        # ── Sample batch ─────────────────────────────────────────────
        if self.use_per:
            beta = self._current_beta()
            (states, actions, rewards, next_states, dones,
             is_weights, tree_indices) = self.memory.sample(self.batch_size, beta)
            is_weights_t = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.batch_size)
            is_weights_t = None
            tree_indices = None
            beta         = None
 
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
 
        # ── Q(s,a) hiện tại ───────────────────────────────────────────
        q_values = self.q_net(states_t).gather(1, actions_t)  # (B, 1)
 
        # ── Target Q theo Double DQN hoặc Vanilla ────────────────────
        with torch.no_grad():
            mask = self._build_mask_batch(next_states_t)    # (B, 8) bool
 
            if self.use_double:
                # Double DQN: q_net chọn action, target_net tính value
                next_q_online = self.q_net(next_states_t)   # (B, 8)
                next_q_online_masked = next_q_online.clone()
                next_q_online_masked[~mask] = float("-inf")
                best_actions = next_q_online_masked.argmax(dim=1, keepdim=True)  # (B,1)
 
                next_q_target = self.target_net(next_states_t)  # (B, 8)
                next_q_values = next_q_target.gather(1, best_actions)  # (B,1)
            else:
                # Vanilla DQN: target_net làm cả chọn action lẫn tính value
                next_q_target = self.target_net(next_states_t)  # (B, 8)
                next_q_target[~mask] = float("-inf")
                next_q_values = next_q_target.max(dim=1, keepdim=True)[0]  # (B,1)
 
            target_q = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)
 
        # ── TD-error và Loss ─────────────────────────────────────────
        td_errors = (q_values - target_q).detach().cpu().numpy().flatten()
 
        # element-wise loss: (B, 1)
        element_loss = self.loss_fn(q_values, target_q)
 
        if self.use_per and is_weights_t is not None:
            # IS-weighted loss: mỗi sample nhân với w_i
            loss = (is_weights_t * element_loss).mean()
        else:
            loss = element_loss.mean()
 
        # ── Backprop ─────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
 
        # ── Cập nhật priorities (PER) ─────────────────────────────────
        if self.use_per and tree_indices is not None:
            self.memory.update_priorities(tree_indices, td_errors)
            self._per_frame += 1
 
        # ── Decay epsilon ─────────────────────────────────────────────
        if self.training:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
 
        # ── Sync target network ───────────────────────────────────────
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
 
        result: Dict[str, Any] = {
            "loss":    loss.item(),
            "epsilon": self.epsilon,
        }
        if self.use_per:
            result["beta"]          = beta
            result["mean_priority"] = float(np.mean(np.abs(td_errors)))
        return result

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
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "steps_done":  self.steps_done,
            "per_frame":   self._per_frame,
            # Lưu flags để load đúng kiến trúc
            "use_double":  self.use_double,
            "use_dueling": self.use_dueling,
            "use_per":     self.use_per,
        }, filepath)
        print(f"D3QN saved → {filepath}")
 
    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon      = ckpt.get("epsilon",    self.eps_min)
        self.steps_done   = ckpt.get("steps_done", 0)
        self._per_frame   = ckpt.get("per_frame",  0)
        print(f"D3QN loaded ← {filepath}")
 
    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #
 
    def network_summary(self) -> str:
        total  = sum(p.numel() for p in self.q_net.parameters())
        flags  = []
        if self.use_double:  flags.append("Double")
        if self.use_dueling: flags.append("Dueling")
        if self.use_per:     flags.append("PER")
        algo   = "+".join(flags) if flags else "Vanilla"
        arch   = "Dueling" if self.use_dueling else "Standard"
        return (
            f"D3QN [{algo}] | arch={arch} | params={total:,} | "
            f"input={self.input_dim} | hidden={self.hidden_dims} | "
            f"output={self.n_actions} | device={self.device} | "
            f"ε={self.epsilon:.4f}"
            + (f" | β={self._current_beta():.3f}" if self.use_per else "")
        )