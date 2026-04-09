"""
scripts/DQN/train.py

Vòng lặp huấn luyện cho DQN Agent trên môi trường NetworkRouting-v0.
Chạy:
    python scripts/DQN/train.py
    python scripts/DQN/train.py --episodes 5000
"""

import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env  # noqa - đăng ký environment
import gymnasium as gym

from agents.DQN.dqn_agent import DQNAgent
from utils.logger import Logger


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config",   default="configs/env_config.yaml")
    parser.add_argument("--agent-config", default="configs/agent_config.yaml")
    parser.add_argument("--episodes",     type=int, default=None)
    args = parser.parse_args()

    ecfg = load_cfg(args.env_config)
    acfg = load_cfg(args.agent_config)
    tcfg = acfg["training"]

    num_episodes = args.episodes or tcfg["num_episodes"]
    ckpt_dir     = tcfg["checkpoint_dir"]
    target_dir = os.path.join(ckpt_dir, "DQN")
    os.makedirs(target_dir, exist_ok=True)

    # Khởi tạo môi trường
    environment = gym.make(
        "NetworkRouting-DQN-v0",
        mean_traffic_mbps=ecfg["env"]["mean_traffic_mbps"],
        max_hops=ecfg["env"]["max_hops"],
        seed=ecfg["env"]["seed"],
    )

    # Khởi tạo DQN Agent
    agent = DQNAgent(
        n_states=1,            # không dùng trực tiếp
        n_actions=8,           # NUM_NODES
        config=acfg["dqn"]
    )

    # Thiết lập mặt nạ hàng xóm từ topology
    raw_env = environment.unwrapped
    agent.set_neighbor_mask(raw_env.topo.adj_matrix)

    full_log_dir = os.path.join(tcfg["log_dir"], "DQN")
    os.makedirs(full_log_dir, exist_ok=True)
    logger = Logger(
        log_dir=full_log_dir,
        print_every=tcfg["print_every"],
    )

    print(f"Huấn luyện DQN trên NetworkRouting-v0 (topology động) - {num_episodes} episodes")
    print(f"Thiết bị: {agent.device}")
    print()

    for ep in range(1, num_episodes + 1):
        obs, _ = environment.reset()
        ep_reward = 0.0
        steps = 0
        done = False

        # Agent mặc định ở chế độ huấn luyện
        agent.train_mode()

        while not done:
            # Chọn hành động
            action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated

            # Lưu transition vào bộ nhớ
            agent.remember(obs, action, reward, next_obs, done)

            # Thực hiện bước học
            loss_info = agent.update()

            ep_reward += reward
            steps += 1
            obs = next_obs

        logger.log({
            "reward":   ep_reward,
            "epsilon":  agent.epsilon,
            "hops":     info["hops"],
            "delay_ms": info["total_delay"],
            "dropped":  float(info["dropped"]),
            "avg_util": info["avg_utilization"],
        })

        if ep % tcfg["print_every"] == 0:
            print(f"Episode {ep:4d} | Reward: {ep_reward:+.2f} | Epsilon: {agent.epsilon:.3f} | "
                  f"Hops: {info['hops']} | Delay: {info['total_delay']:.1f}ms | Drop: {info['dropped']}")

        if ep % tcfg["save_every"] == 0:
            agent.save(os.path.join(ckpt_dir, f"dqn_ep{ep}.pt"))

    environment.close()
    logger.close()
    agent.save(os.path.join(ckpt_dir, "dqn_final.pt"))

    # Demo chính sách tham lam sau huấn luyện
    print("\n=== Đường đi tham lam sau huấn luyện ===")
    agent.eval_mode()   # tắt exploration
    for src, dst in [(0,7), (1,6), (2,5), (3,7), (0,5)]:
        path = agent.best_path(src, dst, raw_env.topo)
        print(f"  {src} → {dst} : {path}")


if __name__ == "__main__":
    main()