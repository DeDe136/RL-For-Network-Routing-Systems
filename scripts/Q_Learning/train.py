"""
scripts/Q_Learning/train.py

Training loop cho Q-Learning agent trên NetworkRouting-v0.

Chạy:
    python scripts/Q_Learning/train.py
    python scripts/Q_Learning/train.py --episodes 5000
"""

import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env  # noqa — kích hoạt gymnasium.register
import gymnasium as gym

from agents.Q_Learning.ql_agent_for_delay import QLearningAgent
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
    target_dir = os.path.join(ckpt_dir, "Q_Learning")
    os.makedirs(target_dir, exist_ok=True)

    # Khởi tạo env
    environment = gym.make(
        "NetworkRouting-QL-v0",
        mean_traffic_mbps=ecfg["env"]["mean_traffic_mbps"],
        max_hops=ecfg["env"]["max_hops"],
        seed=ecfg["env"]["seed"],
    )

    # Khởi tạo agent
    agent = QLearningAgent(config=acfg["q_learning"])

    # Set neighbor mask từ topology
    raw_env = environment.unwrapped
    agent.set_neighbor_mask(raw_env.topo.adj_matrix)

    full_log_dir = os.path.join(tcfg["log_dir"], "Q_learning")
    os.makedirs(full_log_dir, exist_ok=True)
    logger = Logger(
        log_dir=full_log_dir,
        print_every=tcfg["print_every"],
    )

    print(f"Training Q-Learning on NetworkRouting-v0 ({num_episodes} episodes)...")
    print(f"Q-table shape: {agent.Q.shape}  (nodes × dsts × actions)")
    print()

    for ep in range(1, num_episodes + 1):
        obs, _ = environment.reset()
        state  = (obs["current_node"], obs["dst_node"])

        ep_reward  = 0.0
        ep_td      = 0.0
        steps      = 0
        done       = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated

            next_state = (next_obs["current_node"], next_obs["dst_node"])

            result = agent.update(state, action, reward, next_state, done)
            ep_td     += result.get("td_error", 0.0)
            ep_reward += reward
            steps     += 1
            state      = next_state

        logger.log({
            "reward":      ep_reward,
            "td_error":    ep_td / max(steps, 1),
            "epsilon":     agent.epsilon,
            "hops":        info["hops"],
            "delay_ms":    info["total_delay"],
            "dropped":     float(info["dropped"]),
            "avg_util":    info["avg_utilization"],
        })

        if ep % tcfg["save_every"] == 0:
            agent.save(os.path.join(target_dir, f"qtable_ep{ep}.npy"))

    environment.close()
    logger.close()

    # Lưu checkpoint cuối
    agent.save(os.path.join(target_dir, "qtable_final.npy"))

    # In Q-table summary
    print()
    print(agent.q_table_summary())

    # Demo greedy policy
    print("\n=== Greedy policy sau training ===")
    for src, dst in [(0,7), (1,6), (2,5), (3,7), (0,5)]:
        path = agent.best_path(src, dst)
        print(f"  {src} → {dst} : {path}")


if __name__ == "__main__":
    main()