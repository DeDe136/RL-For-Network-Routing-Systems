"""
scripts/DQN/dqn_train.py

Training loop cho DQN Agent trên NetworkRouting-v0.

Chạy:
    python scripts/DQN/dqn_train.py
    python scripts/DQN/dqn_train.py --episodes 5000
"""

import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env  # noqa — kích hoạt gymnasium.register
import gymnasium as gym
import numpy as np

from agents.DQN.dqn_agent import DQNAgent
from utils.logger import Logger


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
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

    environment = gym.make(
        "NetworkRouting-DQN-v0",
        mean_traffic_mbps=ecfg["env"]["mean_traffic_mbps"],
        max_hops=ecfg["env"]["max_hops"],
        bg_intensity=ecfg["env"].get("bg_intensity", 0.3),
        seed=ecfg["env"]["seed"],
    )

    agent = DQNAgent(n_states=1, n_actions=8, config=acfg["dqn"])
    agent.set_neighbor_mask(environment.unwrapped.topo.adj_matrix)

    full_log_dir = os.path.join(tcfg["log_dir"], "DQN")
    os.makedirs(full_log_dir, exist_ok=True)
    logger = Logger(
        log_dir=full_log_dir,
        print_every=tcfg["print_every"],
    )

    print(f"Training DQN | {num_episodes} episodes | device={agent.device}")
    print(f"  {agent.network_summary()}\n")

    for ep in range(1, num_episodes + 1):
        obs, _ = environment.reset()
        ep_reward = 0.0
        ep_loss   = 0.0
        steps     = 0
        done      = False
        agent.train_mode()

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = \
                environment.step(action)
            done = terminated or truncated

            agent.remember(obs, action, reward, next_obs, done)
            result = agent.update()

            ep_reward += reward
            ep_loss   += result.get("loss") or 0.0
            steps     += 1
            obs        = next_obs

        logger.log({
            "reward":   ep_reward,
            "loss":     ep_loss / max(steps, 1),
            "epsilon":  agent.epsilon,
            "hops":     info["hops"],
            "delay_ms": info["total_delay"],
            "dropped":  float(info["dropped"]),
            "avg_util": info["avg_utilization"],
            "buf_size": len(agent.memory),
        })

        if ep % tcfg["save_every"] == 0:
            agent.save(os.path.join(target_dir, f"dqn_ep{ep}.pt"))

    environment.close()
    logger.close()
    agent.save(os.path.join(target_dir, "dqn_final.pt"))

    print(f"\n{agent.network_summary()}")
    print("\n=== Greedy paths sau training ===")
    agent.eval_mode()
    rng = np.random.default_rng(99)
    for src, dst in [(0, 7), (1, 6), (2, 5), (3, 7), (0, 5)]:
        volume = float(max(1.0, rng.poisson(10.0)))
        path = agent.best_path(src, dst, environment.unwrapped.topo, volume)
        print(f"  {src} → {dst} : {path}")

if __name__ == "__main__":
    main()