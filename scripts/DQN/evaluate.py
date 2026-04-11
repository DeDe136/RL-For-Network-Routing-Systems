"""
scripts/evaluate.py

Đánh giá DQN agent đã train trên NetworkRouting-v0.

Chạy:
    python scripts/DQN/evaluate.py --checkpoint checkpoints/DQN/dqn_final.pt
    python scripts/DQN/evaluate.py --checkpoint checkpoints/DQN/dqn_final.pt --render
    python scripts/DQN/evaluate.py --checkpoint checkpoints/DQN/dqn_final.pt --episodes 100
"""

import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import env  # noqa
import gymnasium as gym

from agents.DQN.dqn_agent import DQNAgent


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_evaluate(agent: DQNAgent, environment, n_episodes: int,
                 render: bool) -> dict:
    agent.eval_mode()
    rewards, delays, hops_list, drops, utils = [], [], [], [], []

    for ep in range(1, n_episodes + 1):
        obs, _ = environment.reset()
        ep_reward = 0.0
        done      = False

        if render:
            print(f"\n--- Episode {ep} | "
                  f"{obs['current_node']}→{obs['dst_node']} ---")

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = \
                environment.step(action)
            done = terminated or truncated
            ep_reward += reward

        rewards.append(ep_reward)
        delays.append(info["total_delay"])
        hops_list.append(info["hops"])
        drops.append(float(info["dropped"]))
        utils.append(info["avg_utilization"])

        if render:
            status = "✓ reached" if not info["dropped"] else "✗ dropped"
            print(f"  path={info['path']}  "
                  f"delay={info['total_delay']:.1f}ms  "
                  f"hops={info['hops']}  {status}  "
                  f"r={ep_reward:+.3f}")

    W = 52
    print("\n" + "=" * W)
    print(f"{'Evaluation Results':^{W}}")
    print("=" * W)
    fmt = f"  {{:<22}} {{:>10.4f}}  {{:>10.4f}}"
    print(f"  {'Metric':<22} {'Mean':>10}  {'Std':>10}")
    print("-" * W)
    print(fmt.format("Total reward",    np.mean(rewards),   np.std(rewards)))
    print(fmt.format("Path delay (ms)", np.mean(delays),    np.std(delays)))
    print(fmt.format("Hops",            np.mean(hops_list), np.std(hops_list)))
    print(fmt.format("Drop rate",       np.mean(drops),     np.std(drops)))
    print(fmt.format("Avg link util",   np.mean(utils),     np.std(utils)))
    print("=" * W)

    raw = environment.unwrapped
    print("\n  Greedy paths (eval mode):")
    for src, dst in [(0, 7), (1, 6), (2, 5), (3, 7), (4, 6), (0, 4)]:
        path = agent.best_path(src, dst, raw.topo)
        print(f"    {src} → {dst} : {path}")

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_delay":  float(np.mean(delays)),
        "mean_hops":   float(np.mean(hops_list)),
        "drop_rate":   float(np.mean(drops)),
        "mean_util":   float(np.mean(utils)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/DQN/dqn_final.npy")
    parser.add_argument("--env-config",   default="configs/env_config.yaml")
    parser.add_argument("--agent-config", default="configs/agent_config.yaml")
    parser.add_argument("--episodes",     type=int, default=50)
    parser.add_argument("--render",       action="store_true")
    args = parser.parse_args()

    ecfg = load_cfg(args.env_config)
    acfg = load_cfg(args.agent_config)

    environment = gym.make(
        "NetworkRouting-DQN-v0",
        mean_traffic_mbps=ecfg["env"]["mean_traffic_mbps"],
        max_hops=ecfg["env"]["max_hops"],
        bg_intensity=ecfg["env"].get("bg_intensity", 0.3),
        seed=ecfg["env"]["seed"] + 999,   # seed khác với lúc train
    )

    agent = DQNAgent(n_states=1, n_actions=8, config=acfg["dqn"])
    agent.set_neighbor_mask(environment.unwrapped.topo.adj_matrix)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run 'python scripts/DQN/train.py' first.")
        return

    agent.load(args.checkpoint)
    print(f"Evaluating DQN | {args.episodes} episodes | "
          f"checkpoint: {args.checkpoint}\n")
    run_evaluate(agent, environment, args.episodes, args.render)
    environment.close()


if __name__ == "__main__":
    main()