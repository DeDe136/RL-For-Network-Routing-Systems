"""
scripts/evaluate.py

Đánh giá Q-Learning agent đã train.

Chạy:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/qtable_final.npy
    python scripts/evaluate.py --checkpoint checkpoints/qtable_final.npy --episodes 100 --render
"""

import sys, os, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import env  # noqa
import gymnasium as gym

from agents.q_learning_delay import QLearningAgent


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(agent: QLearningAgent, environment, n_episodes: int, render: bool):
    agent.eval_mode()

    rewards, delays, hops_list, drops, utils = [], [], [], [], []

    for ep in range(1, n_episodes + 1):
        obs, _ = environment.reset()
        state  = (obs["current_node"], obs["dst_node"])
        ep_reward = 0.0
        done = False

        if render:
            print(f"\n--- Episode {ep} | {obs['current_node']}→{obs['dst_node']} ---")

        while not done:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = (obs["current_node"], obs["dst_node"])

        rewards.append(ep_reward)
        delays.append(info["total_delay"])
        hops_list.append(info["hops"])
        drops.append(float(info["dropped"]))
        utils.append(info["avg_utilization"])

        if render:
            status = "✓ reached" if not info["dropped"] else "✗ dropped"
            print(f"  path={info['path']}  delay={info['total_delay']:.1f}ms  "
                  f"hops={info['hops']}  {status}  reward={ep_reward:+.3f}")

    # ── Print summary table ──────────────────────────────────────────
    W = 52
    print("\n" + "=" * W)
    print(f"{'Evaluation Results':^{W}}")
    print("=" * W)
    fmt = f"  {{:<22}} {{:>10.4f}}  {{:>10.4f}}"
    header = f"  {'Metric':<22} {'Mean':>10}  {'Std':>10}"
    print(header)
    print("-" * W)
    print(fmt.format("Total reward",      np.mean(rewards),    np.std(rewards)))
    print(fmt.format("Path delay (ms)",   np.mean(delays),     np.std(delays)))
    print(fmt.format("Hops",              np.mean(hops_list),  np.std(hops_list)))
    print(fmt.format("Drop rate",         np.mean(drops),      np.std(drops)))
    print(fmt.format("Avg link util",     np.mean(utils),      np.std(utils)))
    print("=" * W)

    # ── Greedy policy demo ───────────────────────────────────────────
    print("\n  Greedy path per (src, dst) pair:")
    raw = environment.unwrapped
    for src, dst in [(0,7),(1,6),(2,5),(3,7),(4,6),(0,4)]:
        path = agent.best_path(src, dst)
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
    parser.add_argument("--checkpoint",   default="checkpoints/qtable_final.npy")
    parser.add_argument("--env-config",   default="configs/env_config.yaml")
    parser.add_argument("--agent-config", default="configs/agent_config.yaml")
    parser.add_argument("--episodes",     type=int, default=50)
    parser.add_argument("--render",       action="store_true")
    args = parser.parse_args()

    ecfg = load_cfg(args.env_config)
    acfg = load_cfg(args.agent_config)

    environment = gym.make(
        "NetworkRouting-v0",
        mean_traffic_mbps=ecfg["env"]["mean_traffic_mbps"],
        max_hops=ecfg["env"]["max_hops"],
        seed=ecfg["env"]["seed"] + 999,   # seed khác train
    )

    agent = QLearningAgent(config=acfg["q_learning"])
    agent.set_neighbor_mask(environment.unwrapped.topo.adj_matrix)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run 'python scripts/train.py' first.")
        return

    agent.load(args.checkpoint)
    print(f"Evaluating {args.episodes} episodes "
          f"(checkpoint: {args.checkpoint})...\n")
    evaluate(agent, environment, n_episodes=args.episodes, render=args.render)
    environment.close()


if __name__ == "__main__":
    main()