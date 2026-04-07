"""
main.py — Entry point tổng hợp.

Chạy:
    python main.py                        # env check + demo
    python main.py --mode train
    python main.py --mode eval
    python main.py --mode demo            # greedy demo với Q-table đã train
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import env  # noqa — đăng ký NetworkRouting-v0
import gymnasium as gym
import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  Env check
# ─────────────────────────────────────────────────────────────────────

def run_env_check():
    from gymnasium.utils.env_checker import check_env
    from env.routing_env import NetworkRoutingEnv

    print("=" * 50)
    print("  Gymnasium env_checker")
    print("=" * 50)
    e = NetworkRoutingEnv(max_hops=8)
    check_env(e, warn=True)
    print("✓ check_env passed\n")

    print("  Random rollout (1 episode, render=human)")
    print("-" * 50)
    e2 = NetworkRoutingEnv(max_hops=8, render_mode="human")
    obs, _ = e2.reset()
    print(f"  Demand: {obs['current_node']} → {obs['dst_node']}")
    total, done = 0.0, False
    while not done:
        action = e2.action_space.sample()
        obs, r, terminated, truncated, info = e2.step(action)
        total += r
        done = terminated or truncated
    print(f"\n  Final path : {info['path']}")
    print(f"  Total delay: {info['total_delay']:.1f} ms")
    print(f"  Dropped    : {info['dropped']}")
    print(f"  Total reward: {total:+.3f}")
    e2.close()


# ─────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────

def run_train(episodes: int = None):
    from scripts.train import main as train_main
    argv_backup = sys.argv[:]
    sys.argv = ["train.py"]
    if episodes:
        sys.argv += ["--episodes", str(episodes)]
    train_main()
    sys.argv = argv_backup


# ─────────────────────────────────────────────────────────────────────
#  Eval
# ─────────────────────────────────────────────────────────────────────

def run_eval(render: bool = False):
    from scripts.evaluate import main as eval_main
    argv_backup = sys.argv[:]
    sys.argv = ["evaluate.py"]
    if render:
        sys.argv.append("--render")
    eval_main()
    sys.argv = argv_backup


# ─────────────────────────────────────────────────────────────────────
#  Demo: greedy path hiển thị Q-values
# ─────────────────────────────────────────────────────────────────────

def run_demo():
    from agents.q_learning import QLearningAgent
    from network.topology import NetworkTopology

    ckpt = "checkpoints/qtable_final.npy"
    if not os.path.exists(ckpt):
        print(f"Checkpoint không tìm thấy: {ckpt}")
        print("Chạy 'python main.py --mode train' trước.")
        return

    topo  = NetworkTopology()
    agent = QLearningAgent()
    agent.set_neighbor_mask(topo.adj_matrix)
    agent.load(ckpt)
    agent.eval_mode()

    print("\n" + "=" * 55)
    print("  Q-Learning Greedy Routing Demo")
    print("=" * 55)
    print(agent.q_table_summary())
    print()

    pairs = [(0,7),(1,7),(2,7),(3,7),(0,5),(0,6),(1,4),(2,5)]
    sp_topo = NetworkTopology()

    print(f"  {'Pair':<8} {'QL Path':<30} {'QL delay':>9}  {'SP Path':<30} {'SP delay':>9}")
    print("  " + "-" * 90)

    pair_labels = []
    ql_delays = []
    sp_delays = []

    for src, dst in pairs:
        ql_path = agent.best_path(src, dst)
        sp_path = sp_topo.shortest_path(src, dst)

        ql_delay = sum(topo.link(ql_path[i], ql_path[i+1]).delay
                       for i in range(len(ql_path)-1)) if len(ql_path)>1 else 0
        sp_delay = sum(topo.link(sp_path[i], sp_path[i+1]).delay
                       for i in range(len(sp_path)-1)) if len(sp_path)>1 else 0

        pair_labels.append(f"{src}-{dst}")
        ql_delays.append(ql_delay)
        sp_delays.append(sp_delay)

        ql_str = " → ".join(map(str, ql_path))
        sp_str = " → ".join(map(str, sp_path))
        match  = "✓" if ql_path == sp_path else "≠"
        print(f"  {src}→{dst}  {match}  {ql_str:<28} {ql_delay:>6.0f}ms"
              f"   {sp_str:<28} {sp_delay:>6.0f}ms")

    try:
        import matplotlib.pyplot as plt

        x = np.arange(len(pair_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        rects1 = ax.bar(x - width/2, ql_delays, width, label='Q-learning')
        rects2 = ax.bar(x + width/2, sp_delays, width, label='OSPF')

        ax.set_xlabel('Node pair')
        ax.set_ylabel('Total delay (ms)')
        ax.set_title('So sánh độ trễ Q-learning vs OSPF theo cặp nguồn-đích')
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print('\nMatplotlib chưa được cài đặt, không thể hiển thị biểu đồ.')
        print('Cài đặt bằng: pip install matplotlib')


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "train", "eval", "demo"],
                        default="check")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--render",   action="store_true")
    args = parser.parse_args()

    if args.mode == "check":
        run_env_check()
    elif args.mode == "train":
        run_train(args.episodes)
    elif args.mode == "eval":
        run_eval(args.render)
    elif args.mode == "demo":
        run_demo()