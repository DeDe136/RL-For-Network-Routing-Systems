"""
main.py — Entry point cho DQN routing project.

Chạy:
    python main_dqn.py                    # kiểm tra env
    python main_dqn.py --mode train
    python main_dqn.py --mode eval --checkpoint checkpoints/DQN/dqn_final.pt
    python main_dqn.py --mode demo --checkpoint checkpoints/DQN/dqn_final.pt
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import env  # noqa
import gymnasium as gym


def run_env_check():
    from gymnasium.utils.env_checker import check_env
    from env.DQN.routing_env import NetworkRoutingEnv

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
        # Chỉ sample trong valid neighbors
        import numpy as np
        valid = e2.topo.neighbors(e2._current_node)
        action = int(np.random.choice(valid)) if valid else e2.action_space.sample()
        obs, r, terminated, truncated, info = e2.step(action)
        total += r
        done = terminated or truncated
    print(f"\n  Final path  : {info['path']}")
    print(f"  Total delay : {info['total_delay']:.1f} ms")
    print(f"  Dropped     : {info['dropped']}")
    print(f"  Total reward: {total:+.3f}")
    e2.close()


def run_train(episodes: int = None):
    from scripts.DQN.train import main as train_main
    argv_backup = sys.argv[:]
    sys.argv = ["train.py"]
    if episodes:
        sys.argv += ["--episodes", str(episodes)]
    train_main()
    sys.argv = argv_backup


def run_eval(checkpoint: str, render: bool = False):
    from scripts.DQN.evaluate import main as eval_main
    argv_backup = sys.argv[:]
    sys.argv = ["evaluate.py", "--checkpoint", checkpoint]
    if render:
        sys.argv.append("--render")
    eval_main()
    sys.argv = argv_backup


def run_demo(checkpoint: str):
    import yaml, numpy as np
    from agents.DQN.dqn_agent import DQNAgent
    from network.DQN.topology import NetworkTopology

    ckpt = checkpoint
    if not os.path.exists(ckpt):
        print(f"Checkpoint không tìm thấy: {ckpt}")
        print("Chạy 'python main_ql.py --mode train' trước.")
        return
    
    with open("configs/agent_config.yaml") as f:
        acfg = yaml.safe_load(f)

    topo  = NetworkTopology(seed=0)
    agent = DQNAgent(n_states=1, n_actions=8, config=acfg["dqn"])
    agent.set_neighbor_mask(topo.adj_matrix)
    agent.load(ckpt)
    agent.eval_mode()

    print("\n" + "=" * 58)
    print("  DQN Greedy Routing Demo")
    print("=" * 58)
    print(f"  {agent.network_summary()}\n")

    pairs = [(0,7),(1,7),(2,7),(3,7),(0,5),(0,6),(1,4),(2,5)]
    print(f"  {'Pair':<8} {'DQN Path':<35} {'SP Path':<35}")
    print("  " + "-" * 78)

    rng = np.random.default_rng(99)
    for src, dst in pairs:
        volume = float(max(1.0, rng.poisson(10.0)))
        dqn_path = agent.best_path(src, dst, topo, volume)
        sp_path  = topo.shortest_path(src, dst)
        match    = "✓" if dqn_path == sp_path else "≠"
        dqn_str  = " → ".join(map(str, dqn_path))
        sp_str   = " → ".join(map(str, sp_path))
        print(f"  {src}→{dst}  {match}  {dqn_str:<33}  {sp_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["check","train","eval","demo"],
                        default="check")
    parser.add_argument("--episodes",   type=int, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/DQN/dqn_final.pt")
    parser.add_argument("--render",     action="store_true")
    args = parser.parse_args()

    if args.mode == "check":
        run_env_check()
    elif args.mode == "train":
        run_train(args.episodes)
    elif args.mode == "eval":
        run_eval(args.checkpoint, args.render)
    elif args.mode == "demo":
        run_demo(args.checkpoint)