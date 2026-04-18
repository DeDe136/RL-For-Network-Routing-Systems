"""
main.py — Entry point cho DQN routing project.

Chạy:
    python main_dqn.py                    # kiểm tra env
    python main_dqn.py --mode train
    python main_dqn.py --mode eval  --checkpoint checkpoints/DQN/dqn_final.pt
    python main_dqn.py --mode demo  --checkpoint checkpoints/DQN/dqn_final.pt

Output logs (tất cả trong logs/DQN/):
    train_episodes.csv  — mỗi episode training,    cột algo="DQN"
    eval_episodes.csv   — mỗi episode evaluation,  cột algo="DQN"
    demo_episodes.csv   — DQN + OSPF xen kẽ,       cột algo phân biệt
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import env  # noqa
import gymnasium as gym


# ─────────────────────────────────────────────────────────────────────
#  Env check
# ─────────────────────────────────────────────────────────────────────

def run_env_check():
    from gymnasium.utils.env_checker import check_env
    from env.DQN.routing_env import NetworkRoutingEnv
    import numpy as np

    print("=" * 52)
    print("  Gymnasium env_checker")
    print("=" * 52)
    e = NetworkRoutingEnv(max_hops=8)
    check_env(e, warn=True)
    print("✓ check_env passed\n")

    print("  Random rollout (1 episode, render=human)")
    print("-" * 52)
    e2 = NetworkRoutingEnv(max_hops=8, render_mode="human")
    obs, _ = e2.reset()
    print(f"  Demand: {obs['current_node']} → {obs['dst_node']}")
    total, done = 0.0, False
    while not done:
        valid  = e2.topo.neighbors(e2._current_node)
        action = int(np.random.choice(valid)) if valid else e2.action_space.sample()
        obs, r, terminated, truncated, info = e2.step(action)
        total += r
        done   = terminated or truncated
    print(f"\n  Final path  : {info['path']}")
    print(f"  Total delay : {info['total_delay']:.2f} ms")
    print(f"  Dropped     : {info['dropped']}")
    print(f"  Total reward: {total:+.3f}")
    e2.close()


# ─────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────

def run_train(episodes: int = None):
    from scripts.DQN.train import main as train_main
    argv_backup = sys.argv[:]
    sys.argv = ["dqn_train.py"]
    if episodes:
        sys.argv += ["--episodes", str(episodes)]
    train_main()
    sys.argv = argv_backup


# ─────────────────────────────────────────────────────────────────────
#  Eval
# ─────────────────────────────────────────────────────────────────────

def run_eval(checkpoint: str, render: bool = False):
    from scripts.DQN.evaluate import main as eval_main
    argv_backup = sys.argv[:]
    sys.argv = ["evaluate.py", "--checkpoint", checkpoint]
    if render:
        sys.argv.append("--render")
    eval_main()
    sys.argv = argv_backup


# ─────────────────────────────────────────────────────────────────────
#  Demo — DQN vs OSPF, ghi log cả hai vào demo_episodes.csv
# ─────────────────────────────────────────────────────────────────────

def run_demo(checkpoint: str):
    """
    So sánh DQN và OSPF (Shortest Path theo delay) trên topology độc lập.

    Thiết kế công bằng:
      - topo_dqn  và topo_ospf dùng CÙNG seed → cùng tham số link vật lý
        (delay, bandwidth, queue_size_cur) sau randomize_links().
      - Hai vòng lặp độc lập — DQN xong rồi mới OSPF.
      - Cả hai đều gọi send_traffic() + reduce_load() từng hop để
        trạng thái mạng (utilization, queue_util) thay đổi thực tế.

    Output demo_episodes.csv:
      - Tất cả episode DQN ghi trước (algo="DQN"), ep_idx = 1..N
      - Tất cả episode OSPF ghi sau  (algo="OSPF"), ep_idx = N+1..2N
    """
    import yaml, numpy as np
    from agents.DQN.dqn_agent import DQNAgent
    from network.DQN.topology import NetworkTopology
    from utils.episode_logger import EpisodeLogger

    if not os.path.exists(checkpoint):
        print(f"Checkpoint không tìm thấy: {checkpoint}")
        print("Chạy 'python main.py --mode train' trước.")
        return

    with open("configs/agent_config.yaml", encoding="utf-8") as f:
        acfg = yaml.safe_load(f)
    with open("configs/env_config.yaml", encoding="utf-8") as f:
        ecfg = yaml.safe_load(f)

    max_hops = ecfg["env"]["max_hops"]
    volume   = ecfg["env"].get("mean_traffic_mbps", 10.0)
    log_dir  = os.path.join(acfg["training"]["log_dir"], "DQN")
    os.makedirs(log_dir, exist_ok=True)

    # ── Tạo 2 topology độc lập cùng seed ─────────────────────────────
    # Sau randomize_links() chúng có cùng delay/bandwidth/queue_size,
    # nhưng trạng thái traffic (load, queue_used_cur) hoàn toàn tách biệt.
    DEMO_SEED = 0
    topo_dqn  = NetworkTopology(seed=DEMO_SEED)
    topo_ospf = NetworkTopology(seed=DEMO_SEED)

    agent = DQNAgent(n_states=1, n_actions=8, config=acfg["dqn"])
    agent.set_neighbor_mask(topo_dqn.adj_matrix)
    agent.load(checkpoint)
    agent.eval_mode()

    ep_logger = EpisodeLogger(
        log_dir  = log_dir,
        filename = "demo_episodes.csv",
        max_hops = max_hops,
    )

    print("\n" + "=" * 72)
    print("  DQN vs OSPF Demo  (2 topologies độc lập, cùng seed)")
    print("=" * 72)
    print(f"  {agent.network_summary()}")
    print(f"  volume={volume:.1f} Mbps | seed={DEMO_SEED}\n")

    pairs = [(0,7),(1,7),(2,7),(3,7),(0,5),(0,6),(1,4),(2,5)]

    # ── Helpers ──────────────────────────────────────────────────────

    def collect_link_details(path, topo_ref):
        """Đọc trạng thái link SAU khi send_traffic+reduce_load đã chạy."""
        details = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            lk   = topo_ref.link(u, v)
            details.append({
                "src_node":       u,
                "dst_node":       v,
                "delay":          lk.delay,
                "bandwidth":      lk.bandwidth,
                "load":           lk.load,
                "utilization":    lk.utilization,
                "queue_size_cur": lk.queue_size_cur,
                "queue_used_cur": lk.queue_used_cur,
                "queue_util":     lk.queue_util,
            })
        return details

    def avg_metric(details, key):
        if not details:
            return 0.0
        return float(np.mean([d[key] for d in details]))

    def make_info(src_n, dst_n, path, details, topo_ref):
        return {
            "src":             src_n,
            "dst":             dst_n,
            "path":            path,
            "hops":            len(path) - 1,
            "total_delay":     sum(d["delay"] for d in details),
            "dropped":         False,
            "avg_utilization": avg_metric(details, "utilization"),
            "avg_queue_util":  avg_metric(details, "queue_util"),
            "link_details":    details,
            **topo_ref.summary(),
        }

    def ospf_path_with_traffic(src_n, dst_n, topo_ref, vol):
        """
        OSPF: tìm shortest path (Dijkstra) rồi mô phỏng traffic
        từng hop — send_traffic + reduce_load giống best_path DQN.

        Trả về (path, per_hop_details) trong đó per_hop_details là
        danh sách trạng thái link được đọc SAU send_traffic và
        reduce_load (giống best_path() của DQN).
        """
        full_path = topo_ref.shortest_path(src_n, dst_n)
        if len(full_path) < 2:
            return full_path, []

        for hop_idx in range(1, len(full_path)):
            partial      = full_path[: hop_idx + 1]
            is_first_hop = (hop_idx == 1)

            topo_ref.send_traffic(
                path         = partial,
                volume_mbps  = vol,
                is_first_hop = is_first_hop,
            )

            topo_ref.reduce_load(partial)

        # Trạng thái link mỗi hop, đọc sau send_traffic và reduce_load
        per_hop_details = collect_link_details(full_path, topo_ref)

        return full_path, per_hop_details

    # ═══════════════════════════════════════════════════════════════════
    #  Vòng 1 — DQN
    # ═══════════════════════════════════════════════════════════════════
    print(f"  {'─'*80}")
    print(f"  {'Algo':<5} {'Pair':<7} {'Path':<32} {'Delay':>8}  "
          f"{'AvgUtil':>8}  {'AvgQUtil':>9}")
    print(f"  {'─'*80}")
    print("  [DQN]")

    dqn_results = []   # lưu để so sánh + biểu đồ

    for ep_idx, (src_n, dst_n) in enumerate(pairs, start=1):
        # Reset topology DQN với cùng seed → cùng tham số vật lý
        topo_dqn.reset()
        topo_dqn.step_background(intensity=0.2)

        # best_path gọi send_traffic + reduce_load mỗi hop bên trong
        dqn_path    = agent.best_path(src_n, dst_n, topo_dqn,
                                      volume_mbps=volume)
        dqn_details = collect_link_details(dqn_path, topo_dqn)
        dqn_info    = make_info(src_n, dst_n, dqn_path, dqn_details, topo_dqn)

        ep_logger.log_episode(
            algo    = "DQN",
            episode = ep_idx,
            info    = dqn_info,
        )
        dqn_results.append(dqn_info)

        dqn_str = "→".join(map(str, dqn_path))
        print(f"  {'DQN':<5} {src_n}→{dst_n}   "
              f"{dqn_str:<32} "
              f"{dqn_info['total_delay']:>6.2f}ms  "
              f"{dqn_info['avg_utilization']:>8.4f}  "
              f"{dqn_info['avg_queue_util']:>9.4f}")

    # ═══════════════════════════════════════════════════════════════════
    #  Vòng 2 — OSPF (topology riêng, send_traffic + reduce_load từng hop)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("  [OSPF]")

    ospf_results = []
    n_pairs      = len(pairs)

    for ep_idx, (src_n, dst_n) in enumerate(pairs, start=n_pairs + 1):
        # Reset topology OSPF với cùng seed → cùng tham số vật lý như DQN
        topo_ospf.reset()
        topo_ospf.step_background(intensity=0.2)

        # ospf_path_with_traffic: Dijkstra + send_traffic + reduce_load từng hop
        # sp_details đọc sau send_traffic, trước reduce_load (trạng thái đang tải)
        sp_path, sp_details = ospf_path_with_traffic(src_n, dst_n, topo_ospf, volume)
        sp_info = make_info(src_n, dst_n, sp_path, sp_details, topo_ospf)

        ep_logger.log_episode(
            algo    = "OSPF",
            episode = ep_idx,
            info    = sp_info,
        )
        ospf_results.append(sp_info)

        sp_str  = "→".join(map(str, sp_path))
        dqn_path_for_pair = dqn_results[ep_idx - n_pairs - 1]["path"]
        match   = "✓" if sp_path == dqn_path_for_pair else "≠"
        print(f"  {'OSPF':<5} {src_n}→{dst_n}   "
              f"{sp_str:<32} "
              f"{sp_info['total_delay']:>6.2f}ms  "
              f"{sp_info['avg_utilization']:>8.4f}  "
              f"{sp_info['avg_queue_util']:>9.4f}  {match}")

    ep_logger.close()

    # ── Summary so sánh ──────────────────────────────────────────────
    print()
    print(f"  {'─'*70}")
    dqn_avg_util  = float(np.mean([r["avg_utilization"] for r in dqn_results]))
    dqn_avg_qu    = float(np.mean([r["avg_queue_util"]  for r in dqn_results]))
    dqn_avg_delay = float(np.mean([r["total_delay"]     for r in dqn_results]))
    sp_avg_util   = float(np.mean([r["avg_utilization"] for r in ospf_results]))
    sp_avg_qu     = float(np.mean([r["avg_queue_util"]  for r in ospf_results]))
    sp_avg_delay  = float(np.mean([r["total_delay"]     for r in ospf_results]))

    print(f"  {'':5} {'':7} {'Avg delay':>32}  {'AvgUtil':>8}  {'AvgQUtil':>9}")
    print(f"  {'DQN':<5} {'ALL':<7} {dqn_avg_delay:>32.2f}ms  "
          f"{dqn_avg_util:>8.4f}  {dqn_avg_qu:>9.4f}")
    print(f"  {'OSPF':<5} {'ALL':<7} {sp_avg_delay:>32.2f}ms  "
          f"{sp_avg_util:>8.4f}  {sp_avg_qu:>9.4f}")
    print(f"  {'─'*70}")
    print(f"  Demo log: {ep_logger.filepath}")

    # ── Biểu đồ (nếu matplotlib có sẵn) ─────────────────────────────
    try:
        import matplotlib.pyplot as plt

        pair_labels = [f"{s}→{d}" for s, d in pairs]
        dqn_delays  = [r["total_delay"] for r in dqn_results]
        sp_delays   = [r["total_delay"] for r in ospf_results]

        x     = np.arange(len(pair_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 5))
        bars1 = ax.bar(x - width/2, dqn_delays, width, label="DQN",  color="#4C72B0")
        bars2 = ax.bar(x + width/2, sp_delays,  width, label="OSPF", color="#DD8452")

        ax.set_xlabel("Node pair")
        ax.set_ylabel("Total delay (ms)")
        ax.set_title("So sánh độ trễ DQN vs OSPF (topology độc lập, cùng seed)")
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for rect in list(bars1) + list(bars2):
            h = rect.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        chart_path = os.path.join(log_dir, "demo_delay_comparison.png")
        plt.savefig(chart_path, dpi=150)
        print(f"  Chart saved: {chart_path}")
        plt.show()
    except ImportError:
        print("\n  (matplotlib chưa cài — bỏ qua biểu đồ. pip install matplotlib)")


# ─────────────────────────────────────────────────────────────────────

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