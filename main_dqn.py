"""
main_dqn.py — Entry point cho DQN routing project.

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

def run_demo(checkpoint: str, n_pairs: int = 8):
    """
    So sánh DQN và OSPF trên n_pairs bộ (src, dst, volume) ngẫu nhiên.

    Quy trình:
      1. Sinh n_pairs bộ (src, dst, volume) ngẫu nhiên TRƯỚC.
      2. Vòng 1 — DQN: topo_dqn reset cùng seed cho mỗi bộ,
         gọi best_path (có send_traffic + reduce_load từng hop).
      3. Vòng 2 — OSPF: topo_ospf reset cùng seed (giống DQN) cho mỗi bộ,
         gọi ospf_path_with_traffic (Dijkstra + send_traffic + reduce_load).
      4. In 5 biểu đồ so sánh (nếu matplotlib có sẵn).

    Công bằng:
      - topo_dqn và topo_ospf cùng DEMO_SEED → cùng params vật lý.
      - Mỗi bộ reset cùng seed con để hai thuật toán bắt đầu từ
        trạng thái mạng giống hệt nhau.
      - Cả hai đều mô phỏng traffic thực tế qua từng hop.
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

    max_hops  = ecfg["env"]["max_hops"]
    mean_vol  = ecfg["env"].get("mean_traffic_mbps", 10.0)
    log_dir   = os.path.join(acfg["training"]["log_dir"], "DQN")
    os.makedirs(log_dir, exist_ok=True)

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

    # ── Sinh n_pairs bộ (src, dst, volume) ngẫu nhiên ────────────────
    rng_pairs = np.random.default_rng(DEMO_SEED + 1)
    pairs = []
    while len(pairs) < n_pairs:
        src = int(rng_pairs.integers(0, 8))
        dst = int(rng_pairs.integers(0, 8))
        if src != dst and topo_dqn.shortest_path(src, dst):
            vol = float(max(1.0, rng_pairs.poisson(mean_vol)))
            pairs.append((src, dst, vol))

    print("\n" + "=" * 72)
    print(f"  DQN vs OSPF Demo  ({n_pairs} bộ ngẫu nhiên, seed={DEMO_SEED})")
    print("=" * 72)
    print(f"  {agent.network_summary()}\n")
    print(f"  Các bộ (src, dst, volume):")
    for i,(s,d,v) in enumerate(pairs):
        print(f"    [{i+1:2d}] {s}→{d}  vol={v:.1f} Mbps")
    print()

    # ── Helpers ──────────────────────────────────────────────────────

    def collect_link_details(path, topo_ref):
        """Đọc trạng thái link hiện tại trên path."""
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
                "dropped_data":   lk.dropped_data,
            })
        return details

    def avg_metric(details, key):
        return float(np.mean([d[key] for d in details])) if details else 0.0

    def make_info(src_n, dst_n, path, details, topo_ref):
        return {
            "src":               src_n,
            "dst":               dst_n,
            "path":              path,
            "hops":              len(path) - 1,
            "total_delay":       sum(d["delay"]        for d in details),
            "dropped":           any(d["dropped_data"] > 0 for d in details),
            "avg_utilization":   avg_metric(details, "utilization"),
            "avg_queue_util":    avg_metric(details, "queue_util"),
            "avg_bandwidth":     avg_metric(details, "bandwidth"),
            "avg_load":          avg_metric(details, "load"),
            "avg_queue_used_cur": avg_metric(details, "queue_used_cur"),
            "total_dropped_data":sum(d["dropped_data"] for d in details),
            "link_details":      details,
            **topo_ref.summary(),
        }

    def ospf_path_with_traffic(src_n, dst_n, topo_ref, vol):
        """Dijkstra + send_traffic + reduce_load từng hop."""
        full_path = topo_ref.shortest_path(src_n, dst_n)
        if len(full_path) < 2:
            return full_path
        for hop_idx in range(1, len(full_path)):
            partial = full_path[:hop_idx + 1]
            topo_ref.send_traffic(partial, vol, is_first_hop=(hop_idx == 1))
            topo_ref.reduce_load(partial)
        return full_path

    # ── Seed con cho mỗi bộ (đảm bảo DQN và OSPF bắt đầu giống nhau) ─
    pair_seeds = [DEMO_SEED + 100 + i for i in range(n_pairs)]

    # ═══════════════════════════════════════════════════════════════════
    #  Vòng 1 — DQN
    # ═══════════════════════════════════════════════════════════════════
    print(f"  {'─'*72}")
    print(f"  {'Algo':<5} {'#':<3} {'Pair':<6} {'Vol':>5}  "
          f"{'Path':<28} {'Delay':>7}  {'AvgUtil':>7}  {'AvgQUtil':>8}")
    print(f"  {'─'*72}")
    print("  [DQN]")

    dqn_results = []

    for ep_idx, ((src_n, dst_n, vol), seed) in enumerate(
            zip(pairs, pair_seeds), start=1):
        # Reset với seed con → cùng trạng thái mạng ban đầu như OSPF
        topo_dqn.reset()
        topo_dqn._rng = np.random.default_rng(seed)
        topo_dqn.step_background(intensity=0.3)

        dqn_path    = agent.best_path(src_n, dst_n, topo_dqn, volume_mbps=vol)
        dqn_details = collect_link_details(dqn_path, topo_dqn)
        dqn_info    = make_info(src_n, dst_n, dqn_path, dqn_details, topo_dqn)

        ep_logger.log_episode("DQN", ep_idx, dqn_info, reward=None)
        dqn_results.append(dqn_info)

        dqn_str = "→".join(map(str, dqn_path))
        print(f"  {'DQN':<5} {ep_idx:<3} {src_n}→{dst_n}  {vol:>5.1f}  "
              f"{dqn_str:<28} "
              f"{dqn_info['total_delay']:>5.2f}ms  "
              f"{dqn_info['avg_utilization']:>7.4f}  "
              f"{dqn_info['avg_queue_util']:>8.4f}")

    # ═══════════════════════════════════════════════════════════════════
    #  Vòng 2 — OSPF
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("  [OSPF]")

    ospf_results = []
    n = len(pairs)

    for ep_idx, ((src_n, dst_n, vol), seed) in enumerate(
            zip(pairs, pair_seeds), start=n + 1):
        # Cùng seed con → cùng trạng thái mạng ban đầu như DQN
        topo_ospf.reset()
        topo_ospf._rng = np.random.default_rng(seed)
        topo_ospf.step_background(intensity=0.3)

        sp_path    = ospf_path_with_traffic(src_n, dst_n, topo_ospf, vol)
        sp_details = collect_link_details(sp_path, topo_ospf)
        sp_info    = make_info(src_n, dst_n, sp_path, sp_details, topo_ospf)

        ep_logger.log_episode("OSPF", ep_idx, sp_info, reward=None)
        ospf_results.append(sp_info)

        sp_str  = "→".join(map(str, sp_path))
        dqn_ref = dqn_results[ep_idx - n - 1]
        match   = "✓" if sp_path == dqn_ref["path"] else "≠"
        print(f"  {'OSPF':<5} {ep_idx-n:<3} {src_n}→{dst_n}  {vol:>5.1f}  "
              f"{sp_str:<28} "
              f"{sp_info['total_delay']:>5.2f}ms  "
              f"{sp_info['avg_utilization']:>7.4f}  "
              f"{sp_info['avg_queue_util']:>8.4f}  {match}")

    ep_logger.close()

    # ── Summary tổng hợp ─────────────────────────────────────────────
    print()
    print(f"  {'─'*72}")

    def gmean(results, key):
        return float(np.mean([r[key] for r in results]))

    metrics = [
        ("Avg delay (ms)",        "total_delay"),
        ("Avg utilization",       "avg_utilization"),
        ("Avg queue_util",        "avg_queue_util"),
        ("Avg dropped_data (pkt)","total_dropped_data"),
    ]
    print(f"  {'Metric':<28} {'DQN':>12}  {'OSPF':>12}")
    print(f"  {'─'*56}")
    for label, key in metrics:
        dv = gmean(dqn_results,  key)
        sv = gmean(ospf_results, key)
        print(f"  {label:<28} {dv:>12.4f}  {sv:>12.4f}")
    print(f"  {'─'*56}")
    print(f"  Demo log: {ep_logger.filepath}")

    # ── Biểu đồ ──────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        pair_labels = [f"{s}→{d}" for s,d,_ in pairs]
        x     = np.arange(n_pairs)
        width = 0.35

        def vals(results, key):
            return [r[key] for r in results]

        color_dqn  = "#2E86AB"
        color_ospf = "#F18F01"

        # ── Helper vẽ từng biểu đồ riêng biệt ───────────────────────
        def create_bar_chart(dqn_vals, ospf_vals, title, ylabel, suffix):
            fig, ax = plt.subplots(figsize=(11, 7))
            b1 = ax.bar(x - width/2, dqn_vals,  width, label="DQN",  color=color_dqn)
            b2 = ax.bar(x + width/2, ospf_vals, width, label="OSPF", color=color_ospf)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels, rotation=30, ha="right")
            ax.legend(fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            for rect in list(b1) + list(b2):
                h = rect.get_height()
                if h > 0:
                    ax.annotate(f"{h:.2f}",
                                xy=(rect.get_x()+rect.get_width()/2, h),
                                xytext=(0,2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7)

            fig.suptitle(f"DQN vs OSPF — {n_pairs} bộ ngẫu nhiên (seed={DEMO_SEED})",
                         fontsize=13, fontweight="bold", y=0.98)

            chart_path = os.path.join(log_dir, f"demo_comparison_{suffix}.png")
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            print(f"  Chart saved: {chart_path}")
            plt.close(fig)   # Đóng figure để không chiếm bộ nhớ

        # ═══════════════════════════════════════════════════════════════
        #  5 BIỂU ĐỒ RIÊNG BIỆT
        # ═══════════════════════════════════════════════════════════════

        # Biểu đồ 1: Delay
        create_bar_chart(
            vals(dqn_results,  "total_delay"),
            vals(ospf_results, "total_delay"),
            "1. So sánh delay trung bình từng bộ",
            "Delay (ms)",
            "1_delay"
        )

        # Biểu đồ 2: Avg load
        create_bar_chart(
            vals(dqn_results,  "avg_load"),
            vals(ospf_results, "avg_load"),
            "2. So sánh băng thông sử dụng TB từng bộ",
            "Avg load (Mbps)",
            "2_load"
        )

        # Biểu đồ 3: Avg queue_util
        create_bar_chart(
            vals(dqn_results,  "avg_queue_used_cur"),
            vals(ospf_results, "avg_queue_used_cur"),
            "3. So sánh packets đang được sử dụng TB trên hàng đợi từng bộ",
            "Avg queue_used_cur (packets)",
            "3_queue"
        )

        # Biểu đồ 4: Total dropped_data
        create_bar_chart(
            vals(dqn_results,  "total_dropped_data"),
            vals(ospf_results, "total_dropped_data"),
            "4. So sánh packets bị drop từng bộ",
            "Dropped data (packets)",
            "4_dropped"
        )

        # Biểu đồ 5: Tổng hợp tất cả bộ (summary)
        fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(11, 10))

        metrics_summary = [
            ("Delay TB (ms)",        "total_delay"),
            ("Load TB (Mbps)",       "avg_load"),
            ("Queue util TB",        "avg_queue_util"),
            ("Dropped TB (pkts)",    "total_dropped_data"),
        ]

        m_labels = [m[0] for m in metrics_summary]
        dqn_agg  = [gmean(dqn_results,  m[1]) for m in metrics_summary]
        ospf_agg = [gmean(ospf_results, m[1]) for m in metrics_summary]

        xm = np.arange(len(m_labels))
        width = 0.35

        # ─────────────────────────────────────────
        #  Biểu đồ trên: giá trị tuyệt đối
        # ─────────────────────────────────────────
        b1 = ax5.bar(xm - width/2, dqn_agg,  width, label="DQN",  color=color_dqn)
        b2 = ax5.bar(xm + width/2, ospf_agg, width, label="OSPF", color=color_ospf)

        ax5.set_title("5A. So sánh giá trị trung bình", fontsize=12, fontweight="bold")
        ax5.set_xticks(xm)
        ax5.set_xticklabels(m_labels, fontsize=9)
        ax5.legend()
        ax5.grid(axis="y", linestyle="--", alpha=0.4)

        # annotate giá trị
        for rect in list(b1) + list(b2):
            h = rect.get_height()
            if h > 0:
                ax5.annotate(f"{h:.3f}",
                            xy=(rect.get_x()+rect.get_width()/2, h),
                            xytext=(0,2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        # ─────────────────────────────────────────
        #  Biểu đồ dưới: % improvement
        # ─────────────────────────────────────────
        def improvement(dqn_vals, ospf_vals):
            res = []
            for d, o in zip(dqn_vals, ospf_vals):
                if o == 0:
                    if d == 0:
                        res.append(0.0)          # cả hai đều tốt như nhau
                    else:
                        res.append(-100.0)       # DQN tệ hơn cực mạnh
                else:
                    res.append((o - d) / o * 100.0)
            return res

        improve_vals = improvement(dqn_agg, ospf_agg)

        colors = ["green" if v >= 0 else "red" for v in improve_vals]

        bars = ax6.bar(xm, improve_vals, color=colors)

        ax6.set_title("5B. % cải thiện của DQN so với OSPF",
                    fontsize=12, fontweight="bold")
        ax6.set_ylabel("Improvement (%)")
        ax6.set_xticks(xm)
        ax6.set_xticklabels(m_labels, fontsize=9)
        ax6.grid(axis="y", linestyle="--", alpha=0.4)

        # annotate %
        for rect, val in zip(bars, improve_vals):
            ax6.annotate(f"{val:+.2f}%",
                        xy=(rect.get_x()+rect.get_width()/2, rect.get_height()),
                        xytext=(0,3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        # ─────────────────────────────────────────
        #  Title chung + save
        # ─────────────────────────────────────────
        fig.suptitle(f"DQN vs OSPF — Summary (value + improvement) (seed={DEMO_SEED})",
                    fontsize=13, fontweight="bold", y=0.98)

        chart_path5 = os.path.join(log_dir, "demo_comparison_5_summary.png")
        plt.savefig(chart_path5, dpi=150, bbox_inches="tight")
        print(f"  Chart saved: {chart_path5}")
        plt.close(fig)

        print("\n  ✓ Đã lưu 5 biểu đồ RIÊNG BIỆT vào thư mục logs/DQN/")

    except ImportError:
        print("\n  (matplotlib chưa cài — bỏ qua biểu đồ. pip install matplotlib)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["check","train","eval","demo"],
                        default="check")
    parser.add_argument("--episodes",   type=int, default=None)
    parser.add_argument("--n-pairs",    type=int, default=8,
                        help="Số bộ (src,dst,vol) ngẫu nhiên cho demo")
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
        run_demo(args.checkpoint, n_pairs=args.n_pairs)