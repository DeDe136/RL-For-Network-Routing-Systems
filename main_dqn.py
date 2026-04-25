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

    def collect_link_details(path, topo_ref, accum):
        """
        Đọc trạng thái link trên path.
        load & queue_used_cur lấy từ accum (tích lũy qua send_traffic +
        reduce_load), các field còn lại lấy trực tiếp từ topo_ref.
        """
        details = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            lk   = topo_ref.link(u, v)
            details.append({
                "src_node":       u,
                "dst_node":       v,
                "delay":          lk.delay,
                "bandwidth":      lk.bandwidth,
                "load":           accum["load"].get((u, v), lk.load),
                "utilization":    accum["load"].get((u, v), lk.load) / lk.bandwidth,
                "queue_size_cur": lk.queue_size_cur,
                "queue_used_cur": accum["queue_used_cur"].get((u, v), lk.queue_used_cur),
                "queue_util":     accum["queue_used_cur"].get((u, v), lk.queue_used_cur) / lk.queue_size_cur,
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
        """
        Dijkstra + send_traffic + reduce_load từng hop.
        Trả về (path, accum) trong đó accum tích lũy load và
        queue_used_cur trên từng link sau mỗi send_traffic + reduce_load.
        """
        full_path = topo_ref.shortest_path(src_n, dst_n)
        accum = {"load": {}, "queue_used_cur": {}}
        if len(full_path) < 2:
            return full_path, accum

        for hop_idx in range(1, len(full_path)):
            partial = full_path[:hop_idx + 1]

            # send_traffic — chỉ tác động link cuối (partial[-2] → partial[-1])
            result = topo_ref.send_traffic(partial, vol, is_first_hop=(hop_idx == 1))
            last = (partial[-2], partial[-1])
            accum["load"][last] = (
                accum["load"].get(last, 0.0) + result["load"]
            )
            accum["queue_used_cur"][last] = (
                accum["queue_used_cur"].get(last, 0.0) + result["queue_used_cur"]
            )

            # reduce_load — tác động toàn bộ partial, đọc lại topo sau khi xong
            topo_ref.reduce_load(partial)
            for i in range(len(partial) - 1):
                lk = (partial[i], partial[i + 1])
                attr = topo_ref.link(lk[0], lk[1])
                accum["load"][lk] = (
                    accum["load"].get(lk, 0.0) + attr.load
                )
                accum["queue_used_cur"][lk] = (
                    accum["queue_used_cur"].get(lk, 0.0) + attr.queue_used_cur
                )

        return full_path, accum

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

        dqn_path, dqn_accum = agent.best_path(src_n, dst_n, topo_dqn, volume_mbps=vol)
        dqn_details = collect_link_details(dqn_path, topo_dqn, dqn_accum)
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

        sp_path, sp_accum = ospf_path_with_traffic(src_n, dst_n, topo_ospf, vol)
        sp_details = collect_link_details(sp_path, topo_ospf, sp_accum)
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
        ("Avg delay (ms)",            "total_delay"),
        ("Avg utilization",           "avg_utilization"),
        ("Avg queue_used_cur (pkts)", "avg_queue_used_cur"),
        ("Avg dropped_data (pkt)",    "total_dropped_data"),
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
        import matplotlib.ticker as mticker
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import FancyBboxPatch
        import matplotlib.patheffects as pe

        # ── Bảng màu & style hiện đại ─────────────────────────────────
        plt.rcParams.update({
            "font.family":       "DejaVu Sans",
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })

        BG_DARK   = "#0F1117"
        BG_PANEL  = "#1A1D2E"
        BG_AXES   = "#12152A"
        COLOR_DQN  = "#00D4FF"   # cyan neon
        COLOR_OSPF = "#FF6B35"   # orange neon
        COLOR_GRID = "#2A2D45"
        COLOR_TEXT = "#E8EAF6"
        COLOR_SUB  = "#8B8FA8"

        MARKER_DQN  = "o"
        MARKER_OSPF = "s"
        MARKER_SIZE = 8

        def vals(results, key):
            return [r[key] for r in results]

        pair_labels = [f"{s}→{d}" for s, d, _ in pairs]
        x = np.arange(n_pairs)

        # ── Hàm style chung cho axes ──────────────────────────────────
        def style_ax(ax, title, ylabel):
            ax.set_facecolor(BG_AXES)
            ax.set_title(title, color=COLOR_TEXT, fontsize=12,
                         fontweight="bold", pad=14)
            ax.set_ylabel(ylabel, color=COLOR_SUB, fontsize=10)
            ax.tick_params(colors=COLOR_SUB, labelsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels, rotation=35, ha="right",
                               color=COLOR_SUB, fontsize=8.5)
            ax.grid(axis="y", color=COLOR_GRID, linestyle="--",
                    linewidth=0.8, alpha=0.8)
            ax.grid(axis="x", color=COLOR_GRID, linestyle=":",
                    linewidth=0.5, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)

        def add_legend(ax):
            leg = ax.legend(
                fontsize=9, framealpha=0.2,
                facecolor=BG_PANEL, edgecolor=COLOR_GRID,
                labelcolor=COLOR_TEXT,
            )

        def annotate_line(ax, x_vals, y_vals, color, fmt=".2f"):
            """Chú thích giá trị tại mỗi điểm trên đường."""
            for xi, yi in zip(x_vals, y_vals):
                ax.annotate(
                    f"{yi:{fmt}}",
                    xy=(xi, yi),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2,
                                                foreground=BG_AXES)],
                )

        def save_fig(fig, suffix):
            chart_path = os.path.join(log_dir,
                                      f"demo_comparison_{suffix}.png")
            plt.savefig(chart_path, dpi=150, bbox_inches="tight",
                        facecolor=BG_DARK)
            print(f"  Chart saved: {chart_path}")
            plt.close(fig)

        # ── Helper tạo line chart hiện đại ───────────────────────────
        def create_line_chart(dqn_v, ospf_v, title, ylabel, suffix):
            fig, ax = plt.subplots(figsize=(12, 6),
                                   facecolor=BG_DARK)
            ax.set_facecolor(BG_AXES)

            # Vùng fill giữa hai đường
            ax.fill_between(x, dqn_v, ospf_v,
                            alpha=0.08, color=COLOR_DQN)

            # Đường DQN
            ax.plot(x, dqn_v, color=COLOR_DQN, linewidth=2.5,
                    marker=MARKER_DQN, markersize=MARKER_SIZE,
                    label="DQN",
                    markerfacecolor=BG_DARK,
                    markeredgewidth=2,
                    zorder=5)

            # Đường OSPF
            ax.plot(x, ospf_v, color=COLOR_OSPF, linewidth=2.5,
                    marker=MARKER_OSPF, markersize=MARKER_SIZE,
                    label="OSPF",
                    markerfacecolor=BG_DARK,
                    markeredgewidth=2,
                    linestyle="--",
                    zorder=5)

            # Highlight điểm DQN tốt hơn / tệ hơn
            for xi, (dv, ov) in enumerate(zip(dqn_v, ospf_v)):
                clr = "#00FF9C" if dv <= ov else "#FF4C6E"
                ax.scatter([xi], [dv], color=clr, s=60,
                           zorder=6, edgecolors=BG_DARK, linewidth=1.5)

            annotate_line(ax, x, dqn_v,  COLOR_DQN)
            annotate_line(ax, x, ospf_v, COLOR_OSPF)

            style_ax(ax, title, ylabel)
            add_legend(ax)

            fig.suptitle(
                f"DQN vs OSPF — {n_pairs} bộ ngẫu nhiên  (seed={DEMO_SEED})",
                color=COLOR_TEXT, fontsize=13, fontweight="bold", y=1.01,
            )
            fig.tight_layout()
            save_fig(fig, suffix)

        # ═══════════════════════════════════════════════════════════════
        #  BIỂU ĐỒ 1 — Delay  (line)
        # ═══════════════════════════════════════════════════════════════
        create_line_chart(
            vals(dqn_results,  "total_delay"),
            vals(ospf_results, "total_delay"),
            "① Delay từng bộ (thấp hơn = tốt hơn)",
            "Delay (ms)",
            "1_delay",
        )

        # ═══════════════════════════════════════════════════════════════
        #  BIỂU ĐỒ 2 — Avg load  (line)
        # ═══════════════════════════════════════════════════════════════
        create_line_chart(
            vals(dqn_results,  "avg_load"),
            vals(ospf_results, "avg_load"),
            "② Băng thông sử dụng TB từng bộ",
            "Avg load (Mbps)",
            "2_load",
        )

        # ═══════════════════════════════════════════════════════════════
        #  BIỂU ĐỒ 3 — Avg queue_used_cur  (line)
        # ═══════════════════════════════════════════════════════════════
        create_line_chart(
            vals(dqn_results,  "avg_queue_used_cur"),
            vals(ospf_results, "avg_queue_used_cur"),
            "③ Packets đang dùng TB trong hàng đợi từng bộ (thấp hơn = tốt hơn)",
            "Avg queue_used_cur (packets)",
            "3_queue",
        )

        # ═══════════════════════════════════════════════════════════════
        #  BIỂU ĐỒ 4 — Total dropped_data  (line)
        # ═══════════════════════════════════════════════════════════════
        create_line_chart(
            vals(dqn_results,  "total_dropped_data"),
            vals(ospf_results, "total_dropped_data"),
            "④ Packets bị drop từng bộ (thấp hơn = tốt hơn)",
            "Dropped data (packets)",
            "4_dropped",
        )

        # ═══════════════════════════════════════════════════════════════
        #  BIỂU ĐỒ 5 — Summary: giá trị TB + % cải thiện  (bar + diverging)
        # ═══════════════════════════════════════════════════════════════
        metrics_summary = [
            ("Delay TB\n(ms)",          "total_delay"),
            ("Load TB\n(Mbps)",         "avg_load"),
            ("Queue used TB\n(pkts)",   "avg_queue_used_cur"),
            ("Dropped TB\n(pkts)",      "total_dropped_data"),
        ]

        m_labels = [m[0] for m in metrics_summary]
        dqn_agg  = [gmean(dqn_results,  m[1]) for m in metrics_summary]
        ospf_agg = [gmean(ospf_results, m[1]) for m in metrics_summary]

        def improvement(dqn_v, ospf_v):
            res = []
            for d, o in zip(dqn_v, ospf_v):
                if o == 0:
                    res.append(0.0 if d == 0 else -100.0)
                else:
                    res.append((o - d) / o * 100.0)
            return res

        improve_vals = improvement(dqn_agg, ospf_agg)

        fig5 = plt.figure(figsize=(13, 10), facecolor=BG_DARK)
        gs   = GridSpec(2, 1, figure=fig5, hspace=0.52,
                        top=0.93, bottom=0.08, left=0.09, right=0.97)

        ax5a = fig5.add_subplot(gs[0])
        ax5b = fig5.add_subplot(gs[1])

        # ── 5A: grouped bar với gradient feel ────────────────────────
        xm    = np.arange(len(m_labels))
        width = 0.32

        bars_dqn  = ax5a.bar(xm - width/2, dqn_agg,  width,
                             label="DQN",  color=COLOR_DQN,
                             alpha=0.85, zorder=3,
                             linewidth=0, edgecolor="none")
        bars_ospf = ax5a.bar(xm + width/2, ospf_agg, width,
                             label="OSPF", color=COLOR_OSPF,
                             alpha=0.85, zorder=3,
                             linewidth=0, edgecolor="none")

        # annotate bar
        for rect, color in (
            [(r, COLOR_DQN)  for r in bars_dqn] +
            [(r, COLOR_OSPF) for r in bars_ospf]
        ):
            h = rect.get_height()
            if h > 0:
                ax5a.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=color,
                    path_effects=[pe.withStroke(linewidth=2,
                                                foreground=BG_AXES)],
                )

        style_ax(ax5a, "⑤A  Giá trị trung bình tổng hợp (DQN vs OSPF)",
                 "Giá trị TB")
        ax5a.set_xticks(xm)
        ax5a.set_xticklabels(m_labels, fontsize=9.5, color=COLOR_SUB,
                             rotation=0)
        ax5a.set_xlim(-0.6, len(m_labels) - 0.4)
        add_legend(ax5a)

        # ── 5B: diverging bar % cải thiện ────────────────────────────
        ax5b.set_facecolor(BG_AXES)
        ax5b.axhline(0, color=COLOR_TEXT, linewidth=1, alpha=0.5)

        bar_colors = ["#00FF9C" if v >= 0 else "#FF4C6E"
                      for v in improve_vals]
        bars_imp = ax5b.bar(xm, improve_vals, width=0.45,
                            color=bar_colors, alpha=0.88,
                            zorder=3, linewidth=0)

        # annotate %
        for rect, val in zip(bars_imp, improve_vals):
            vert_offset = 4 if val >= 0 else -14
            ax5b.annotate(
                f"{val:+.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, val),
                xytext=(0, vert_offset), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold",
                color="#00FF9C" if val >= 0 else "#FF4C6E",
                path_effects=[pe.withStroke(linewidth=2,
                                            foreground=BG_AXES)],
            )

        style_ax(ax5b,
                 "⑤B  % cải thiện của DQN so với OSPF  (dương = DQN tốt hơn)",
                 "Improvement (%)")
        ax5b.set_xticks(xm)
        ax5b.set_xticklabels(m_labels, fontsize=9.5, color=COLOR_SUB,
                             rotation=0)
        ax5b.set_xlim(-0.6, len(m_labels) - 0.4)
        ax5b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))
        for spine in ax5b.spines.values():
            spine.set_visible(False)

        fig5.suptitle(
            f"DQN vs OSPF — Summary  (seed={DEMO_SEED})",
            color=COLOR_TEXT, fontsize=14, fontweight="bold",
        )
        save_fig(fig5, "5_summary")

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